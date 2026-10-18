"""Fetch a web page safely and return its main text.

Every URL passes these checks before and while fetching:
  1. scheme     - http/https only (no file://, ftp://, ...)
  2. SSRF guard - the host must resolve to a public IP address. The agent
                  fetches URLs taken from search results, and once deployed the
                  server sits inside a network: "http://169.254.169.254/" (cloud
                  metadata, which can expose credentials) or "http://localhost:8000"
                  must never be reachable through it.
  3. robots.txt - honour the site's crawling rules (RFC 9309), cached per host
  4. redirects  - followed manually, so every hop goes through checks 1-3; a
                  public URL redirecting to an internal one is a classic bypass
  5. response   - HTML or plain text only, at most `max_bytes` downloaded,
                  a hard timeout, then at most `max_chars` of extracted text

Known gap: DNS rebinding (the name resolves to a public IP during our check
and a private one at connect time). Closing it means pinning the resolved IP
for the connection, which is worth doing in production.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from analyst.tools.html_text import html_to_text

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
MAX_REDIRECTS = 5

Resolver = Callable[[str], list[str]]


class FetchError(RuntimeError):
    """The page couldn't or shouldn't be fetched. The message says why.

    `blocked` is True when the *site* refused us (401/403/429, or robots.txt).
    Such refusals usually apply to the whole host, so callers can skip that
    host's other URLs instead of trying them one by one.
    """

    def __init__(self, message: str, *, blocked: bool = False):
        super().__init__(message)
        self.blocked = blocked


@dataclass(frozen=True)
class FetchedPage:
    url: str  # final URL, after redirects
    title: str
    text: str
    truncated: bool  # True if the text was cut at max_chars


def resolve_host(host: str) -> list[str]:
    """All IP addresses a hostname resolves to (the default, real resolver)."""
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise FetchError(f"cannot resolve host {host!r}") from exc


class PageFetcher:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        user_agent: str = "MultiAgentAnalyst/0.1",
        timeout_seconds: float = 10.0,
        max_bytes: int = 2_000_000,
        max_chars: int = 30_000,
        respect_robots: bool = True,
        resolver: Resolver = resolve_host,
    ):
        # follow_redirects=False on purpose: we follow them ourselves so each
        # hop is validated (see check 4 above).
        self.client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=False)
        self.client.headers["User-Agent"] = user_agent
        self.user_agent = user_agent
        self.timeout = timeout_seconds
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.respect_robots = respect_robots
        self.resolver = resolver
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._robots_lock = threading.Lock()  # pages are fetched in parallel threads

    # -- public ----------------------------------------------------------------
    def fetch(self, url: str) -> FetchedPage:
        for _ in range(MAX_REDIRECTS + 1):
            self._check_url(url)
            try:
                with self.client.stream("GET", url, timeout=self.timeout) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError(f"redirect without a Location header from {url}")
                        url = urljoin(url, location)  # Location may be relative
                        continue
                    return self._read(url, response)
            except httpx.TimeoutException as exc:
                raise FetchError(f"timed out after {self.timeout:g}s: {url}") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"request failed for {url}: {exc}") from exc
        raise FetchError(f"too many redirects (> {MAX_REDIRECTS})")

    # -- checks 1-3 --------------------------------------------------------------
    def _check_url(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise FetchError(f"only http(s) URLs are allowed, got {parts.scheme or 'none'!r}")
        if not parts.hostname:
            raise FetchError(f"URL has no host: {url}")
        for ip in self._resolve(parts.hostname):
            addr = ipaddress.ip_address(ip)
            # is_global is False for private, loopback, link-local (incl.
            # 169.254.169.254), reserved and other non-public ranges.
            if not addr.is_global:
                raise FetchError(f"refusing to fetch non-public address {ip} ({parts.hostname})")
        if self.respect_robots and not self._robots_allows(url):
            raise FetchError(f"disallowed by robots.txt: {url}", blocked=True)

    def _resolve(self, host: str) -> list[str]:
        try:
            return [str(ipaddress.ip_address(host))]  # already an IP literal
        except ValueError:
            return self.resolver(host)

    def _robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            parser = self._robots.get(origin)
        if parser is None:
            parser = self._load_robots(origin)
            with self._robots_lock:
                self._robots[origin] = parser
        return parser.can_fetch(self.user_agent, url)

    def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser:
        """Fetch and parse robots.txt, following RFC 9309 for failures:
        4xx (none published) -> everything allowed; 5xx or unreachable -> nothing allowed."""
        parser = urllib.robotparser.RobotFileParser()
        try:
            response = self.client.get(f"{origin}/robots.txt", timeout=self.timeout)
        except httpx.HTTPError:
            parser.disallow_all = True
            return parser
        if response.status_code >= 500:
            parser.disallow_all = True
        elif response.status_code >= 400:
            parser.allow_all = True
        else:
            parser.parse(response.text.splitlines())
        return parser

    # -- check 5 -------------------------------------------------------------------
    def _read(self, url: str, response: httpx.Response) -> FetchedPage:
        if response.status_code >= 400:
            refused = response.status_code in (401, 403, 429)
            raise FetchError(f"HTTP {response.status_code} for {url}", blocked=refused)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            # Reject before downloading the body: no point pulling a 50 MB PDF or video.
            raise FetchError(f"unsupported content type {content_type or 'unknown'!r}: {url}")

        # Stream and stop at max_bytes instead of buffering whatever the server sends.
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) >= self.max_bytes:
                del body[self.max_bytes :]
                break
        raw = body.decode(response.encoding or "utf-8", errors="replace")

        if content_type == "text/plain":
            title, text = "", raw.strip()
        else:
            page = html_to_text(raw)
            title, text = page.title, page.text
        truncated = len(text) > self.max_chars
        return FetchedPage(url=url, title=title, text=text[: self.max_chars], truncated=truncated)
