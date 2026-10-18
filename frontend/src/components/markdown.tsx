import { memo, useMemo, type ComponentProps } from "react";
import type { Element, ElementContent, Root, RootContent } from "hast";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { rehypeCodeHighlight } from "@/lib/highlight";
import { cn } from "@/lib/utils";

// Sanitization: the report is written by an LLM from fetched web pages, so it
// is untrusted text. react-markdown never turns it into raw HTML: it builds
// React elements from a syntax tree, and HTML inside the Markdown is dropped
// (we don't add rehype-raw). Its default urlTransform also strips
// `javascript:` and other unsafe link targets. So there is no innerHTML, and
// nothing to inject script with; that's the same guarantee marked + DOMPurify
// gave the old page, by construction rather than by cleaning up afterwards.

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** "[1, 3]" in the text -> small badges linking to the matching Sources entry.
 *  Done on the HTML tree (not with a regex on the Markdown) so citations in
 *  code blocks, inline code and existing links stay as they are. */
function rehypeCitations(options: { prefix: string }) {
  return (tree: Root) => {
    markSources(tree, options.prefix);
    walk(tree, options.prefix);
  };
}

function walk(node: Root | Element, prefix: string) {
  const out: (RootContent | ElementContent)[] = [];
  for (const child of node.children) {
    if (child.type === "text") {
      out.push(...splitCitations(child.value, prefix));
    } else {
      if (child.type === "element" && !["code", "pre", "a"].includes(child.tagName)) {
        walk(child, prefix);
      }
      out.push(child);
    }
  }
  node.children = out as typeof node.children;
}

function splitCitations(text: string, prefix: string): ElementContent[] {
  const parts: ElementContent[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > last) parts.push({ type: "text", value: text.slice(last, match.index) });
    const numbers = match[1].split(",").map((n) => n.trim());
    parts.push({
      type: "element",
      tagName: "span",
      properties: { className: ["citations"] },
      children: numbers.map((n) => ({
        type: "element",
        tagName: "a",
        properties: { href: `#${prefix}-src-${n}`, className: ["citation"], dataCitation: n },
        children: [{ type: "text", value: n }],
      })),
    });
    last = match.index + match[0].length;
  }
  if (last === 0) return [{ type: "text", value: text }];
  if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
  return parts;
}

/** Give the list after "## Sources" anchor ids, so citation badges can link to it.
 *  Ids carry the run's prefix: several reports share the page. */
function markSources(tree: Root, prefix: string) {
  const nodes = tree.children;
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (node.type !== "element" || node.tagName !== "h2" || textOf(node).trim() !== "Sources") continue;
    const list = nodes.slice(i + 1).find((n): n is Element => n.type === "element");
    if (list?.tagName !== "ol") return;
    list.properties = { ...list.properties, className: ["sources"] };
    const start = Number(list.properties.start ?? 1);
    list.children
      .filter((c): c is Element => c.type === "element" && c.tagName === "li")
      .forEach((li, k) => (li.properties = { ...li.properties, id: `${prefix}-src-${start + k}` }));
    return;
  }
}

function textOf(node: ElementContent | Element): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(textOf).join("");
  return "";
}

/** Sources entry text by number, for the citation tooltips. */
function parseSources(markdown: string): Map<string, string> {
  const map = new Map<string, string>();
  const section = markdown.split(/^##\s+Sources\s*$/m)[1];
  if (!section) return map;
  for (const line of section.split("\n")) {
    const m = line.match(/^\s*(\d+)\.\s+(.*)$/);
    if (m) map.set(m[1], m[2].replace(/\*\*/g, ""));
  }
  return map;
}

function CitationBadge({ n, source, href }: { n: string; source?: string; href?: string }) {
  const badge = (
    <a
      href={href}
      className="citation mx-px inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded-[5px] bg-primary/10 px-1 align-middle font-mono text-[0.68rem] font-semibold text-primary no-underline transition-colors hover:bg-primary hover:text-primary-foreground"
      aria-label={`Source ${n}`}
    >
      {n}
    </a>
  );
  if (!source) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent side="top" className="text-left">
        <span className="font-semibold">[{n}]</span> {source}
      </TooltipContent>
    </Tooltip>
  );
}

interface MarkdownProps {
  children: string;
  /** Unique per report on the page, for the Sources anchors. */
  idPrefix: string;
  className?: string;
}

export const Markdown = memo(function Markdown({ children, idPrefix, className }: MarkdownProps) {
  const sources = useMemo(() => parseSources(children), [children]);

  const components = useMemo<Components>(
    () => ({
      a: ({ node: _node, ...props }: ComponentProps<"a"> & { node?: unknown }) => {
        const n = (props as Record<string, unknown>)["data-citation"] as string | undefined;
        if (n) return <CitationBadge n={n} source={sources.get(n)} href={props.href} />;
        // External sources: new tab, and no handle back to this page.
        const external = props.href?.startsWith("http");
        return (
          <a {...props} {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})} />
        );
      },
    }),
    [sources],
  );

  return (
    <div className={cn("report", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeCodeHighlight, [rehypeCitations, { prefix: idPrefix }]]}
        components={components}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
});
