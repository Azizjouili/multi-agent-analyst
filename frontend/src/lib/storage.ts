// localStorage can be missing or throw (private windows, blocked site data),
// and saved JSON can be stale or hand-edited. Every access goes through here,
// so the app always works without it; storage only adds conveniences.

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable or full: nothing to do */
  }
}
