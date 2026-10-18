import { useCallback, useSyncExternalStore } from "react";

export type Theme = "light" | "dark";

// Same key as the pre-paint script in index.html, stored as a plain string
// (not JSON) so that script can read it without parsing.
const KEY = "analyst.theme";

const listeners = new Set<() => void>();

function current(): Theme {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

function apply(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* not persisted, but still applied for this visit */
  }
  listeners.forEach((notify) => notify());
}

function subscribe(notify: () => void) {
  listeners.add(notify);
  return () => listeners.delete(notify);
}

/** The theme lives on <html class="dark">: one source of truth that the
 *  pre-paint script, Tailwind's dark variant and every hook user share. */
export function useTheme() {
  const theme = useSyncExternalStore(subscribe, current);
  const toggle = useCallback(() => apply(current() === "dark" ? "light" : "dark"), []);
  return { theme, toggle };
}
