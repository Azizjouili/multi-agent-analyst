import { lazy, Suspense, type ComponentProps } from "react";

import { Skeleton } from "@/components/ui/skeleton";

// The Markdown pipeline (react-markdown, micromark, syntax highlighting) is
// most of the app's JavaScript, and nothing needs it until a draft or report
// is on screen. Loading it on first use keeps the first page load small;
// Vite emits it as its own hashed chunk under /assets.
const Markdown = lazy(() => import("@/components/markdown").then((m) => ({ default: m.Markdown })));

export function LazyMarkdown(props: ComponentProps<typeof Markdown>) {
  return (
    <Suspense
      fallback={
        <div className="space-y-2.5" aria-label="Loading report">
          <Skeleton className="h-5 w-1/3" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
        </div>
      }
    >
      <Markdown {...props} />
    </Suspense>
  );
}
