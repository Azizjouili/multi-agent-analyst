import { forwardRef, useState } from "react";
import { ArrowUp, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// Same bounds as the API's StartRun model, so the button is disabled exactly
// when the server would answer 422.
const MIN = 3;
const MAX = 1000;

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => void;
  busy: boolean;
}

export const Composer = forwardRef<HTMLTextAreaElement, ComposerProps>(function Composer(
  { value, onChange, onSubmit, busy },
  ref,
) {
  const [focused, setFocused] = useState(false);
  const question = value.trim();
  const canSend = !busy && question.length >= MIN;
  const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

  function submit() {
    if (canSend) onSubmit(question);
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className={cn(
        "rounded-2xl border bg-card p-2 shadow-lg shadow-black/[0.03] transition-[box-shadow,border-color] dark:shadow-black/20",
        focused && "border-ring/60 ring-[3px] ring-ring/15",
      )}
    >
      <Textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            submit();
          }
        }}
        maxLength={MAX}
        rows={2}
        placeholder="Ask about Apple, Microsoft, NVIDIA, Amazon, Alphabet, Meta, Tesla, Walmart…"
        aria-label="Your question"
        id="question"
        className="max-h-48 min-h-14 resize-none border-0 bg-transparent px-3 shadow-none focus-visible:ring-0 dark:bg-transparent"
      />
      <div className="flex items-center gap-2 px-2 pt-1">
        <span className="text-xs text-muted-foreground">
          <kbd className="rounded border bg-muted px-1 font-mono text-[0.7rem]">{isMac ? "⌘" : "Ctrl"}</kbd>{" "}
          + <kbd className="rounded border bg-muted px-1 font-mono text-[0.7rem]">Enter</kbd> to send
        </span>
        {value.length > MAX * 0.8 && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {value.length}/{MAX}
          </span>
        )}
        <Button type="submit" size="sm" className="ml-auto rounded-full px-4" disabled={!canSend}>
          {busy ? <Loader2 className="animate-spin" /> : <ArrowUp />}
          Ask
        </Button>
      </div>
    </form>
  );
});
