import { useState } from "react";
import { Check, Loader2, MessageSquareReply, ShieldCheck } from "lucide-react";

import { LazyMarkdown as Markdown } from "@/components/lazy-markdown";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import type { ReviewPayload } from "@/lib/api";

const VERDICT_VARIANT = { accept: "success", revise: "warning", research: "info" } as const;

interface ReviewPanelProps {
  threadId: string;
  review: ReviewPayload;
  busy: boolean;
  onDecision: (approved: boolean, feedback?: string) => void;
}

export function ReviewPanel({ threadId, review, busy, onDecision }: ReviewPanelProps) {
  const [feedback, setFeedback] = useState("");
  const [triedEmpty, setTriedEmpty] = useState(false);
  const critique = review.critique;
  const points = Array.isArray(critique?.feedback) ? critique.feedback : critique?.feedback ? [critique.feedback] : [];

  function sendFeedback() {
    if (!feedback.trim()) {
      // The writer needs something to act on; an empty rejection helps no one.
      setTriedEmpty(true);
      return;
    }
    onDecision(false, feedback.trim());
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <ShieldCheck className="size-4 text-warning" />
        <span className="font-medium">Your review</span>
        {critique && (
          <Badge variant={VERDICT_VARIANT[critique.verdict] ?? "secondary"}>critic: {critique.verdict}</Badge>
        )}
        {critique?.scores &&
          Object.entries(critique.scores).map(([name, score]) => (
            <Badge key={name} variant="outline" className="font-normal text-muted-foreground">
              {name} <span className="font-mono font-semibold text-foreground">{score}/5</span>
            </Badge>
          ))}
        {!!review.revision_count && (
          <span className="text-xs text-muted-foreground">
            {review.revision_count} revision{review.revision_count > 1 ? "s" : ""}
          </span>
        )}
      </div>

      {points.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
          {points.slice(0, 4).map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}

      {/* `preview` is the draft exactly as the final report will read
          (numbered citations + Sources): you approve what the reader gets. */}
      <ScrollArea className="h-[26rem] rounded-lg border bg-muted/30">
        <div className="p-5">
          <Markdown idPrefix={`${threadId}-draft`}>{review.preview || review.draft || ""}</Markdown>
        </div>
      </ScrollArea>

      <Separator />

      <div className="space-y-2">
        <label htmlFor={`feedback-${threadId}`} className="text-sm font-medium">
          Feedback for the writer
        </label>
        <Textarea
          id={`feedback-${threadId}`}
          value={feedback}
          onChange={(e) => {
            setFeedback(e.target.value);
            setTriedEmpty(false);
          }}
          placeholder="e.g. Add more detail on supplier concentration risk"
          maxLength={2000}
          disabled={busy}
          aria-invalid={triedEmpty || undefined}
          className="min-h-20"
        />
        {triedEmpty && <p className="text-xs text-destructive">Say what should change before sending feedback.</p>}
      </div>

      <div className="flex flex-wrap justify-end gap-2">
        <Button variant="outline" onClick={sendFeedback} disabled={busy}>
          <MessageSquareReply /> Send feedback
        </Button>
        <Button variant="success" onClick={() => onDecision(true)} disabled={busy}>
          {busy ? <Loader2 className="animate-spin" /> : <Check />} Approve
        </Button>
      </div>
    </div>
  );
}
