import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  Download,
  Loader2,
  RefreshCw,
  Sparkles,
  WifiOff,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { AgentTimeline, timelineSummary } from "@/components/agent-timeline";
import { LazyMarkdown as Markdown } from "@/components/lazy-markdown";
import { ReviewPanel } from "@/components/review-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useRun } from "@/hooks/use-run";
import type { RunStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

type PillState = RunStatus | "pending";

const PILL: Record<PillState, { label: string; variant: "secondary" | "info" | "warning" | "success" | "destructive" }> = {
  pending: { label: "pending", variant: "secondary" },
  running: { label: "running", variant: "info" },
  awaiting_review: { label: "awaiting review", variant: "warning" },
  done: { label: "done", variant: "success" },
  failed: { label: "failed", variant: "destructive" },
};

export function StatusPill({ state }: { state: PillState }) {
  const { label, variant } = PILL[state];
  return (
    <Badge variant={variant} className="gap-1.5 rounded-full px-2.5">
      {(state === "pending" || state === "running") && <Loader2 className="animate-spin" />}
      {label}
    </Badge>
  );
}

export function QuestionBubble({ question }: { question: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm leading-relaxed text-primary-foreground shadow-sm">
        {question}
      </div>
    </div>
  );
}

interface RunCardProps {
  threadId: string;
  question: string | null;
  onRemove: () => void;
  onQuestion: (question: string) => void;
}

export function RunCard({ threadId, question, onRemove, onQuestion }: RunCardProps) {
  const label = question ?? `Run ${threadId}`;
  const { info, error, reconnecting, reviewing, retry, review } = useRun(threadId, label);
  const [showSteps, setShowSteps] = useState(false);

  // A run restored from the URL arrives without its question; learn it once known.
  const serverQuestion = info?.question ?? info?.review?.question;
  useEffect(() => {
    if (!question && serverQuestion) onQuestion(serverQuestion);
  }, [question, serverQuestion, onQuestion]);

  const status: PillState = info?.status ?? "pending";
  const stages = info?.stages ?? [];
  const live = status === "running";

  return (
    <section className="space-y-3 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
      {/* A run reopened from a #run= link while it's executing has no question
          yet: the API's "running" answer doesn't carry it. It fills in later. */}
      <QuestionBubble question={question ?? serverQuestion ?? `Reconnected run ${threadId}`} />

      <Card className="gap-0 overflow-hidden py-0">
        <CardHeader className="flex flex-row items-center gap-2 border-b bg-muted/30 px-5 py-3">
          <span className="flex size-6 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Sparkles className="size-3.5" />
          </span>
          <span className="text-sm font-medium">Analyst</span>
          <StatusPill state={error ? "failed" : status} />
          {reconnecting && (
            <Badge variant="outline" className="gap-1 text-muted-foreground">
              <WifiOff /> reconnecting
            </Badge>
          )}
          <span className="ml-auto hidden font-mono text-[0.7rem] text-muted-foreground sm:inline">{threadId}</span>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onRemove} aria-label="Remove from thread">
                <X />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Remove from this thread (the run stays on the server)</TooltipContent>
          </Tooltip>
        </CardHeader>

        <CardContent className="px-5 py-5">
          {error ? (
            <ErrorState kind={error.kind} message={error.message} onRetry={retry} onRemove={onRemove} />
          ) : !info ? (
            <LoadingState />
          ) : (
            <div className="space-y-5">
              {/* Live while working; folded away once there's a result to read. */}
              {live || status === "failed" ? (
                <AgentTimeline stages={stages} status={info.status} />
              ) : (
                stages.length > 0 && (
                  <div>
                    <button
                      type="button"
                      onClick={() => setShowSteps((s) => !s)}
                      className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <ChevronDown className={cn("size-3.5 transition-transform", showSteps && "rotate-180")} />
                      Agent steps · {timelineSummary(stages)}
                    </button>
                    {showSteps && (
                      <div className="mt-4">
                        <AgentTimeline stages={stages} status={info.status} />
                      </div>
                    )}
                  </div>
                )
              )}

              {live && <AnswerSkeleton />}

              {status === "awaiting_review" && info.review && (
                <>
                  {stages.length > 0 && <Separator />}
                  <ReviewPanel threadId={threadId} review={info.review} busy={reviewing} onDecision={review} />
                </>
              )}

              {status === "done" && <Report threadId={threadId} markdown={info.report} hasSteps={stages.length > 0} />}

              {status === "failed" && (
                <div className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
                  <div>
                    <p className="font-medium text-destructive">The run failed</p>
                    <p className="mt-1 text-muted-foreground">{info.error ?? "Unknown error"}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}

function Report({ threadId, markdown, hasSteps }: { threadId: string; markdown: string | null; hasSteps: boolean }) {
  const [copied, setCopied] = useState(false);
  const text = markdown ?? "";

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success("Report copied as Markdown");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard needs a secure context (https or localhost) and permission.
      toast.error("Couldn't access the clipboard");
    }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `report-${threadId}.md` });
    a.click();
    URL.revokeObjectURL(url);
  }

  if (!text) return <p className="text-sm text-muted-foreground">The run finished without a report.</p>;
  return (
    <>
      {hasSteps && <Separator />}
      <div className="relative">
        <div className="absolute top-0 right-0 flex gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="icon-sm" onClick={copy} aria-label="Copy report">
                {copied ? <Check className="text-success" /> : <Copy />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Copy Markdown</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="icon-sm" onClick={download} aria-label="Download report">
                <Download />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Download .md</TooltipContent>
          </Tooltip>
        </div>
        <Markdown idPrefix={threadId} className="pr-20">
          {text}
        </Markdown>
      </div>
    </>
  );
}

function LoadingState() {
  return (
    <div className="space-y-4" aria-label="Loading run">
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex items-center gap-3">
          <Skeleton className="size-8 rounded-full" />
          <div className="flex-1 space-y-1.5">
            <Skeleton className="h-3.5 w-32" />
            <Skeleton className="h-3 w-56" />
          </div>
        </div>
      ))}
    </div>
  );
}

function AnswerSkeleton() {
  return (
    <div className="space-y-2.5 rounded-lg border border-dashed p-4" aria-hidden>
      <Skeleton className="h-4 w-2/5" />
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-11/12" />
      <Skeleton className="h-3 w-4/5" />
      <p className="pt-1 text-xs text-muted-foreground">
        The report appears here. A full run makes 10–15 model calls and takes a few minutes.
      </p>
    </div>
  );
}

function ErrorState({
  kind,
  message,
  onRetry,
  onRemove,
}: {
  kind: "not_found" | "network" | "other";
  message: string;
  onRetry: () => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex flex-col items-start gap-3 text-sm sm:flex-row sm:items-center">
      <AlertTriangle className="size-4 shrink-0 text-destructive" />
      <p className="flex-1 text-muted-foreground">
        {kind === "network" ? `Lost contact with the API: ${message}` : message}
      </p>
      {kind === "not_found" ? (
        <Button variant="outline" size="sm" onClick={onRemove}>
          Remove
        </Button>
      ) : (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw /> Retry
        </Button>
      )}
    </div>
  );
}
