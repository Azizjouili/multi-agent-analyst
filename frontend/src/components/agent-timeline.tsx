import {
  CheckCircle2,
  Circle,
  Database,
  FileCheck2,
  FileText,
  Globe,
  Loader2,
  type LucideIcon,
  Network,
  PauseCircle,
  PenLine,
  ScanSearch,
  UserCheck,
  Users,
  XCircle,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useNow } from "@/hooks/use-run";
import type { RunStatus, Stage } from "@/lib/api";
import { buildTimeline, nodeInfo, type TimelineItem, type TimelineStatus } from "@/lib/stages";
import { cn, formatDuration } from "@/lib/utils";

const ICONS: Record<string, LucideIcon> = {
  supervisor: Network,
  rag: FileText,
  data_analysis: Database,
  web_research: Globe,
  specialists: Users,
  writer: PenLine,
  critic: ScanSearch,
  human_gate: UserCheck,
  finalize: FileCheck2,
};

function StatusIcon({ status, className }: { status: TimelineStatus; className?: string }) {
  const base = cn("size-4 shrink-0", className);
  switch (status) {
    case "running":
      return <Loader2 className={cn(base, "animate-spin text-primary")} aria-label="running" />;
    case "done":
      return <CheckCircle2 className={cn(base, "text-success")} aria-label="done" />;
    case "waiting":
      return <PauseCircle className={cn(base, "text-warning")} aria-label="waiting for you" />;
    case "failed":
      return <XCircle className={cn(base, "text-destructive")} aria-label="failed" />;
    default:
      return <Circle className={cn(base, "text-muted-foreground/40")} aria-label="pending" />;
  }
}

function duration(start: number | null, end: number | null, now: number) {
  if (start === null) return null;
  return formatDuration((end ?? now) - start);
}

function Row({ item, now, last }: { item: TimelineItem; now: number; last: boolean }) {
  const Icon = ICONS[item.node] ?? Circle;
  const info = nodeInfo(item.node);
  const pending = item.status === "pending";
  const parallel = item.stages.length > 1;
  const time = item.status === "waiting" ? null : duration(item.startedAt, item.finishedAt, now);

  return (
    <li className="relative flex gap-3 pb-4 last:pb-0">
      {/* The rail connecting the steps; solid once a step has run. */}
      {!last && (
        <span
          aria-hidden
          className={cn(
            "absolute top-8 bottom-0 left-[15px] w-px",
            pending ? "bg-border" : "bg-gradient-to-b from-primary/40 to-border",
          )}
        />
      )}
      <span
        className={cn(
          "relative z-10 flex size-8 shrink-0 items-center justify-center rounded-full border bg-card transition-colors",
          item.status === "running" && "border-primary/50 bg-primary/10 text-primary shadow-[0_0_0_4px] shadow-primary/10",
          item.status === "done" && "text-foreground",
          item.status === "waiting" && "border-warning/50 bg-warning/10 text-warning",
          item.status === "failed" && "border-destructive/50 bg-destructive/10 text-destructive",
          pending && "border-dashed text-muted-foreground/50",
        )}
      >
        <Icon className="size-4" />
      </span>

      <div className="min-w-0 flex-1 pt-1">
        <div className="flex items-center gap-2">
          <span className={cn("text-sm font-medium", pending && "text-muted-foreground")}>{info.label}</span>
          {parallel && (
            <Badge variant="secondary" className="h-5 px-1.5 text-[0.65rem] uppercase tracking-wide">
              parallel
            </Badge>
          )}
          <StatusIcon status={item.status} className="ml-auto" />
          {time && <span className="w-14 text-right font-mono text-xs tabular-nums text-muted-foreground">{time}</span>}
        </div>
        {!parallel && <p className="text-xs text-muted-foreground">{item.status === "waiting" ? "Waiting for your decision" : info.detail}</p>}

        {parallel && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {item.stages.map((stage) => (
              <SpecialistChip key={`${stage.node}-${stage.started_at}`} stage={stage} now={now} />
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

function SpecialistChip({ stage, now }: { stage: Stage; now: number }) {
  const Icon = ICONS[stage.node] ?? Circle;
  const info = nodeInfo(stage.node);
  const time = duration(Date.parse(stage.started_at), stage.finished_at ? Date.parse(stage.finished_at) : null, now);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border bg-background px-2.5 py-1 text-xs",
            stage.status === "running" && "border-primary/40 text-primary",
          )}
        >
          <Icon className="size-3.5" />
          {info.label}
          <StatusIcon status={stage.status} className="size-3.5" />
          {time && <span className="font-mono tabular-nums text-muted-foreground">{time}</span>}
        </span>
      </TooltipTrigger>
      <TooltipContent>{info.detail}</TooltipContent>
    </Tooltip>
  );
}

export function AgentTimeline({ stages, status }: { stages: Stage[]; status: RunStatus }) {
  const items = buildTimeline(stages, status);
  const now = useNow(items.some((i) => i.status === "running"));
  return (
    <ol aria-label="Agent progress" className="relative">
      {items.map((item, i) => (
        <Row key={`${item.node}-${i}`} item={item} now={now} last={i === items.length - 1} />
      ))}
    </ol>
  );
}

/** One line for a finished run: "8 steps · 1m 42s". */
export function timelineSummary(stages: Stage[]): string | null {
  if (!stages.length) return null;
  const start = Math.min(...stages.map((s) => Date.parse(s.started_at)));
  const ends = stages.map((s) => (s.finished_at ? Date.parse(s.finished_at) : 0));
  return `${stages.length} steps · ${formatDuration(Math.max(...ends) - start)}`;
}
