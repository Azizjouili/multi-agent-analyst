import type { RunStatus, Stage } from "@/lib/api";

// The graph's node names (src/analyst/graph.py) and how the timeline shows them.
export const SPECIALISTS = ["rag", "data_analysis", "web_research"] as const;

export const NODE_INFO: Record<string, { label: string; detail: string }> = {
  supervisor: { label: "Supervisor", detail: "Plans the research and assigns sub-tasks" },
  rag: { label: "Documents", detail: "Hybrid search over the annual reports" },
  data_analysis: { label: "SQL analysis", detail: "Queries the company database" },
  web_research: { label: "Web research", detail: "Searches and reads web pages" },
  specialists: { label: "Specialists", detail: "Documents, SQL and web research in parallel" },
  writer: { label: "Writer", detail: "Drafts the report with citations" },
  critic: { label: "Critic", detail: "Checks citations and judges the draft" },
  human_gate: { label: "Human review", detail: "Waits for your approval" },
  finalize: { label: "Finalize", detail: "Numbers citations and adds the Sources list" },
};

export function nodeInfo(node: string) {
  return NODE_INFO[node] ?? { label: node, detail: "" };
}

// The usual path through the graph. Loops (critic -> writer, critic ->
// supervisor) only ever go *back* along it, so "what comes next" is always
// the phases after the last one that started.
const PHASES = ["supervisor", "specialists", "writer", "critic", "human_gate", "finalize"];

function phaseOf(node: string): number {
  return PHASES.indexOf((SPECIALISTS as readonly string[]).includes(node) ? "specialists" : node);
}

export type TimelineStatus = Stage["status"] | "pending";

export interface TimelineItem {
  stages: Stage[]; // >1 when nodes ran in parallel (the specialists)
  node: string; // the node, or "specialists" for a parallel group / placeholder
  status: TimelineStatus;
  startedAt: number | null;
  finishedAt: number | null;
}

/** Recorded stages grouped into timeline rows, then the phases still to come. */
export function buildTimeline(stages: Stage[], runStatus: RunStatus): TimelineItem[] {
  const items: TimelineItem[] = [];
  for (const stage of stages) {
    const prev = items.at(-1);
    // Same superstep = parallel tasks (LangGraph runs a step's tasks together).
    if (prev && stage.step !== null && prev.stages[0].step === stage.step) {
      prev.stages.push(stage);
      prev.node = "specialists";
    } else {
      items.push({ stages: [stage], node: stage.node, status: stage.status, startedAt: null, finishedAt: null });
    }
  }
  for (const item of items) {
    const statuses = item.stages.map((s) => s.status);
    item.status = statuses.includes("failed")
      ? "failed"
      : statuses.includes("running")
        ? "running"
        : statuses.includes("waiting")
          ? "waiting"
          : "done";
    item.startedAt = Math.min(...item.stages.map((s) => Date.parse(s.started_at)));
    const ends = item.stages.map((s) => (s.finished_at ? Date.parse(s.finished_at) : NaN));
    item.finishedAt = ends.some(Number.isNaN) ? null : Math.max(...ends);
  }

  // Placeholders only while the run is still going somewhere.
  if (runStatus === "running" || runStatus === "awaiting_review") {
    const last = items.at(-1);
    const from = last ? phaseOf(last.node) + 1 : 0;
    for (const phase of PHASES.slice(Math.max(from, 0))) {
      items.push({ stages: [], node: phase, status: "pending", startedAt: null, finishedAt: null });
    }
  }
  return items;
}
