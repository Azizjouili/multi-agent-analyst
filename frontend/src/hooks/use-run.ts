import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { api, ApiError, type RunInfo } from "@/lib/api";

// Why polling and not WebSockets/SSE: the API already has GET /runs/{id}, a
// run takes minutes, and the timeline changes a few times a minute, so a 2 s
// poll looks live without any new server code. It also survives API restarts
// and proxies with no reconnect logic: the next poll just works, because the
// server reads run state from its checkpoint DB.
const POLL_MS = 2000;
// Ride out short outages (server restarting, Wi-Fi blip) before giving up.
const MAX_RETRIES = 6;

export type RunError = { kind: "not_found" | "network" | "other"; message: string };

export function useRun(threadId: string, label: string) {
  const [info, setInfo] = useState<RunInfo | null>(null);
  const [error, setError] = useState<RunError | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  // Bumping this restarts the polling loop (after a review, or "Retry").
  const [generation, setGeneration] = useState(0);
  const lastStatus = useRef<RunInfo["status"] | null>(null);
  // A ref, not an effect dependency: learning the question later (a run
  // restored from the URL) must not restart the polling loop.
  const labelRef = useRef(label);
  labelRef.current = label;

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    async function tick() {
      try {
        const next = await api.getRun(threadId);
        if (cancelled) return;
        if (failures > 0) toast.success("Reconnected to the API", { id: `net-${threadId}` });
        failures = 0;
        setReconnecting(false);
        setError(null);
        setInfo(next);
        announce(lastStatus.current, next.status, labelRef.current);
        lastStatus.current = next.status;
        // Keep polling only while the server is working. awaiting_review
        // waits on the user and done/failed are final: polling would only
        // repeat the same answer.
        if (next.status === "running") timer = setTimeout(tick, POLL_MS);
      } catch (err) {
        if (cancelled) return;
        const e = err instanceof ApiError ? err : new ApiError(String(err), null);
        if (e.status === 404) {
          setError({ kind: "not_found", message: "This run no longer exists on the server." });
        } else if (e.retryable && failures < MAX_RETRIES) {
          failures += 1;
          setReconnecting(true);
          // One toast per outage (same id), not one per failed poll.
          if (failures === 1) toast.error(`${e.message} Retrying...`, { id: `net-${threadId}` });
          timer = setTimeout(tick, POLL_MS * failures); // back off a little each time
        } else {
          setReconnecting(false);
          setError({ kind: e.retryable ? "network" : "other", message: e.message });
        }
      }
    }

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [threadId, generation]);

  const retry = useCallback(() => {
    setError(null);
    setGeneration((g) => g + 1);
  }, []);

  const review = useCallback(
    async (approved: boolean, feedback?: string) => {
      setReviewing(true);
      try {
        const next = await api.review(threadId, approved, feedback);
        setInfo(next);
        lastStatus.current = next.status;
        toast.success(approved ? "Approved: finalizing the report" : "Feedback sent: revising the draft");
        setGeneration((g) => g + 1); // the run is running again: resume polling
      } catch (err) {
        const e = err instanceof ApiError ? err : new ApiError(String(err), null);
        if (e.status === 409) {
          // Decided elsewhere (another tab, the CLI) or still working: the
          // server's view wins, so re-sync instead of showing a stale form.
          toast.info("This draft was already reviewed. Refreshing the run.");
          setGeneration((g) => g + 1);
        } else {
          toast.error(`Couldn't send the review: ${e.message}`);
        }
      } finally {
        setReviewing(false);
      }
    },
    [threadId],
  );

  return { info, error, reconnecting, reviewing, retry, review };
}

/** Toasts for transitions seen live; nothing on the first load of an old run. */
function announce(prev: RunInfo["status"] | null, next: RunInfo["status"], label: string) {
  if (prev !== "running" || next === prev) return;
  const short = label.length > 60 ? `${label.slice(0, 57)}...` : label;
  if (next === "awaiting_review") toast("Draft ready for your review", { description: short });
  if (next === "done") toast.success("Report ready", { description: short });
  if (next === "failed") toast.error("The run failed", { description: short });
}

/** The current time, refreshed every `ms` while `active` (for live durations). */
export function useNow(active: boolean, ms = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [active, ms]);
  return now;
}
