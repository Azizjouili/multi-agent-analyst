import { useCallback, useEffect, useRef, useState } from "react";
import { Bot, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { AppHeader } from "@/components/app-header";
import { Composer } from "@/components/composer";
import { ExampleChips } from "@/components/example-chips";
import { QuestionBubble, RunCard, StatusPill } from "@/components/run-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError } from "@/lib/api";
import { load, save } from "@/lib/storage";

interface Entry {
  id: string;
  question: string | null; // null: restored from a URL, question not known yet
}

// The thread is a per-browser convenience: the runs themselves live on the
// server (checkpoint DB), so losing this list loses nothing but the view.
const THREAD_KEY = "analyst.thread";
const MAX_ENTRIES = 20;

// "#run=<id>" makes a run linkable and reload-safe: on load, that run is
// re-polled from the server, which knows its state even after a restart.
function runFromHash(): string | null {
  return window.location.hash.match(/^#run=([\w-]+)$/)?.[1] ?? null;
}

function setHash(id: string | null) {
  history.replaceState(null, "", id ? `#run=${id}` : window.location.pathname);
}

function initialThread(): Entry[] {
  const saved = load<Entry[]>(THREAD_KEY, []);
  const entries = Array.isArray(saved) ? saved.filter((e) => typeof e?.id === "string") : [];
  const fromUrl = runFromHash();
  if (fromUrl && !entries.some((e) => e.id === fromUrl)) entries.push({ id: fromUrl, question: null });
  return entries.slice(-MAX_ENTRIES);
}

export default function App() {
  const [entries, setEntries] = useState<Entry[]>(initialThread);
  const [draft, setDraft] = useState("");
  const [starting, setStarting] = useState<string | null>(null); // question being POSTed
  const viewport = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  useEffect(() => save(THREAD_KEY, entries), [entries]);

  // Opening a shared #run=<id> link in an open tab adds that run.
  useEffect(() => {
    const onHash = () => {
      const id = runFromHash();
      if (id) setEntries((es) => (es.some((e) => e.id === id) ? es : [...es, { id, question: null }]));
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const scrollToEnd = useCallback(() => {
    requestAnimationFrame(() =>
      viewport.current?.scrollTo({ top: viewport.current.scrollHeight, behavior: "smooth" }),
    );
  }, []);

  useEffect(scrollToEnd, [entries.length, starting, scrollToEnd]);

  async function ask(question: string) {
    setStarting(question);
    try {
      const run = await api.startRun(question);
      setEntries((es) => [...es, { id: run.thread_id, question }].slice(-MAX_ENTRIES));
      setHash(run.thread_id);
      setDraft("");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : String(err);
      toast.error(`Couldn't start the run: ${message}`);
    } finally {
      setStarting(null);
    }
  }

  const remove = useCallback((id: string) => {
    setEntries((es) => {
      const rest = es.filter((e) => e.id !== id);
      if (runFromHash() === id) setHash(rest.at(-1)?.id ?? null);
      return rest;
    });
  }, []);

  const learnQuestion = useCallback((id: string, question: string) => {
    setEntries((es) => es.map((e) => (e.id === id && !e.question ? { ...e, question } : e)));
  }, []);

  function pick(question: string) {
    setDraft(question);
    composer.current?.focus();
  }

  const empty = entries.length === 0 && !starting;

  return (
    <div className="flex h-full flex-col">
      <AppHeader />

      <ScrollArea className="min-h-0 flex-1" viewportRef={viewport}>
        <main className="mx-auto w-full max-w-3xl px-4 pt-8 pb-10">
          {empty ? (
            <EmptyState onPick={pick} />
          ) : (
            <div className="space-y-10">
              {entries.length > 1 && (
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground"
                    onClick={() => {
                      setEntries([]);
                      setHash(null);
                    }}
                  >
                    <Trash2 /> Clear thread
                  </Button>
                </div>
              )}
              {entries.map((entry) => (
                <RunCard
                  key={entry.id}
                  threadId={entry.id}
                  question={entry.question}
                  onRemove={() => remove(entry.id)}
                  onQuestion={(q) => learnQuestion(entry.id, q)}
                />
              ))}
              {starting && <StartingCard question={starting} />}
            </div>
          )}
        </main>
      </ScrollArea>

      <div className="border-t bg-gradient-to-t from-background via-background to-background/60">
        <div className="mx-auto w-full max-w-3xl space-y-2 px-4 py-3">
          {!empty && !draft && <ExampleChips onPick={pick} compact />}
          <Composer ref={composer} value={draft} onChange={setDraft} onSubmit={ask} busy={starting !== null} />
          <p className="text-center text-[0.7rem] text-muted-foreground">
            Answers are researched by AI agents from documents, a database and the web. Check the cited sources.
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center pt-6 text-center sm:pt-12">
      <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/60 text-primary-foreground shadow-lg shadow-primary/25">
        <Bot className="size-7" />
      </div>
      <h2 className="mt-5 text-2xl font-semibold tracking-tight sm:text-3xl">What should we research?</h2>
      <p className="mt-2 max-w-lg text-sm leading-relaxed text-muted-foreground">
        A supervisor splits your question into sub-tasks for specialists that search annual reports, query the
        company database and read the web. A writer drafts a cited report, a critic checks it, and you approve it.
      </p>
      <div className="mt-8 w-full text-left">
        <ExampleChips onPick={onPick} />
      </div>
    </div>
  );
}

/** Between "Ask" and the server's 202: the run has no id yet. */
function StartingCard({ question }: { question: string }) {
  return (
    <section className="space-y-3">
      <QuestionBubble question={question} />
      <Card className="gap-0 py-0">
        <div className="flex items-center gap-2 border-b px-5 py-3">
          <Skeleton className="size-6 rounded-md" />
          <span className="text-sm font-medium">Analyst</span>
          <StatusPill state="pending" />
        </div>
        <CardContent className="space-y-2.5 px-5 py-5">
          <Skeleton className="h-3.5 w-1/3" />
          <Skeleton className="h-3 w-2/3" />
        </CardContent>
      </Card>
    </section>
  );
}
