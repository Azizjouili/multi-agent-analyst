import { AlertTriangle, FlaskConical, ShoppingBag, TrendingUp, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

// Questions the pinned SEC corpus can answer (close to evals/questions.yaml),
// so a first-time visitor gets a real, cited report rather than "not found".
export const EXAMPLES: { icon: LucideIcon; title: string; question: string }[] = [
  {
    icon: TrendingUp,
    title: "NVIDIA's growth",
    question:
      "How much did NVIDIA's revenue grow in fiscal 2025 compared with fiscal 2024, and what does its 10-K say drove it?",
  },
  {
    icon: AlertTriangle,
    title: "Apple's supply-chain risks",
    question: "What supply-chain risks does Apple describe in its fiscal 2024 10-K, and how large is its revenue at stake?",
  },
  {
    icon: FlaskConical,
    title: "R&D intensity",
    question: "Compare R&D spending as a share of revenue for Microsoft, Alphabet and Meta in their latest fiscal year.",
  },
  {
    icon: ShoppingBag,
    title: "Nike's wholesale decline",
    question: "Why did Nike's revenue fall in fiscal 2025, and how did wholesale and NIKE Direct differ?",
  },
];

export function ExampleChips({ onPick, compact = false }: { onPick: (q: string) => void; compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {EXAMPLES.map(({ icon: Icon, title, question }) => (
          <button
            key={title}
            type="button"
            onClick={() => onPick(question)}
            className="inline-flex items-center gap-1.5 rounded-full border bg-card px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
          >
            <Icon className="size-3.5" />
            {title}
          </button>
        ))}
      </div>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {EXAMPLES.map(({ icon: Icon, title, question }) => (
        <button
          key={title}
          type="button"
          onClick={() => onPick(question)}
          className={cn(
            "group flex items-start gap-3 rounded-xl border bg-card p-4 text-left transition-all",
            "hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
          )}
        >
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
            <Icon className="size-4" />
          </span>
          <span>
            <span className="block text-sm font-medium">{title}</span>
            <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">{question}</span>
          </span>
        </button>
      ))}
    </div>
  );
}
