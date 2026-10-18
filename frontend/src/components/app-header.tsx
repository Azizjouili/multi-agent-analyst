import { useEffect, useState } from "react";
import { BookOpen, Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useTheme } from "@/hooks/use-theme";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

function useApiHealth() {
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api.health().then(
        () => !cancelled && setOnline(true),
        () => !cancelled && setOnline(false),
      );
    check();
    // Infrequent: this is a hint in the header, not the run's progress.
    const id = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);
  return online;
}

export function AppHeader() {
  const { theme, toggle } = useTheme();
  const online = useApiHealth();

  return (
    <header className="sticky top-0 z-20 border-b bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-3xl items-center gap-3 px-4">
        <img src="/favicon.svg" alt="" className="size-7" />
        <div className="leading-tight">
          <h1 className="text-sm font-semibold tracking-tight">Multi-Agent Analyst</h1>
          <p className="hidden text-xs text-muted-foreground sm:block">Supervisor · specialists · critic · you</p>
        </div>

        <div className="ml-auto flex items-center gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="mr-2 inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                <span
                  className={cn(
                    "size-2 rounded-full",
                    online === null && "bg-muted-foreground/40",
                    online === true && "bg-success shadow-[0_0_0_3px] shadow-success/20",
                    online === false && "bg-destructive",
                  )}
                />
                <span className="hidden sm:inline">{online === false ? "API offline" : "API"}</span>
              </span>
            </TooltipTrigger>
            <TooltipContent>
              {online === false ? "GET /health is failing: is the server running?" : "GET /health is answering"}
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" asChild>
                <a href="/docs" target="_blank" rel="noreferrer" aria-label="API docs">
                  <BookOpen />
                </a>
              </Button>
            </TooltipTrigger>
            <TooltipContent>API docs (/docs)</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={toggle} aria-label="Toggle dark mode">
                {theme === "dark" ? <Sun /> : <Moon />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{theme === "dark" ? "Light mode" : "Dark mode"}</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </header>
  );
}
