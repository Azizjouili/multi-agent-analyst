import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API the dev server forwards to. Override when it runs elsewhere:
//   API_URL=http://127.0.0.1:8010 npm run dev
const API_URL = process.env.API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // "@/components/ui/button" instead of "../../components/ui/button";
    // shadcn/ui's generated code assumes this alias (see components.json).
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    // Docker copies this folder into src/analyst/static; locally, FastAPI
    // serves it straight from here (api.py: FRONTEND_DIST).
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    // Why a proxy: in dev the page comes from Vite (:5173) but the API is on
    // :8000. Forwarding the API paths keeps the browser on one origin, so the
    // app uses the same relative URLs (/runs, /health) as in production and
    // the API needs no CORS settings. Plain path prefixes: "/runs" covers
    // /runs, /runs/{id} and /runs/{id}/review.
    proxy: {
      "/runs": API_URL,
      "/health": API_URL,
    },
  },
});
