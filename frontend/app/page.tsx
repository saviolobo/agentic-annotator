"use client";

import { useState } from "react";
import { annotate, type AnnotateResponse } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { QueueStatus } from "@/components/QueueStatus";

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<AnnotateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnnotate() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await annotate(query));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background p-6 md:p-12 space-y-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Agentic Annotator</h1>
        <p className="text-muted-foreground mt-1">
          Multi-agent Banking77 intent classification · Inspired by JP Morgan MAFA (AAAI 2026)
        </p>
      </div>

      <Separator />

      {/* Live demo */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Live Demo</h2>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder="Type a banking customer query…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAnnotate()}
          />
          <Button onClick={handleAnnotate} disabled={loading || !query.trim()}>
            {loading ? "Annotating…" : "Annotate"}
          </Button>
        </div>

        {error && (
          <p className="text-sm text-destructive">
            Error: {error}. Is the API server running? (python main.py)
          </p>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2 text-base">
                <span className="font-mono text-lg">{result.final_label}</span>
                <Badge variant={result.route_to_human ? "destructive" : "default"}>
                  {result.route_to_human ? "→ Human Review" : "Auto-labeled"}
                </Badge>
                <Badge variant="outline">{result.route}</Badge>
                {result.qc_approved ? (
                  <Badge variant="outline" className="text-green-600">
                    QC ✓
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-yellow-600">
                    QC ✗
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-1">
              {result.qc_flags.length > 0 && (
                <p>Flags: {result.qc_flags.join(" · ")}</p>
              )}
              <p>Elapsed: {result.elapsed_seconds}s</p>
            </CardContent>
          </Card>
        )}
      </section>

      <Separator />

      {/* Metrics */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Ablation Results</h2>
        <p className="text-sm text-muted-foreground">
          Run{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            python -m eval.run_eval --smoke --n 10 --output eval_results.json
          </code>{" "}
          to populate.
        </p>
        <MetricsDashboard />
      </section>

      <Separator />

      {/* Queue */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Human Review Queue</h2>
        <QueueStatus />
      </section>
    </main>
  );
}
