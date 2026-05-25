"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getEvalResults, type EvalSummary } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const MAFA_REFERENCE = 86.0;

function shortName(name: string) {
  if (name.includes("Single")) return "Single LLM";
  if (name.includes("no MCP")) return "Multi-agent";
  if (name.includes("MCP")) return "Full pipeline";
  return name;
}

export function MetricsDashboard() {
  const [summaries, setSummaries] = useState<EvalSummary[]>([]);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvalResults()
      .then((r) => {
        setAvailable(r.available);
        setSummaries(r.summaries);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <p className="text-sm text-muted-foreground">
        Could not load eval results — is the API server running?
      </p>
    );
  }

  if (available === null) {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }

  if (!available || summaries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No eval results yet. Run the eval runner to populate this chart.
      </p>
    );
  }

  const chartData = summaries.map((s) => ({
    name: shortName(s.config_name),
    "Agreement %": +(s.agreement_rate * 100).toFixed(1),
    "Macro-F1 %": +(s.macro_f1 * 100).toFixed(1),
    "Human Review %": +(s.human_review_rate * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Agreement Rate &amp; Macro-F1 by Config</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => [`${v}%`]} />
              <Legend />
              <Bar dataKey="Agreement %" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Macro-F1 %" fill="#16a34a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          {/* Reference lines legend */}
          <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="inline-block w-6 border-t-2 border-dashed border-orange-500" />
              MAFA (AAAI 2026) {MAFA_REFERENCE}%
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Summary table */}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              {["Config", "N", "Agree%", "Macro-F1", "Human%", "items/hr"].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-medium text-muted-foreground">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summaries.map((s) => (
              <tr key={s.config_name} className="border-t">
                <td className="px-3 py-2 font-medium">{shortName(s.config_name)}</td>
                <td className="px-3 py-2">{s.n_items}</td>
                <td className="px-3 py-2">{(s.agreement_rate * 100).toFixed(1)}%</td>
                <td className="px-3 py-2">{(s.macro_f1 * 100).toFixed(2)}%</td>
                <td className="px-3 py-2">{(s.human_review_rate * 100).toFixed(1)}%</td>
                <td className="px-3 py-2">{s.items_per_hour.toFixed(1)}</td>
              </tr>
            ))}
            {/* MAFA reference row */}
            <tr className="border-t bg-orange-50 dark:bg-orange-950/20 text-muted-foreground italic">
              <td className="px-3 py-2">JP Morgan MAFA (ref)</td>
              <td className="px-3 py-2">—</td>
              <td className="px-3 py-2 font-medium text-orange-600">86.0%</td>
              <td className="px-3 py-2">—</td>
              <td className="px-3 py-2">—</td>
              <td className="px-3 py-2">—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
