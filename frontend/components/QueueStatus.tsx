"use client";

import { useEffect, useState } from "react";
import { getQueueStats } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function QueueStatus() {
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getQueueStats()
      .then((s) => setCount(s.pending_count))
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <p className="text-sm text-muted-foreground">
        Could not load queue — is the API server running?
      </p>
    );
  }

  return (
    <Card className="max-w-xs">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Pending Reviews</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center gap-3">
        {count === null ? (
          <span className="text-sm text-muted-foreground">Loading…</span>
        ) : (
          <>
            <span className="text-4xl font-bold">{count}</span>
            <Badge variant={count > 0 ? "destructive" : "outline"}>
              {count > 0 ? "Needs attention" : "Queue empty"}
            </Badge>
          </>
        )}
      </CardContent>
    </Card>
  );
}
