"use client";

import { motion } from "framer-motion";
import { XCircle, Clock, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { RepAttempt } from "@/lib/api";
import { formatTimestamp, getFailureReasonLabel, cn } from "@/lib/utils";

interface NoRepListProps {
  noReps: RepAttempt[];
  onSeek?: (timestamp: number) => void;
  className?: string;
}

export function NoRepList({ noReps, onSeek, className }: NoRepListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (noReps.length === 0) {
    return (
      <div className={cn("card p-6 text-center", className)}>
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-kb-success/20 flex items-center justify-center">
          <span className="text-3xl">🎉</span>
        </div>
        <h3 className="font-display text-lg font-semibold mb-2">
          Perfect Set!
        </h3>
        <p className="text-kb-muted">No failed reps detected in this workout.</p>
      </div>
    );
  }

  // Group no-reps by failure reason
  const reasonCounts: Record<string, number> = {};
  noReps.forEach((rep) => {
    rep.failure_reasons?.forEach((reason) => {
      reasonCounts[reason] = (reasonCounts[reason] || 0) + 1;
    });
  });

  const sortedReasons = Object.entries(reasonCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Summary */}
      <div className="card p-4">
        <h3 className="font-display font-semibold mb-3 flex items-center gap-2">
          <XCircle className="w-5 h-5 text-kb-danger" />
          No-Rep Summary
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {sortedReasons.map(([reason, count]) => (
            <div
              key={reason}
              className="flex items-center justify-between p-2 bg-kb-surface rounded-lg text-sm"
            >
              <span className="text-kb-muted truncate">
                {getFailureReasonLabel(reason)}
              </span>
              <span className="text-kb-danger font-medium ml-2">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Individual no-reps */}
      <div className="card overflow-hidden">
        <div className="p-4 border-b border-kb-border">
          <h3 className="font-display font-semibold">
            All No-Reps ({noReps.length})
          </h3>
        </div>
        <div className="divide-y divide-kb-border max-h-96 overflow-y-auto">
          {noReps.map((rep, i) => (
            <motion.div
              key={rep.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <div
                className={cn(
                  "p-4 hover:bg-kb-surface/50 transition-colors cursor-pointer",
                  expandedId === rep.id && "bg-kb-surface/50"
                )}
                onClick={() =>
                  setExpandedId(expandedId === rep.id ? null : rep.id)
                }
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-kb-danger/20 flex items-center justify-center text-kb-danger text-sm font-bold">
                      #{rep.rep_number}
                    </div>
                    <div>
                      <div className="flex items-center gap-2 text-sm">
                        <Clock className="w-3.5 h-3.5 text-kb-muted" />
                        <span className="text-kb-muted">
                          {formatTimestamp(rep.timestamp_start)}
                        </span>
                      </div>
                      <div className="text-xs text-kb-danger mt-0.5">
                        {rep.failure_reasons?.length || 0} issue
                        {(rep.failure_reasons?.length || 0) !== 1 && "s"}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {onSeek && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSeek(rep.timestamp_start);
                        }}
                        className="text-xs text-kb-accent hover:underline"
                      >
                        Jump to
                      </button>
                    )}
                    {expandedId === rep.id ? (
                      <ChevronUp className="w-4 h-4 text-kb-muted" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-kb-muted" />
                    )}
                  </div>
                </div>

                {/* Expanded details */}
                {expandedId === rep.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-3 pt-3 border-t border-kb-border"
                  >
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-white">
                        Failure Reasons:
                      </div>
                      <ul className="space-y-1.5">
                        {rep.failure_reasons?.map((reason, j) => (
                          <li
                            key={j}
                            className="flex items-start gap-2 text-sm text-kb-muted"
                          >
                            <span className="text-kb-danger mt-0.5">•</span>
                            {getFailureReasonLabel(reason)}
                          </li>
                        ))}
                      </ul>
                      {rep.metrics && (
                        <div className="mt-3 pt-3 border-t border-kb-border/50">
                          <div className="text-xs text-kb-muted font-medium mb-2">
                            Metrics:
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            {rep.metrics?.lockout_angle && (
                              <div>
                                <span className="text-kb-muted">Lockout: </span>
                                <span className="text-white">
                                  {(rep.metrics.lockout_angle.min_angle ?? 0).toFixed(1)}°
                                </span>
                              </div>
                            )}
                            {rep.metrics?.tempo_ms != null && (
                              <div>
                                <span className="text-kb-muted">Tempo: </span>
                                <span className="text-white">
                                  {(rep.metrics.tempo_ms ?? 0).toFixed(0)}ms
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

