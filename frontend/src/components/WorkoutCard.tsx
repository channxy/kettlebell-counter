"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { format, formatDistanceToNow } from "date-fns";
import { Clock, Dumbbell, CheckCircle, XCircle, AlertCircle, ChevronRight, Timer } from "lucide-react";
import Link from "next/link";
import { Workout } from "@/lib/api";
import { cn, formatDuration, getLiftTypeLabel, getStatusColor } from "@/lib/utils";

interface WorkoutCardProps {
  workout: Workout;
  index?: number;
}

function ProcessingTimer({ createdAt }: { createdAt: string }) {
  const [elapsed, setElapsed] = useState("0:00");

  useEffect(() => {
    // Parse timestamp - treat as UTC if no timezone specified
    // Append 'Z' if no timezone info to indicate UTC
    const timestamp = createdAt.includes('Z') || createdAt.includes('+') 
      ? createdAt 
      : createdAt + 'Z';
    const start = new Date(timestamp).getTime();
    
    const updateTimer = () => {
      const now = Date.now();
      const diff = Math.max(0, Math.floor((now - start) / 1000));
      const mins = Math.floor(diff / 60);
      const secs = diff % 60;
      setElapsed(`${mins}:${secs.toString().padStart(2, "0")}`);
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [createdAt]);

  return (
    <div className="flex items-center gap-1.5 text-kb-warning">
      <Timer className="w-3.5 h-3.5 animate-pulse" />
      <span className="text-sm font-mono">{elapsed}</span>
    </div>
  );
}

export function WorkoutCard({ workout, index = 0 }: WorkoutCardProps) {
  const isProcessing = ["pending", "processing", "analyzing", "queued"].includes(
    workout.processing_status
  );
  const isFailed = workout.processing_status === "failed";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Link href={`/workouts/${workout.id}`}>
        <div
          className={cn(
            "card p-5 hover:border-kb-accent/50 transition-all cursor-pointer group",
            isProcessing && "animate-pulse-slow"
          )}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-kb-accent/20 flex items-center justify-center">
                <Dumbbell className="w-5 h-5 text-kb-accent" />
              </div>
              <div>
                <h3 className="font-display font-semibold">
                  {getLiftTypeLabel(workout.lift_type)}
                </h3>
                <p className="text-sm text-kb-muted">
                  {format(new Date(workout.workout_date), "MMM d, yyyy")}
                </p>
              </div>
            </div>
            <ChevronRight className="w-5 h-5 text-kb-muted group-hover:text-kb-accent transition-colors" />
          </div>

          {/* Status or stats */}
          {isProcessing ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-kb-accent animate-pulse" />
                  <span className={cn("text-sm", getStatusColor(workout.processing_status))}>
                    {workout.processing_status === "pending" || workout.processing_status === "queued"
                      ? "Queued..."
                      : workout.processing_status === "processing"
                      ? "Processing video..."
                      : "Analyzing reps..."}
                  </span>
                </div>
                <ProcessingTimer createdAt={workout.created_at} />
              </div>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-2 bg-kb-surface rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-kb-accent to-kb-success"
                    initial={{ width: 0 }}
                    animate={{ width: `${workout.processing_progress * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-sm font-mono text-kb-muted min-w-[3rem] text-right">
                  {Math.round(workout.processing_progress * 100)}%
                </span>
              </div>
            </div>
          ) : isFailed ? (
            <div className="flex items-center gap-2 text-kb-danger">
              <XCircle className="w-4 h-4" />
              <span className="text-sm">Processing failed</span>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-3">
              {/* Total */}
              <div className="text-center">
                <div className="text-2xl font-display font-bold text-white">
                  {workout.total_attempts}
                </div>
                <div className="text-xs text-kb-muted">Total</div>
              </div>

              {/* Valid */}
              <div className="text-center">
                <div className="text-2xl font-display font-bold text-kb-success flex items-center justify-center gap-1">
                  <CheckCircle className="w-4 h-4" />
                  {workout.valid_reps}
                </div>
                <div className="text-xs text-kb-muted">Valid</div>
              </div>

              {/* No-Reps */}
              <div className="text-center">
                <div className="text-2xl font-display font-bold text-kb-danger flex items-center justify-center gap-1">
                  <XCircle className="w-4 h-4" />
                  {workout.no_reps}
                </div>
                <div className="text-xs text-kb-muted">No-Rep</div>
              </div>

              {/* Duration */}
              <div className="text-center">
                <div className="text-2xl font-display font-bold text-kb-muted flex items-center justify-center gap-1">
                  <Clock className="w-4 h-4" />
                  {workout.video_duration_seconds
                    ? formatDuration(workout.video_duration_seconds)
                    : "--:--"}
                </div>
                <div className="text-xs text-kb-muted">Duration</div>
              </div>
            </div>
          )}

          {/* Valid rate bar */}
          {!isProcessing && !isFailed && workout.total_attempts > 0 && (
            <div className="mt-4 pt-3 border-t border-kb-border">
              <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-kb-muted">Valid Rate</span>
                <span className="text-kb-success font-medium">
                  {((workout.valid_reps / workout.total_attempts) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-kb-surface rounded-full overflow-hidden flex">
                <div
                  className="bg-kb-success transition-all"
                  style={{
                    width: `${(workout.valid_reps / workout.total_attempts) * 100}%`,
                  }}
                />
                <div
                  className="bg-kb-danger transition-all"
                  style={{
                    width: `${(workout.no_reps / workout.total_attempts) * 100}%`,
                  }}
                />
                <div
                  className="bg-kb-warning transition-all"
                  style={{
                    width: `${(workout.ambiguous_reps / workout.total_attempts) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </Link>
    </motion.div>
  );
}

