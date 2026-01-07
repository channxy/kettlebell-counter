"use client";

import { motion } from "framer-motion";
import { Check, X, AlertTriangle, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface RepCounterProps {
  totalAttempts: number;
  validReps: number;
  noReps: number;
  ambiguousReps: number;
  className?: string;
  showPercentage?: boolean;
}

export function RepCounter({
  totalAttempts,
  validReps,
  noReps,
  ambiguousReps,
  className,
  showPercentage = true,
}: RepCounterProps) {
  const validRate = totalAttempts > 0 ? (validReps / totalAttempts) * 100 : 0;

  return (
    <div className={cn("grid grid-cols-2 lg:grid-cols-4 gap-4", className)}>
      {/* Total Attempts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0 }}
        className="stat-card"
      >
        <div className="flex items-center gap-2 text-kb-muted">
          <TrendingUp className="w-4 h-4" />
          <span className="stat-label">Total Attempts</span>
        </div>
        <motion.span
          key={totalAttempts}
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          className="stat-value text-white"
        >
          {totalAttempts}
        </motion.span>
      </motion.div>

      {/* Valid Reps */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="stat-card border-kb-success/30"
      >
        <div className="flex items-center gap-2 text-kb-success">
          <Check className="w-4 h-4" />
          <span className="stat-label">Valid Reps</span>
        </div>
        <div className="flex items-baseline gap-3">
          <motion.span
            key={validReps}
            initial={{ scale: 1.2 }}
            animate={{ scale: 1 }}
            className="stat-value text-kb-success"
          >
            {validReps}
          </motion.span>
          {showPercentage && totalAttempts > 0 && (
            <span className="text-kb-success/60 text-sm">
              {validRate.toFixed(1)}%
            </span>
          )}
        </div>
      </motion.div>

      {/* No-Reps */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="stat-card border-kb-danger/30"
      >
        <div className="flex items-center gap-2 text-kb-danger">
          <X className="w-4 h-4" />
          <span className="stat-label">No-Reps</span>
        </div>
        <motion.span
          key={noReps}
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          className="stat-value text-kb-danger"
        >
          {noReps}
        </motion.span>
      </motion.div>

      {/* Ambiguous */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="stat-card border-kb-warning/30"
      >
        <div className="flex items-center gap-2 text-kb-warning">
          <AlertTriangle className="w-4 h-4" />
          <span className="stat-label">Ambiguous</span>
        </div>
        <motion.span
          key={ambiguousReps}
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          className="stat-value text-kb-warning"
        >
          {ambiguousReps}
        </motion.span>
      </motion.div>
    </div>
  );
}

// Large display version for main dashboard
export function RepCounterLarge({
  validReps,
  totalAttempts,
}: {
  validReps: number;
  totalAttempts: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card-glow p-8 text-center"
    >
      <div className="text-kb-muted text-sm uppercase tracking-wider mb-2">
        Valid Reps
      </div>
      <motion.div
        key={validReps}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="font-display text-8xl font-black text-gradient"
      >
        {validReps}
      </motion.div>
      <div className="text-kb-muted mt-4">
        out of{" "}
        <span className="text-white font-semibold">{totalAttempts}</span>{" "}
        attempts
      </div>
    </motion.div>
  );
}

