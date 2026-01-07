"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TimelinePoint } from "@/lib/api";
import { formatTimestamp, getFailureReasonLabel, cn } from "@/lib/utils";

interface VideoTimelineProps {
  timeline: TimelinePoint[];
  duration: number;
  currentTime?: number;
  onSeek?: (time: number) => void;
  className?: string;
}

export function VideoTimeline({
  timeline,
  duration,
  currentTime = 0,
  onSeek,
  className,
}: VideoTimelineProps) {
  const [hoveredRep, setHoveredRep] = useState<TimelinePoint | null>(null);
  const [hoverX, setHoverX] = useState(0);

  const getSegmentStyle = (point: TimelinePoint) => {
    const left = (point.timestamp_start / duration) * 100;
    const width = ((point.timestamp_end - point.timestamp_start) / duration) * 100;

    return {
      left: `${left}%`,
      width: `${Math.max(width, 0.5)}%`,
    };
  };

  const getColorClass = (color: string) => {
    switch (color) {
      case "green":
        return "bg-kb-success hover:bg-kb-success/80";
      case "red":
        return "bg-kb-danger hover:bg-kb-danger/80";
      case "yellow":
        return "bg-kb-warning hover:bg-kb-warning/80";
      default:
        return "bg-kb-muted";
    }
  };

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const time = percentage * duration;
    onSeek(time);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoverX(e.clientX - rect.left);
  };

  return (
    <div className={cn("relative", className)}>
      {/* Timeline header */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-kb-muted">Timeline</span>
        <div className="flex gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-kb-success" />
            Valid
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-kb-danger" />
            No-Rep
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-kb-warning" />
            Ambiguous
          </span>
        </div>
      </div>

      {/* Timeline bar */}
      <div
        className="relative h-12 bg-kb-surface rounded-lg overflow-hidden cursor-pointer"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredRep(null)}
      >
        {/* Background grid */}
        <div className="absolute inset-0 flex">
          {Array.from({ length: 10 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 border-r border-kb-border/30 last:border-r-0"
            />
          ))}
        </div>

        {/* Rep segments */}
        {timeline.map((point, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: i * 0.02 }}
            className={cn(
              "absolute top-1 bottom-1 rounded cursor-pointer transition-all",
              getColorClass(point.color),
              hoveredRep === point && "ring-2 ring-white ring-offset-2 ring-offset-kb-surface"
            )}
            style={getSegmentStyle(point)}
            onMouseEnter={() => setHoveredRep(point)}
          />
        ))}

        {/* Current time indicator */}
        {currentTime > 0 && (
          <motion.div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg shadow-white/50"
            style={{ left: `${(currentTime / duration) * 100}%` }}
            initial={false}
            animate={{ left: `${(currentTime / duration) * 100}%` }}
          />
        )}
      </div>

      {/* Time markers */}
      <div className="flex justify-between mt-1 text-xs text-kb-muted">
        <span>0:00</span>
        <span>{formatTimestamp(duration / 4)}</span>
        <span>{formatTimestamp(duration / 2)}</span>
        <span>{formatTimestamp((duration * 3) / 4)}</span>
        <span>{formatTimestamp(duration)}</span>
      </div>

      {/* Hover tooltip */}
      <AnimatePresence>
        {hoveredRep && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute z-10 mt-2 p-3 bg-kb-card border border-kb-border rounded-lg shadow-xl"
            style={{
              left: Math.min(Math.max(hoverX - 100, 0), 200),
              minWidth: "200px",
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-display font-semibold">
                Rep #{hoveredRep.rep_number}
              </span>
              <span
                className={cn(
                  "badge",
                  hoveredRep.classification === "valid" && "badge-success",
                  hoveredRep.classification === "no_rep" && "badge-danger",
                  hoveredRep.classification === "ambiguous" && "badge-warning"
                )}
              >
                {hoveredRep.classification.replace("_", "-")}
              </span>
            </div>
            <div className="text-sm text-kb-muted mb-1">
              {formatTimestamp(hoveredRep.timestamp_start)} →{" "}
              {formatTimestamp(hoveredRep.timestamp_end)}
            </div>
            {hoveredRep.failure_reasons && hoveredRep.failure_reasons.length > 0 && (
              <div className="mt-2 pt-2 border-t border-kb-border">
                <div className="text-xs text-kb-danger font-medium mb-1">
                  Failure Reasons:
                </div>
                <ul className="text-xs text-kb-muted space-y-0.5">
                  {hoveredRep.failure_reasons.map((reason, i) => (
                    <li key={i}>• {getFailureReasonLabel(reason)}</li>
                  ))}
                </ul>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

