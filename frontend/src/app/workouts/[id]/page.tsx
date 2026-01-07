"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { format } from "date-fns";
import {
  ArrowLeft,
  Clock,
  Calendar,
  Loader2,
  Timer,
  Trash2,
  Save,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Edit3,
  Smile,
  Frown,
  Meh,
  Battery,
  BatteryLow,
  X,
  Check,
} from "lucide-react";
import Link from "next/link";

import { workoutApi, RepAttempt } from "@/lib/api";
import { RepCounter } from "@/components/RepCounter";
import { VideoTimeline } from "@/components/VideoTimeline";
import { NoRepList } from "@/components/NoRepList";
import { cn, formatDuration, getLiftTypeLabel } from "@/lib/utils";

// Mood options with icons and colors
const MOOD_OPTIONS = [
  { value: "great", label: "Great", icon: Smile, color: "text-green-400", bg: "bg-green-400" },
  { value: "good", label: "Good", icon: Smile, color: "text-lime-400", bg: "bg-lime-400" },
  { value: "okay", label: "Okay", icon: Meh, color: "text-yellow-400", bg: "bg-yellow-400" },
  { value: "tired", label: "Tired", icon: BatteryLow, color: "text-orange-400", bg: "bg-orange-400" },
  { value: "heavy", label: "Heavy Day", icon: Frown, color: "text-red-400", bg: "bg-red-400" },
];

function ProcessingTimer({ startedAt }: { startedAt: string | null }) {
  const [elapsed, setElapsed] = useState("0:00");

  useEffect(() => {
    if (!startedAt) {
      setElapsed("0:00");
      return;
    }

    const timestamp = startedAt.includes('Z') || startedAt.includes('+') 
      ? startedAt 
      : startedAt + 'Z';
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
  }, [startedAt]);

  if (!startedAt) {
    return (
      <div className="flex items-center gap-2 text-kb-muted">
        <Timer className="w-5 h-5" />
        <span className="text-xl font-mono font-semibold">Waiting...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-kb-warning">
      <Timer className="w-5 h-5 animate-pulse" />
      <span className="text-xl font-mono font-semibold">{elapsed}</span>
    </div>
  );
}

function DeleteConfirmModal({ 
  isOpen, 
  onClose, 
  onConfirm, 
  isDeleting 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onConfirm: () => void;
  isDeleting: boolean;
}) {
  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="card p-6 max-w-md w-full"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-kb-danger/20 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6 text-kb-danger" />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold">Delete Workout?</h3>
            <p className="text-sm text-kb-muted">This action cannot be undone</p>
          </div>
        </div>
        <p className="text-kb-muted mb-6">
          This will permanently delete this workout, including all rep data and analytics.
        </p>
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 btn-secondary"
            disabled={isDeleting}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isDeleting}
            className="flex-1 bg-kb-danger hover:bg-kb-danger/80 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {isDeleting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
            Delete
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

const RPE_LABELS = [
  { value: 1, label: "Very Light", color: "bg-green-500" },
  { value: 2, label: "Light", color: "bg-green-400" },
  { value: 3, label: "Light", color: "bg-lime-400" },
  { value: 4, label: "Moderate", color: "bg-yellow-400" },
  { value: 5, label: "Moderate", color: "bg-yellow-500" },
  { value: 6, label: "Moderate", color: "bg-orange-400" },
  { value: 7, label: "Hard", color: "bg-orange-500" },
  { value: 8, label: "Hard", color: "bg-red-400" },
  { value: 9, label: "Very Hard", color: "bg-red-500" },
  { value: 10, label: "Max Effort", color: "bg-red-600" },
];

export default function WorkoutDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const workoutId = params.id as string;

  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isEditingNotes, setIsEditingNotes] = useState(false);
  const [isEditingDate, setIsEditingDate] = useState(false);
  const [notes, setNotes] = useState("");
  const [perceivedEffort, setPerceivedEffort] = useState<number | null>(null);
  const [mood, setMood] = useState<string | null>(null);
  const [editedDate, setEditedDate] = useState("");

  const {
    data: workout,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["workout", workoutId],
    queryFn: () => workoutApi.get(workoutId),
    refetchInterval: (data) => {
      if (
        data?.state?.data?.processing_status &&
        !["completed", "failed"].includes(data.state.data.processing_status)
      ) {
        return 2000;
      }
      return false;
    },
  });

  // Sync local state with workout data
  useEffect(() => {
    if (workout) {
      setNotes(workout.notes || "");
      setPerceivedEffort(workout.perceived_effort);
      setMood(workout.mood);
      setEditedDate(format(new Date(workout.workout_date), "yyyy-MM-dd"));
    }
  }, [workout]);

  const updateMutation = useMutation({
    mutationFn: (data: { notes?: string; perceived_effort?: number; mood?: string; workout_date?: string }) =>
      workoutApi.update(workoutId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workout", workoutId] });
      queryClient.invalidateQueries({ queryKey: ["workouts"] });
      setIsEditingNotes(false);
      setIsEditingDate(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => workoutApi.delete(workoutId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workouts"] });
      router.push("/");
    },
  });

  const handleSaveNotes = () => {
    updateMutation.mutate({ 
      notes, 
      perceived_effort: perceivedEffort ?? undefined 
    });
  };

  const handleSaveDate = () => {
    if (editedDate) {
      updateMutation.mutate({ workout_date: new Date(editedDate).toISOString() });
    }
  };

  const handleRPEChange = (value: number) => {
    setPerceivedEffort(value);
    updateMutation.mutate({ perceived_effort: value });
  };

  const handleMoodChange = (value: string) => {
    setMood(value);
    updateMutation.mutate({ mood: value });
  };

  const noReps =
    workout?.rep_attempts.filter((r) => r.classification === "no_rep") || [];

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-kb-bg">
        <Loader2 className="w-8 h-8 animate-spin text-kb-accent" />
      </div>
    );
  }

  if (error || !workout) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-kb-bg">
        <div className="card p-8 text-center max-w-md">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-kb-danger/20 flex items-center justify-center">
            <XCircle className="w-8 h-8 text-kb-danger" />
          </div>
          <h2 className="font-display text-xl font-semibold mb-2">Failed to Load Workout</h2>
          <p className="text-kb-muted mb-6">
            The workout could not be found or you don&apos;t have permission to view it.
          </p>
          <button onClick={() => router.push("/")} className="btn-primary">
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const isProcessing = !["completed", "failed"].includes(
    workout.processing_status
  );

  const selectedMood = MOOD_OPTIONS.find(m => m.value === mood);

  return (
    <div className="min-h-screen pb-12 bg-kb-bg">
      {/* Delete confirmation modal */}
      <AnimatePresence>
        {showDeleteModal && (
          <DeleteConfirmModal
            isOpen={showDeleteModal}
            onClose={() => setShowDeleteModal(false)}
            onConfirm={() => deleteMutation.mutate()}
            isDeleting={deleteMutation.isPending}
          />
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="border-b border-kb-border bg-kb-surface/80 backdrop-blur-xl sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="p-2 -ml-2 hover:bg-kb-card rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div>
                <h1 className="font-display text-xl font-bold">
                  {getLiftTypeLabel(workout.lift_type)}
                </h1>
                {/* Editable Date */}
                <div className="flex items-center gap-2">
                  {isEditingDate ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="date"
                        value={editedDate}
                        onChange={(e) => setEditedDate(e.target.value)}
                        className="bg-kb-surface border border-kb-border rounded px-2 py-1 text-sm focus:outline-none focus:border-kb-accent"
                      />
                      <button
                        onClick={handleSaveDate}
                        disabled={updateMutation.isPending}
                        className="p-1 hover:bg-kb-surface rounded text-kb-success"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => {
                          setEditedDate(format(new Date(workout.workout_date), "yyyy-MM-dd"));
                          setIsEditingDate(false);
                        }}
                        className="p-1 hover:bg-kb-surface rounded text-kb-danger"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <p className="text-sm text-kb-muted">
                        {format(new Date(workout.workout_date), "EEEE, MMMM d, yyyy")}
                      </p>
                      <button
                        onClick={() => setIsEditingDate(true)}
                        className="p-1 hover:bg-kb-surface rounded text-kb-muted hover:text-white"
                      >
                        <Edit3 className="w-3 h-3" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowDeleteModal(true)}
                className="p-2 hover:bg-kb-danger/20 rounded-lg transition-colors text-kb-muted hover:text-kb-danger"
                title="Delete workout"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Workout Info Card - Always visible */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6 mb-6"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-kb-accent/20 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-kb-accent" />
              </div>
              <div>
                <div className="text-xs text-kb-muted uppercase tracking-wider">Date</div>
                <div className="font-semibold">
                  {format(new Date(workout.workout_date), "MMM d, yyyy")}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-kb-accent/20 flex items-center justify-center">
                <Clock className="w-5 h-5 text-kb-accent" />
              </div>
              <div>
                <div className="text-xs text-kb-muted uppercase tracking-wider">Duration</div>
                <div className="font-semibold">
                  {workout.video_duration_seconds
                    ? formatDuration(workout.video_duration_seconds)
                    : "--:--"}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-kb-success/20 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-kb-success" />
              </div>
              <div>
                <div className="text-xs text-kb-muted uppercase tracking-wider">Valid Reps</div>
                <div className="font-semibold text-kb-success">{workout.valid_reps}</div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-kb-danger/20 flex items-center justify-center">
                <XCircle className="w-5 h-5 text-kb-danger" />
              </div>
              <div>
                <div className="text-xs text-kb-muted uppercase tracking-wider">No-Reps</div>
                <div className="font-semibold text-kb-danger">{workout.no_reps}</div>
              </div>
            </div>
          </div>

          {/* Mood Selector */}
          <div className="mt-6 pt-6 border-t border-kb-border">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-kb-muted">How did you feel?</span>
              {selectedMood && (
                <span className={cn("text-sm font-medium", selectedMood.color)}>
                  {selectedMood.label}
                </span>
              )}
            </div>
            <div className="flex gap-2">
              {MOOD_OPTIONS.map((option) => {
                const Icon = option.icon;
                return (
                  <button
                    key={option.value}
                    onClick={() => handleMoodChange(option.value)}
                    className={cn(
                      "flex-1 flex flex-col items-center gap-1 p-3 rounded-lg transition-all",
                      mood === option.value
                        ? `${option.bg} text-white shadow-lg`
                        : "bg-kb-surface hover:bg-kb-card text-kb-muted hover:text-white"
                    )}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="text-xs">{option.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* RPE Selector */}
          <div className="mt-6 pt-6 border-t border-kb-border">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-kb-muted">Perceived Effort (RPE)</span>
              {perceivedEffort && (
                <span className="text-sm font-medium">
                  {RPE_LABELS[perceivedEffort - 1]?.label}
                </span>
              )}
            </div>
            <div className="flex gap-1">
              {RPE_LABELS.map((rpe) => (
                <button
                  key={rpe.value}
                  onClick={() => handleRPEChange(rpe.value)}
                  className={cn(
                    "flex-1 h-8 rounded transition-all text-xs font-medium",
                    perceivedEffort === rpe.value
                      ? `${rpe.color} text-white shadow-lg scale-110`
                      : "bg-kb-surface hover:bg-kb-card text-kb-muted"
                  )}
                >
                  {rpe.value}
                </button>
              ))}
            </div>
          </div>

          {/* Notes Section */}
          <div className="mt-6 pt-6 border-t border-kb-border">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-kb-muted">Workout Notes & Description</span>
              {!isEditingNotes && (
                <button
                  onClick={() => setIsEditingNotes(true)}
                  className="text-sm text-kb-accent hover:underline flex items-center gap-1"
                >
                  <Edit3 className="w-3 h-3" />
                  {notes ? "Edit" : "Add notes"}
                </button>
              )}
            </div>
            
            {isEditingNotes ? (
              <div className="space-y-3">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="How did you feel? Any technique observations? Goals for next time..."
                  className="w-full h-32 bg-kb-surface border border-kb-border rounded-lg p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-kb-accent/50"
                />
                <div className="flex gap-2 justify-end">
                  <button
                    onClick={() => {
                      setNotes(workout.notes || "");
                      setIsEditingNotes(false);
                    }}
                    className="btn-secondary text-sm"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveNotes}
                    disabled={updateMutation.isPending}
                    className="btn-primary text-sm flex items-center gap-2"
                  >
                    {updateMutation.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Save className="w-4 h-4" />
                    )}
                    Save
                  </button>
                </div>
              </div>
            ) : notes ? (
              <p className="text-sm whitespace-pre-wrap bg-kb-surface rounded-lg p-4">{notes}</p>
            ) : (
              <p className="text-sm text-kb-muted italic">No notes yet. Click &quot;Add notes&quot; to record your thoughts.</p>
            )}
          </div>
        </motion.div>

        {/* Processing state */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-8 text-center mb-8"
          >
            <div className="flex items-center justify-center gap-4 mb-6">
              <Loader2 className="w-10 h-10 animate-spin text-kb-accent" />
              <ProcessingTimer startedAt={workout.processing_started_at} />
            </div>
            <h2 className="font-display text-xl font-semibold mb-2">
              Analyzing Your Workout
            </h2>
            <p className="text-kb-muted mb-6">
              {workout.processing_status === "pending" || workout.processing_status === "queued"
                ? "Queued for processing..."
                : workout.processing_status === "processing"
                ? "Extracting frames and detecting poses..."
                : "Validating reps and computing analytics..."}
            </p>
            <div className="max-w-md mx-auto">
              <div className="flex items-center gap-4">
                <div className="flex-1 h-3 bg-kb-surface rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-kb-accent to-kb-success"
                    animate={{ width: `${workout.processing_progress * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-lg font-mono font-semibold min-w-[4rem] text-right">
                  {Math.round(workout.processing_progress * 100)}%
                </span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Failed state */}
        {workout.processing_status === "failed" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-8 text-center border-kb-danger/50 mb-8"
          >
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-kb-danger/20 flex items-center justify-center">
              <XCircle className="w-8 h-8 text-kb-danger" />
            </div>
            <h2 className="font-display text-xl font-semibold text-kb-danger mb-2">
              Processing Failed
            </h2>
            <p className="text-kb-muted mb-4">{workout.processing_error || "An unknown error occurred"}</p>
          </motion.div>
        )}

        {/* Completed state - detailed analytics */}
        {workout.processing_status === "completed" && (
          <>
            {/* Rep counts */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="mb-8"
            >
              <RepCounter
                totalAttempts={workout.total_attempts}
                validReps={workout.valid_reps}
                noReps={workout.no_reps}
                ambiguousReps={workout.ambiguous_reps}
              />
            </motion.div>

            {/* Timeline */}
            {workout.timeline.length > 0 && workout.video_duration_seconds && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="card p-6 mb-8"
              >
                <VideoTimeline
                  timeline={workout.timeline}
                  duration={workout.video_duration_seconds}
                />
              </motion.div>
            )}

            {/* Analytics & No-Reps grid */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Analytics */}
              {workout.analytics_summary && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="card p-6"
                >
                  <h3 className="font-display font-semibold mb-4">Form Analytics</h3>
                  <div className="space-y-4">
                    {/* Valid rate */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-kb-muted">Valid Rep Rate</span>
                        <span className="text-kb-success font-medium">
                          {((workout.analytics_summary?.valid_rep_rate ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 bg-kb-surface rounded-full overflow-hidden">
                        <div
                          className="h-full bg-kb-success"
                          style={{
                            width: `${(workout.analytics_summary?.valid_rep_rate ?? 0) * 100}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Tempo */}
                    {workout.analytics_summary?.tempo && (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 bg-kb-surface rounded-lg">
                          <div className="text-xs text-kb-muted mb-1">Avg Tempo</div>
                          <div className="text-lg font-semibold">
                            {(workout.analytics_summary.tempo.avg_seconds ?? 0).toFixed(1)}s
                          </div>
                        </div>
                        <div className="p-3 bg-kb-surface rounded-lg">
                          <div className="text-xs text-kb-muted mb-1">Consistency</div>
                          <div className="text-lg font-semibold">
                            {((workout.analytics_summary.tempo.consistency ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Fatigue indicator */}
                    {workout.analytics_summary?.fatigue?.fatigue_detected && (
                      <div className="p-3 bg-kb-surface rounded-lg">
                        <div className="text-xs text-kb-muted mb-2">Fatigue Analysis</div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Overall Score</span>
                          <span
                            className={cn(
                              "font-medium",
                              (workout.analytics_summary.fatigue.fatigue_score ?? 0) < 0.3
                                ? "text-kb-success"
                                : "text-kb-warning"
                            )}
                          >
                            {((1 - (workout.analytics_summary.fatigue.fatigue_score ?? 0)) * 100).toFixed(0)}% fresh
                          </span>
                        </div>
                        {(workout.analytics_summary.fatigue.fatigue_score ?? 0) > 0.3 && (
                          <div className="mt-2 text-xs text-kb-warning">
                            ⚠️ Fatigue detected: form may have degraded
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* No-reps list */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <NoRepList noReps={noReps as RepAttempt[]} />
              </motion.div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
