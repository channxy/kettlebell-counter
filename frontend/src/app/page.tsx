"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { 
  Plus, Dumbbell, TrendingUp, Activity, LogOut, Upload, 
  User, ChevronDown, ChevronUp, Trash2, X, Check, Edit2, Pencil
} from "lucide-react";
import { workoutApi, analyticsApi, authApi } from "@/lib/api";
import { VideoUpload } from "@/components/VideoUpload";
import { WorkoutCard } from "@/components/WorkoutCard";
import { cn } from "@/lib/utils";

export default function HomePage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const topRef = useRef<HTMLDivElement>(null);
  
  const [showUpload, setShowUpload] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAllWorkouts, setShowAllWorkouts] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [selectedWorkouts, setSelectedWorkouts] = useState<Set<string>>(new Set());
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  // User name editing
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState("");

  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    } else {
      setIsAuthenticated(true);
    }
  }, [router]);

  // Fetch user data
  const { data: userData, refetch: refetchUser } = useQuery({
    queryKey: ["user"],
    queryFn: () => authApi.getMe(),
    enabled: isAuthenticated,
  });

  // Update user mutation
  const updateUserMutation = useMutation({
    mutationFn: (fullName: string) => authApi.updateMe(fullName),
    onSuccess: () => {
      refetchUser();
      setIsEditingName(false);
    },
  });

  // Fetch workouts - get more when viewing all
  const { data: workoutsData, refetch: refetchWorkouts } = useQuery({
    queryKey: ["workouts", showAllWorkouts],
    queryFn: () => workoutApi.list(1, showAllWorkouts ? 100 : 10),
    enabled: isAuthenticated,
    refetchInterval: (data) => {
      const hasProcessing = data?.state?.data?.items?.some(
        (w: any) => ["pending", "queued", "processing", "analyzing"].includes(w.processing_status)
      );
      return hasProcessing ? 2000 : 30000;
    },
  });

  const { data: trendsData } = useQuery({
    queryKey: ["trends"],
    queryFn: () => analyticsApi.getTrends(30),
    enabled: isAuthenticated,
  });

  // Delete workouts mutation
  const deleteWorkoutsMutation = useMutation({
    mutationFn: (ids: string[]) => workoutApi.deleteMultiple(ids),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workouts"] });
      queryClient.invalidateQueries({ queryKey: ["trends"] });
      setSelectedWorkouts(new Set());
      setEditMode(false);
      setShowDeleteConfirm(false);
    },
  });

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/login");
  };

  const handleUploadComplete = (workoutId: string) => {
    setShowUpload(false);
    refetchWorkouts();
  };

  const scrollToTop = () => {
    topRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleWorkoutSelection = (id: string) => {
    const newSelected = new Set(selectedWorkouts);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedWorkouts(newSelected);
  };

  const handleDeleteSelected = () => {
    if (selectedWorkouts.size > 0) {
      setShowDeleteConfirm(true);
    }
  };

  const confirmDelete = () => {
    deleteWorkoutsMutation.mutate(Array.from(selectedWorkouts));
  };

  const handleSaveName = () => {
    if (editedName.trim()) {
      updateUserMutation.mutate(editedName.trim());
    }
  };

  const startEditingName = () => {
    setEditedName(userData?.full_name || "");
    setIsEditingName(true);
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-kb-bg">
        <div className="animate-spin w-8 h-8 border-4 border-kb-accent border-t-transparent rounded-full" />
      </div>
    );
  }

  const workouts = workoutsData?.items || [];
  const displayedWorkouts = showAllWorkouts ? workouts : workouts.slice(0, 3);
  const summary = trendsData?.summary;
  const userName = userData?.full_name || userData?.email?.split("@")[0] || "User";

  return (
    <div className="min-h-screen bg-kb-bg" ref={topRef}>
      {/* Header */}
      <header className="border-b border-kb-border bg-kb-surface/80 backdrop-blur-xl sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo - Click to scroll to top */}
            <button 
              onClick={scrollToTop}
              className="flex items-center gap-3 hover:opacity-80 transition-opacity"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-kb-accent to-orange-600 flex items-center justify-center shadow-lg shadow-kb-accent/20">
                <Dumbbell className="w-5 h-5 text-white" />
              </div>
              <div className="text-left">
                <h1 className="font-display text-lg font-bold">Kettlebell Counter</h1>
                <p className="text-xs text-kb-muted">Competition Rep Tracking</p>
              </div>
            </button>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 p-2 hover:bg-kb-card rounded-lg transition-colors"
              >
                <div className="w-8 h-8 rounded-full bg-kb-accent/20 flex items-center justify-center">
                  <User className="w-4 h-4 text-kb-accent" />
                </div>
                <span className="text-sm font-medium hidden sm:block">{userName}</span>
              </button>
              
              <AnimatePresence>
                {showUserMenu && (
                  <>
                    <div 
                      className="fixed inset-0 z-40" 
                      onClick={() => {
                        setShowUserMenu(false);
                        setIsEditingName(false);
                      }} 
                    />
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      className="absolute right-0 mt-2 w-64 bg-kb-card border border-kb-border rounded-xl shadow-xl z-50 overflow-hidden"
                    >
                      <div className="p-4 border-b border-kb-border">
                        <div className="flex items-center gap-3 mb-3">
                          <div className="w-12 h-12 rounded-full bg-kb-accent/20 flex items-center justify-center">
                            <User className="w-6 h-6 text-kb-accent" />
                          </div>
                          <div className="flex-1">
                            {isEditingName ? (
                              <div className="flex items-center gap-2">
                                <input
                                  type="text"
                                  value={editedName}
                                  onChange={(e) => setEditedName(e.target.value)}
                                  className="flex-1 bg-kb-surface border border-kb-border rounded px-2 py-1 text-sm focus:outline-none focus:border-kb-accent"
                                  placeholder="Your name"
                                  autoFocus
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") handleSaveName();
                                    if (e.key === "Escape") setIsEditingName(false);
                                  }}
                                />
                                <button
                                  onClick={handleSaveName}
                                  className="p-1 hover:bg-kb-surface rounded text-kb-success"
                                >
                                  <Check className="w-4 h-4" />
                                </button>
                                <button
                                  onClick={() => setIsEditingName(false)}
                                  className="p-1 hover:bg-kb-surface rounded text-kb-danger"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                            ) : (
                              <div className="flex items-center gap-2">
                                <span className="font-medium">{userName}</span>
                                <button
                                  onClick={startEditingName}
                                  className="p-1 hover:bg-kb-surface rounded text-kb-muted hover:text-white"
                                >
                                  <Pencil className="w-3 h-3" />
                                </button>
                              </div>
                            )}
                            <p className="text-xs text-kb-muted truncate">{userData?.email}</p>
                          </div>
                        </div>
                      </div>
                      <div className="p-2">
                        <button
                          onClick={handleLogout}
                          className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-kb-muted hover:text-white hover:bg-kb-surface rounded-lg transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          Sign Out
                        </button>
                      </div>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 pb-24">
        {/* Upload modal */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
              onClick={() => setShowUpload(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-lg"
              >
                <VideoUpload onUploadComplete={handleUploadComplete} />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Delete confirmation modal */}
        <AnimatePresence>
          {showDeleteConfirm && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
              onClick={() => setShowDeleteConfirm(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
                className="bg-kb-card border border-kb-border rounded-xl p-6 max-w-sm w-full"
              >
                <h3 className="font-display text-lg font-semibold mb-2">Delete Workouts?</h3>
                <p className="text-kb-muted text-sm mb-6">
                  Are you sure you want to delete {selectedWorkouts.size} workout{selectedWorkouts.size > 1 ? "s" : ""}? 
                  This action cannot be undone.
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowDeleteConfirm(false)}
                    className="flex-1 px-4 py-2 bg-kb-surface hover:bg-kb-border rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmDelete}
                    disabled={deleteWorkoutsMutation.isPending}
                    className="flex-1 px-4 py-2 bg-kb-danger hover:bg-red-600 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {deleteWorkoutsMutation.isPending ? (
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <>
                        <Trash2 className="w-4 h-4" />
                        Delete
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Summary stats */}
        {summary && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
          >
            <div className="card p-5">
              <div className="flex items-center gap-2 text-kb-muted mb-2">
                <Activity className="w-4 h-4" />
                <span className="text-xs uppercase tracking-wider">30-Day Total</span>
              </div>
              <div className="font-display text-3xl font-bold">
                {summary.total_attempts}
              </div>
              <div className="text-sm text-kb-muted mt-1">attempts</div>
            </div>

            <div className="card p-5 border-kb-success/30">
              <div className="flex items-center gap-2 text-kb-success mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs uppercase tracking-wider">Valid Reps</span>
              </div>
              <div className="font-display text-3xl font-bold text-kb-success">
                {summary.total_valid}
              </div>
              <div className="text-sm text-kb-muted mt-1">
                {((summary?.overall_valid_rate ?? 0) * 100).toFixed(1)}% rate
              </div>
            </div>

            <div className="card p-5 border-kb-danger/30">
              <div className="flex items-center gap-2 text-kb-danger mb-2">
                <span className="text-xs uppercase tracking-wider">No-Reps</span>
              </div>
              <div className="font-display text-3xl font-bold text-kb-danger">
                {summary.total_no_reps}
              </div>
              <div className="text-sm text-kb-muted mt-1">to review</div>
            </div>

            <div className="card p-5">
              <div className="flex items-center gap-2 text-kb-muted mb-2">
                <span className="text-xs uppercase tracking-wider">Workouts</span>
              </div>
              <div className="font-display text-3xl font-bold">
                {trendsData?.total_workouts || 0}
              </div>
              <div className="text-sm text-kb-muted mt-1">this month</div>
            </div>
          </motion.div>
        )}

        {/* Recent workouts */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display text-xl font-semibold">Recent Workouts</h2>
            <div className="flex items-center gap-3">
              {workouts.length > 0 && (
                <button
                  onClick={() => {
                    setEditMode(!editMode);
                    if (editMode) {
                      setSelectedWorkouts(new Set());
                    }
                  }}
                  className={cn(
                    "text-sm px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5",
                    editMode 
                      ? "bg-kb-accent text-white" 
                      : "text-kb-muted hover:text-white hover:bg-kb-surface"
                  )}
                >
                  <Edit2 className="w-3.5 h-3.5" />
                  {editMode ? "Done" : "Edit"}
                </button>
              )}
              {workouts.length > 3 && (
                <button 
                  onClick={() => setShowAllWorkouts(!showAllWorkouts)}
                  className="text-sm text-kb-accent hover:underline flex items-center gap-1"
                >
                  {showAllWorkouts ? (
                    <>
                      Show less
                      <ChevronUp className="w-4 h-4" />
                    </>
                  ) : (
                    <>
                      View all ({workouts.length})
                      <ChevronDown className="w-4 h-4" />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Edit mode actions */}
          <AnimatePresence>
            {editMode && selectedWorkouts.size > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-4"
              >
                <div className="flex items-center justify-between bg-kb-surface rounded-lg p-3">
                  <span className="text-sm text-kb-muted">
                    {selectedWorkouts.size} selected
                  </span>
                  <button
                    onClick={handleDeleteSelected}
                    className="flex items-center gap-2 px-3 py-1.5 bg-kb-danger/20 hover:bg-kb-danger/30 text-kb-danger rounded-lg transition-colors text-sm"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete selected
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {workouts.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card p-12 text-center"
            >
              <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-kb-accent/20 to-orange-600/20 flex items-center justify-center">
                <Upload className="w-12 h-12 text-kb-accent" />
              </div>
              <h3 className="font-display text-2xl font-semibold mb-3">
                Ready to analyze your form?
              </h3>
              <p className="text-kb-muted mb-8 max-w-md mx-auto leading-relaxed">
                Upload a kettlebell workout video to get detailed rep counting, form analysis, 
                and no-rep detection with explainable feedback.
              </p>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowUpload(true)}
                className="btn-primary inline-flex items-center gap-2 px-6 py-3"
              >
                <Upload className="w-5 h-5" />
                Upload Your First Workout
              </motion.button>
              <p className="text-xs text-kb-muted mt-4">
                Supports MP4, MOV up to 60 minutes
              </p>
            </motion.div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {displayedWorkouts.map((workout, i) => (
                <div key={workout.id} className="relative">
                  {editMode && (
                    <button
                      onClick={() => toggleWorkoutSelection(workout.id)}
                      className={cn(
                        "absolute -left-2 -top-2 z-10 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all",
                        selectedWorkouts.has(workout.id)
                          ? "bg-kb-accent border-kb-accent"
                          : "bg-kb-card border-kb-border hover:border-kb-accent"
                      )}
                    >
                      {selectedWorkouts.has(workout.id) && (
                        <Check className="w-4 h-4 text-white" />
                      )}
                    </button>
                  )}
                  <WorkoutCard 
                    workout={workout} 
                    index={i} 
                    selectable={editMode}
                    selected={selectedWorkouts.has(workout.id)}
                    onSelect={() => toggleWorkoutSelection(workout.id)}
                  />
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      {/* Floating Action Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowUpload(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-br from-kb-accent to-orange-600 shadow-lg shadow-kb-accent/30 flex items-center justify-center z-30 hover:shadow-xl hover:shadow-kb-accent/40 transition-shadow"
      >
        <Plus className="w-6 h-6 text-white" />
      </motion.button>
    </div>
  );
}
