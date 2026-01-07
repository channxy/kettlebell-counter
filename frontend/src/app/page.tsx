"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Plus, Dumbbell, TrendingUp, Activity, LogOut, Upload, Settings, User, Menu, X } from "lucide-react";
import { workoutApi, analyticsApi } from "@/lib/api";
import { VideoUpload } from "@/components/VideoUpload";
import { WorkoutCard } from "@/components/WorkoutCard";

export default function HomePage() {
  const router = useRouter();
  const [showUpload, setShowUpload] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    } else {
      setIsAuthenticated(true);
    }
  }, [router]);

  // All hooks must be called before any conditional returns
  const { data: workoutsData, refetch: refetchWorkouts } = useQuery({
    queryKey: ["workouts"],
    queryFn: () => workoutApi.list(1, 10),
    enabled: isAuthenticated,
    refetchInterval: (data) => {
      // Poll more frequently if any workout is still processing
      const hasProcessing = data?.state?.data?.items?.some(
        (w: any) => ["pending", "queued", "processing", "analyzing"].includes(w.processing_status)
      );
      return hasProcessing ? 2000 : 30000; // 2s when processing, 30s otherwise
    },
  });

  const { data: trendsData } = useQuery({
    queryKey: ["trends"],
    queryFn: () => analyticsApi.getTrends(30),
    enabled: isAuthenticated,
  });

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/login");
  };

  const handleUploadComplete = (workoutId: string) => {
    setShowUpload(false);
    refetchWorkouts();
  };

  // Conditional return AFTER all hooks
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-kb-bg">
        <div className="animate-spin w-8 h-8 border-4 border-kb-accent border-t-transparent rounded-full" />
      </div>
    );
  }

  const workouts = workoutsData?.items || [];
  const summary = trendsData?.summary;

  return (
    <div className="min-h-screen bg-kb-bg">
      {/* Header */}
      <header className="border-b border-kb-border bg-kb-surface/80 backdrop-blur-xl sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-kb-accent to-orange-600 flex items-center justify-center shadow-lg shadow-kb-accent/20">
                <Dumbbell className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-display text-lg font-bold">Kettlebell Counter</h1>
                <p className="text-xs text-kb-muted">Competition Rep Tracking</p>
              </div>
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 p-2 hover:bg-kb-card rounded-lg transition-colors"
              >
                <div className="w-8 h-8 rounded-full bg-kb-accent/20 flex items-center justify-center">
                  <User className="w-4 h-4 text-kb-accent" />
                </div>
              </button>
              
              <AnimatePresence>
                {showUserMenu && (
                  <>
                    <div 
                      className="fixed inset-0 z-40" 
                      onClick={() => setShowUserMenu(false)} 
                    />
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      className="absolute right-0 mt-2 w-48 bg-kb-card border border-kb-border rounded-xl shadow-xl z-50 overflow-hidden"
                    >
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
                {(summary.overall_valid_rate * 100).toFixed(1)}% rate
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
            <button className="text-sm text-kb-accent hover:underline">View all</button>
          </div>

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
              {workouts.map((workout, i) => (
                <WorkoutCard key={workout.id} workout={workout} index={i} />
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

      {/* FAB Label (shows on hover) */}
      <motion.div
        initial={{ opacity: 0, x: 10 }}
        whileHover={{ opacity: 1, x: 0 }}
        className="fixed bottom-8 right-24 bg-kb-card border border-kb-border px-3 py-1.5 rounded-lg shadow-lg z-30 pointer-events-none"
      >
        <span className="text-sm font-medium whitespace-nowrap">New Workout</span>
      </motion.div>
    </div>
  );
}
