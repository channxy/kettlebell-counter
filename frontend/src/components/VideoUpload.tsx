"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Film, X, Loader2 } from "lucide-react";
import { videoApi } from "@/lib/api";
import { cn, getLiftTypeLabel } from "@/lib/utils";

interface VideoUploadProps {
  onUploadComplete: (workoutId: string) => void;
  className?: string;
}

const LIFT_TYPES = [
  { value: "auto", label: "Auto-Detect" },
  { value: "jerk", label: "Jerk" },
  { value: "long_cycle", label: "Long Cycle" },
  { value: "snatch", label: "Snatch" },
];

export function VideoUpload({ onUploadComplete, className }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [liftType, setLiftType] = useState<string>("auto");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isVideoFile(droppedFile)) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError("Please upload a video file (MP4, MOV, AVI, MKV)");
    }
  }, []);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (selectedFile && isVideoFile(selectedFile)) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError("Please upload a video file (MP4, MOV, AVI, MKV)");
      }
    },
    []
  );

  const isVideoFile = (file: File) => {
    const validTypes = ["video/mp4", "video/quicktime", "video/avi", "video/x-matroska"];
    return validTypes.includes(file.type) || /\.(mp4|mov|avi|mkv)$/i.test(file.name);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const workout = await videoApi.upload(file, liftType);
      onUploadComplete(workout.id);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setError(null);
  };

  return (
    <div className={cn("card p-6", className)}>
      <h2 className="font-display text-xl font-semibold mb-4">Upload Workout Video</h2>

      {/* Drop zone */}
      <div
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 text-center transition-all",
          isDragging
            ? "border-kb-accent bg-kb-accent/10"
            : "border-kb-border hover:border-kb-accent/50",
          file && "border-kb-success bg-kb-success/5"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime,video/avi,video/x-matroska,.mp4,.mov,.avi,.mkv"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />

        <AnimatePresence mode="wait">
          {file ? (
            <motion.div
              key="file"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center"
            >
              <Film className="w-12 h-12 text-kb-success mb-3" />
              <p className="font-medium text-white mb-1">{file.name}</p>
              <p className="text-sm text-kb-muted">{formatFileSize(file.size)}</p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  clearFile();
                }}
                className="mt-3 text-sm text-kb-danger hover:underline"
              >
                Remove
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center"
            >
              <Upload
                className={cn(
                  "w-12 h-12 mb-3 transition-colors",
                  isDragging ? "text-kb-accent" : "text-kb-muted"
                )}
              />
              <p className="text-white mb-1">
                Drop your workout video here, or{" "}
                <span className="text-kb-accent">browse</span>
              </p>
              <p className="text-sm text-kb-muted">MP4, MOV, AVI, MKV up to 5GB</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Error message */}
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 text-sm text-kb-danger"
        >
          {error}
        </motion.p>
      )}

      {/* Lift type selector */}
      <div className="mt-6">
        <label className="block text-sm text-kb-muted mb-2">Lift Type</label>
        <div className="flex gap-2">
          {LIFT_TYPES.map((type) => (
            <button
              key={type.value}
              onClick={() => setLiftType(type.value)}
              className={cn(
                "flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                liftType === type.value
                  ? "bg-kb-accent text-white"
                  : "bg-kb-surface border border-kb-border text-kb-muted hover:border-kb-accent/50"
              )}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {/* Upload button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleUpload}
        disabled={!file || isUploading}
        className={cn(
          "w-full mt-6 btn-primary flex items-center justify-center gap-2",
          (!file || isUploading) && "opacity-50 cursor-not-allowed"
        )}
      >
        {isUploading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Uploading...
          </>
        ) : (
          <>
            <Upload className="w-5 h-5" />
            Upload & Analyze
          </>
        )}
      </motion.button>

      {/* Upload progress */}
      {isUploading && (
        <div className="mt-4">
          <div className="h-2 bg-kb-surface rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-kb-accent"
              initial={{ width: 0 }}
              animate={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="text-sm text-kb-muted text-center mt-2">
            Processing will begin after upload...
          </p>
        </div>
      )}
    </div>
  );
}

