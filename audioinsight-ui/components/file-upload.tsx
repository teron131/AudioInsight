"use client";

import { cn } from '@/lib/utils';
import { FileAudio, Upload } from 'lucide-react';
import { useCallback, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  className?: string;
}

export function FileUpload({ onFileSelect, disabled, className }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    const audioFile = files.find(file => file.type.startsWith('audio/'));
    
    if (audioFile) {
      onFileSelect(audioFile);
    }
  }, [disabled, onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
      // Reset input
      e.target.value = '';
    }
  }, [onFileSelect]);

  return (
    <div
      className={cn(
        "border-2 border-dashed border-border rounded-lg p-8 text-center transition-all",
        "hover:border-muted-foreground hover:bg-accent/50",
        isDragOver && "border-blue-500 bg-blue-50 dark:bg-blue-950/20",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center gap-4">
        <div className="p-4 rounded-full bg-secondary">
          {isDragOver ? (
            <FileAudio className="w-8 h-8 text-blue-600" />
          ) : (
            <Upload className="w-8 h-8 text-muted-foreground" />
          )}
        </div>
        
        <div className="space-y-2">
          <h3 className="text-lg font-medium">
            {isDragOver ? 'Drop your audio file here' : 'Upload Audio File'}
          </h3>
          <p className="text-sm text-muted-foreground">
            Drag and drop an audio file, or click to browse
          </p>
          <p className="text-xs text-muted-foreground">
            Supports MP3, WAV, M4A, and other audio formats
          </p>
        </div>

        <label className="cursor-pointer">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileInput}
            disabled={disabled}
            className="hidden"
          />
          <div className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
            Browse Files
          </div>
        </label>
      </div>
    </div>
  );
} 