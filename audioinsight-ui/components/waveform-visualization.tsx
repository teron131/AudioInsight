"use client";

import { cn } from '@/lib/utils';
import { useEffect, useState } from 'react';

interface WaveformVisualizationProps {
  isActive: boolean;
  className?: string;
}

export function WaveformVisualization({ isActive, className }: WaveformVisualizationProps) {
  const [barHeights, setBarHeights] = useState<number[]>(Array(12).fill(8));

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (isActive) {
      intervalId = setInterval(() => {
        setBarHeights(prev => 
          prev.map(() => Math.random() * 32 + 8)
        );
      }, 400);
    } else {
      setBarHeights(Array(12).fill(8));
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isActive]);

  return (
    <div className={cn(
      "bg-secondary border border-border rounded-lg p-8 mb-5 flex items-center justify-center min-h-[120px] relative transition-all",
      "hover:border-muted-foreground",
      className
    )}>
      {/* Recording indicator */}
      {isActive && (
        <div className="absolute top-3 right-3 bg-destructive text-destructive-foreground px-2 py-1 rounded text-xs font-medium animate-pulse">
          ● REC
        </div>
      )}
      
      {/* Waveform bars */}
      <div className="flex items-center gap-0.5 h-10">
        {barHeights.map((height, index) => (
          <div
            key={index}
            className="w-0.5 bg-blue-600 rounded-sm transition-all duration-100"
            style={{ 
              height: `${height}px`,
              animationDelay: `${index * 0.1}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
} 