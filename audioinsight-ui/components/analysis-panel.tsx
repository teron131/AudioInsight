"use client";

import { Badge } from '@/components/ui/badge';
import { Analysis } from '@/hooks/use-audioinsight';
import { cn } from '@/lib/utils';

interface AnalysisPanelProps {
  analysis: Analysis | null;
  className?: string;
}

export function AnalysisPanel({ analysis, className }: AnalysisPanelProps) {
  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary Section */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-foreground">Summary</h3>
        <div className={cn(
          "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
          "hover:border-muted-foreground",
          analysis?.summary ? "text-foreground" : "text-muted-foreground"
        )}>
          {analysis?.summary || "The AI-generated summary will appear here..."}
        </div>
      </div>

      {/* Key Points Section */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-foreground">Key Points</h3>
        <div className={cn(
          "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
          "hover:border-muted-foreground",
          analysis?.key_points?.length ? "text-foreground" : "text-muted-foreground"
        )}>
          {analysis?.key_points?.length ? (
            <ul className="space-y-1">
              {analysis.key_points.map((point, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-blue-600 font-medium">•</span>
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          ) : (
            <ul>
              <li>Analysis points will appear here</li>
            </ul>
          )}
        </div>
      </div>

      {/* Keywords Section */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-foreground">Keywords</h3>
        <div className="flex flex-wrap gap-2">
          {analysis?.keywords?.length ? (
            analysis.keywords.map((keyword, index) => (
              <Badge 
                key={index}
                variant={index % 3 === 0 ? "default" : "secondary"}
                className="cursor-pointer hover:bg-blue-600 hover:text-white transition-colors"
              >
                {keyword}
              </Badge>
            ))
          ) : (
            <>
              <Badge variant="secondary">relevant</Badge>
              <Badge variant="default">keywords</Badge>
              <Badge variant="secondary">here</Badge>
            </>
          )}
        </div>
      </div>
    </div>
  );
} 