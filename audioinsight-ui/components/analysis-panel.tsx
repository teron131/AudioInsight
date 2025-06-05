"use client";

import { AnalysisData } from '@/hooks/use-audioinsight';
import { cn } from '@/lib/utils';

interface AnalysisPanelProps {
  analysis: AnalysisData | null;
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
              {analysis.key_points.map((point: string, index: number) => (
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

      {/* Response Suggestions Section */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-foreground">Response Suggestions</h3>
        <div className={cn(
          "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
          "hover:border-muted-foreground",
          analysis?.response_suggestions?.length ? "text-foreground" : "text-muted-foreground"
        )}>
          {analysis?.response_suggestions?.length ? (
            <ul className="space-y-1">
              {analysis.response_suggestions.map((suggestion: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-green-600 font-medium">💬</span>
                  <span>{suggestion}</span>
                </li>
              ))}
            </ul>
          ) : (
            <ul>
              <li>Response suggestions will appear here</li>
            </ul>
          )}
        </div>
      </div>

      {/* Action Plan Section */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-foreground">Action Plan</h3>
        <div className={cn(
          "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
          "hover:border-muted-foreground",
          analysis?.action_plan?.length ? "text-foreground" : "text-muted-foreground"
        )}>
          {analysis?.action_plan?.length ? (
            <ul className="space-y-1">
              {analysis.action_plan.map((action: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-purple-600 font-medium">📋</span>
                  <span>{action}</span>
                </li>
              ))}
            </ul>
          ) : (
            <ul>
              <li>Action recommendations will appear here</li>
            </ul>
          )}
        </div>
      </div>
    </div>
  );
} 