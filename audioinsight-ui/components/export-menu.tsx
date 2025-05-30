"use client";

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Download } from 'lucide-react';

interface ExportMenuProps {
  onExport: (format: 'txt' | 'srt' | 'vtt' | 'json') => void;
  disabled?: boolean;
  isIconOnly?: boolean;
}

export function ExportMenu({ onExport, disabled, isIconOnly }: ExportMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size={isIconOnly ? "icon" : "sm"} disabled={disabled}>
          <Download className={`w-4 h-4 ${!isIconOnly ? 'mr-2' : ''}`} />
          {!isIconOnly && 'Export'}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => onExport('txt')}>
          Export as TXT
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onExport('srt')}>
          Export as SRT
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onExport('vtt')}>
          Export as VTT
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onExport('json')}>
          Export as JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
} 