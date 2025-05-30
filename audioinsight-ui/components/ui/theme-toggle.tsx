"use client"

import { Monitor, Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import * as React from "react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Only render after hydration to prevent theme mismatch
  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="inline-flex rounded-md border bg-transparent p-0.5">
        <Button variant="ghost" size="icon" className="h-8 w-8 rounded-sm">
          <Sun className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8 rounded-sm">
          <Moon className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8 rounded-sm">
          <Monitor className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  const options = [
    { name: "light", icon: Sun },
    { name: "dark", icon: Moon },
    { name: "system", icon: Monitor },
  ]

  // Determine the currently active theme, resolving "system" to actual theme
  const currentActiveTheme = theme === "system" ? resolvedTheme : theme

  return (
    <div className="inline-flex rounded-md border bg-transparent p-0.5">
      {options.map((opt) => {
        const Icon = opt.icon
        const isActive = opt.name === theme // Check against the set theme, not resolvedTheme for UI selection
        return (
          <Button
            key={opt.name}
            variant="ghost" // Use ghost variant for a flatter look initially
            size="icon"
            className={cn(
              "h-8 w-8 rounded-sm transition-colors",
              isActive
                ? "bg-gray-900 text-white dark:bg-white dark:text-black shadow-sm" // Explicit dark/light mode colors
                : "hover:bg-gray-200 hover:text-gray-900 dark:hover:bg-accent/50 dark:hover:text-muted-foreground text-muted-foreground",
            )}
            onClick={() => setTheme(opt.name)}
            title={`Set theme to ${opt.name}`}
          >
            <Icon className="h-4 w-4" />
            <span className="sr-only">{opt.name}</span>
          </Button>
        )
      })}
    </div>
  )
} 