"use client"

import { useState } from "react"
import { Button, type ButtonProps } from "@/components/ui/button"
import { Copy, Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface CopyButtonProps extends ButtonProps {
  textToCopy: string
  buttonText?: string
  tooltipText?: string
}

export function CopyButton({
  textToCopy,
  buttonText = "Copy",
  tooltipText = "Copy to clipboard",
  className,
  size,
  variant,
  ...props
}: CopyButtonProps) {
  const [hasCopied, setHasCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy)
      setHasCopied(true)
      setTimeout(() => setHasCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy text: ", err)
      // Optionally, show an error state to the user
    }
  }

  return (
    <Button
      variant={variant || "outline"}
      size={size || "default"}
      onClick={handleCopy}
      className={cn("text-xs", className)}
      {...props}
      title={tooltipText}
    >
      {hasCopied ? <Check className="w-3.5 h-3.5 mr-1.5 text-green-600" /> : <Copy className="w-3.5 h-3.5 mr-1.5" />}
      {hasCopied ? "Copied!" : buttonText}
    </Button>
  )
}
