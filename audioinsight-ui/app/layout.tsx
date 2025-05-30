import { ThemeProvider } from "@/components/theme-provider"
import { ThemeToggle } from "@/components/ui/theme-toggle"
import { Toaster } from "@/components/ui/toaster"
import { cn } from "@/lib/utils"
import { Activity, Home, Settings } from "lucide-react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import Link from "next/link"
import type React from "react"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "AudioInsight",
  description: "Advanced audio transcription and analysis.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn(inter.className, "bg-background text-foreground")}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen">
            <aside className="w-16 border-r border-border bg-card flex flex-col items-center py-6 space-y-6">
              <Link
                href="/"
                className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
                title="Home"
              >
                <Home size={24} />
              </Link>
              <Link
                href="/settings"
                className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
                title="Settings"
              >
                <Settings size={24} />
              </Link>
            </aside>
            <div className="flex-1 flex flex-col">
              <header className="border-b border-border bg-card sticky top-0 z-10">
                <div className="container mx-auto px-6 h-16 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Activity className="w-7 h-7 text-blue-600" />
                    <h1 className="text-xl font-semibold text-foreground">AudioInsight</h1>
                  </div>
                  <ThemeToggle />
                </div>
              </header>
              <main className="flex-1 overflow-y-auto">{children}</main>
            </div>
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
