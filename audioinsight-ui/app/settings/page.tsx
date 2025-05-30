"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    // Model Configuration
    model: "base",
    backend: "faster-whisper",
    language: "auto",
    task: "transcribe",

    // Processing Configuration
    warmupFile: "",
    minChunkSize: 1.0,
    bufferTrimming: "sentence",
    bufferTrimmingSec: 15.0,
    vacChunkSize: 0.04,

    // Feature Configuration
    confidenceValidation: false,
    diarization: false, // Default to false
    noTranscription: false,
    vac: true,
    noVad: false,

    // LLM Configuration
    llmInference: true,
    fastLlm: "google/gemini-flash-1.5-8b",
    baseLlm: "openai/gpt-4o-mini",
    llmTriggerTime: 5.0,
    llmConversationTrigger: 2,

    // General Settings
    apiKey: "",
    autoScroll: true,
    darkMode: false,
  })

  // Load settings from localStorage on component mount
  useEffect(() => {
    const savedSettings = localStorage.getItem("audioinsight-settings")
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings)
        setSettings((prev) => ({ ...prev, ...parsed }))
      } catch (error) {
        console.error("Failed to parse saved settings:", error)
      }
    }
  }, [])

  const handleSave = () => {
    // Save to localStorage
    localStorage.setItem("audioinsight-settings", JSON.stringify(settings))

    // Dispatch custom event to notify other components
    window.dispatchEvent(new CustomEvent("settings-updated", { detail: settings }))

    console.log("Saving settings:", settings)
    alert("Settings saved successfully!")
  }

  const updateSetting = (key: string, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800">AudioInsight Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            {/* Model Configuration */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-medium text-gray-800">Model Configuration</h3>
                <Badge variant="outline" className="text-xs">
                  Core
                </Badge>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="model" className="text-sm font-medium text-gray-700">
                    Whisper Model
                  </Label>
                  <Select value={settings.model} onValueChange={(value) => updateSetting("model", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="tiny.en">tiny.en</SelectItem>
                      <SelectItem value="tiny">tiny</SelectItem>
                      <SelectItem value="base.en">base.en</SelectItem>
                      <SelectItem value="base">base</SelectItem>
                      <SelectItem value="small.en">small.en</SelectItem>
                      <SelectItem value="small">small</SelectItem>
                      <SelectItem value="medium.en">medium.en</SelectItem>
                      <SelectItem value="medium">medium</SelectItem>
                      <SelectItem value="large-v1">large-v1</SelectItem>
                      <SelectItem value="large-v2">large-v2</SelectItem>
                      <SelectItem value="large-v3">large-v3</SelectItem>
                      <SelectItem value="large">large</SelectItem>
                      <SelectItem value="large-v3-turbo">large-v3-turbo</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500">Size of the Whisper model to use</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="backend" className="text-sm font-medium text-gray-700">
                    Backend
                  </Label>
                  <Select value={settings.backend} onValueChange={(value) => updateSetting("backend", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select backend" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="faster-whisper">faster-whisper</SelectItem>
                      <SelectItem value="whisper_timestamped">whisper_timestamped</SelectItem>
                      <SelectItem value="mlx-whisper">mlx-whisper</SelectItem>
                      <SelectItem value="openai-api">openai-api</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500">Whisper processing backend</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="language" className="text-sm font-medium text-gray-700">
                    Language
                  </Label>
                  <Input
                    id="language"
                    value={settings.language}
                    onChange={(e) => updateSetting("language", e.target.value)}
                    placeholder="auto, en, de, cs..."
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Source language code or 'auto' for detection</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="task" className="text-sm font-medium text-gray-700">
                    Task
                  </Label>
                  <Select value={settings.task} onValueChange={(value) => updateSetting("task", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select task" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="transcribe">Transcribe</SelectItem>
                      <SelectItem value="translate">Translate</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500">Transcribe or translate audio</p>
                </div>
              </div>
            </div>

            <Separator />

            {/* Processing Configuration */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-medium text-gray-800">Processing Configuration</h3>
                <Badge variant="outline" className="text-xs">
                  Performance
                </Badge>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="warmup-file" className="text-sm font-medium text-gray-700">
                    Warmup File Path
                  </Label>
                  <Input
                    id="warmup-file"
                    value={settings.warmupFile}
                    onChange={(e) => updateSetting("warmupFile", e.target.value)}
                    placeholder="/path/to/warmup.wav"
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Audio file to warm up Whisper for faster first chunk</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="min-chunk-size" className="text-sm font-medium text-gray-700">
                    Min Chunk Size (seconds)
                  </Label>
                  <Input
                    id="min-chunk-size"
                    type="number"
                    step="0.1"
                    value={settings.minChunkSize}
                    onChange={(e) => updateSetting("minChunkSize", Number.parseFloat(e.target.value))}
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Minimum audio chunk size for processing</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="buffer-trimming" className="text-sm font-medium text-gray-700">
                    Buffer Trimming Strategy
                  </Label>
                  <Select
                    value={settings.bufferTrimming}
                    onValueChange={(value) => updateSetting("bufferTrimming", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select strategy" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sentence">Sentence</SelectItem>
                      <SelectItem value="segment">Segment</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500">How to trim the audio buffer</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="buffer-trimming-sec" className="text-sm font-medium text-gray-700">
                    Buffer Trimming Threshold (seconds)
                  </Label>
                  <Input
                    id="buffer-trimming-sec"
                    type="number"
                    step="0.1"
                    value={settings.bufferTrimmingSec}
                    onChange={(e) => updateSetting("bufferTrimmingSec", Number.parseFloat(e.target.value))}
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Buffer length threshold for trimming</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="vac-chunk-size" className="text-sm font-medium text-gray-700">
                    VAC Chunk Size (seconds)
                  </Label>
                  <Input
                    id="vac-chunk-size"
                    type="number"
                    step="0.01"
                    value={settings.vacChunkSize}
                    onChange={(e) => updateSetting("vacChunkSize", Number.parseFloat(e.target.value))}
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Voice activity controller sample size</p>
                </div>
              </div>
            </div>

            <Separator />

            {/* Feature Configuration */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-medium text-gray-800">Feature Configuration</h3>
                <Badge variant="outline" className="text-xs">
                  Features
                </Badge>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                  <div>
                    <Label htmlFor="confidence-validation" className="text-sm font-medium text-gray-700">
                      Confidence Validation
                    </Label>
                    <p className="text-xs text-gray-500">Faster transcription, less accurate punctuation</p>
                  </div>
                  <Switch
                    id="confidence-validation"
                    checked={settings.confidenceValidation}
                    onCheckedChange={(checked) => updateSetting("confidenceValidation", checked)}
                  />
                </div>

                <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                  <div>
                    <Label htmlFor="diarization" className="text-sm font-medium text-gray-700">
                      Speaker Diarization
                    </Label>
                    <p className="text-xs text-gray-500">Identify and separate different speakers</p>
                  </div>
                  <Switch
                    id="diarization"
                    checked={settings.diarization}
                    onCheckedChange={(checked) => updateSetting("diarization", checked)}
                  />
                </div>

                <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                  <div>
                    <Label htmlFor="vac" className="text-sm font-medium text-gray-700">
                      Voice Activity Controller
                    </Label>
                    <p className="text-xs text-gray-500">Recommended. Requires torch</p>
                  </div>
                  <Switch
                    id="vac"
                    checked={settings.vac}
                    onCheckedChange={(checked) => updateSetting("vac", checked)}
                  />
                </div>

                <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                  <div>
                    <Label htmlFor="no-vad" className="text-sm font-medium text-gray-700">
                      Disable VAD
                    </Label>
                    <p className="text-xs text-gray-500">Turn off voice activity detection</p>
                  </div>
                  <Switch
                    id="no-vad"
                    checked={settings.noVad}
                    onCheckedChange={(checked) => updateSetting("noVad", checked)}
                  />
                </div>
              </div>
            </div>

            <Separator />

            {/* LLM Configuration */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-medium text-gray-800">LLM Inference Configuration</h3>
                <Badge variant="outline" className="text-xs">
                  AI Analysis
                </Badge>
              </div>

              <div className="flex items-center justify-between space-x-2 border p-4 rounded-md mb-4">
                <div>
                  <Label htmlFor="llm-inference" className="text-sm font-medium text-gray-700">
                    Enable LLM Inference
                  </Label>
                  <p className="text-xs text-gray-500">AI-based analysis after periods of inactivity</p>
                </div>
                <Switch
                  id="llm-inference"
                  checked={settings.llmInference}
                  onCheckedChange={(checked) => updateSetting("llmInference", checked)}
                />
              </div>

              {settings.llmInference && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="fast-llm" className="text-sm font-medium text-gray-700">
                      Fast LLM Model
                    </Label>
                    <Input
                      id="fast-llm"
                      value={settings.fastLlm}
                      onChange={(e) => updateSetting("fastLlm", e.target.value)}
                      className="text-sm"
                    />
                    <p className="text-xs text-gray-500">For text parsing and quick operations</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="base-llm" className="text-sm font-medium text-gray-700">
                      Base LLM Model
                    </Label>
                    <Input
                      id="base-llm"
                      value={settings.baseLlm}
                      onChange={(e) => updateSetting("baseLlm", e.target.value)}
                      className="text-sm"
                    />
                    <p className="text-xs text-gray-500">For summarization and complex operations</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="llm-trigger-time" className="text-sm font-medium text-gray-700">
                      LLM Trigger Time (seconds)
                    </Label>
                    <Input
                      id="llm-trigger-time"
                      type="number"
                      step="0.1"
                      value={settings.llmTriggerTime}
                      onChange={(e) => updateSetting("llmTriggerTime", Number.parseFloat(e.target.value))}
                      className="text-sm"
                    />
                    <p className="text-xs text-gray-500">Time after which to trigger inference</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="llm-conversation-trigger" className="text-sm font-medium text-gray-700">
                      Conversation Trigger Count
                    </Label>
                    <Input
                      id="llm-conversation-trigger"
                      type="number"
                      value={settings.llmConversationTrigger}
                      onChange={(e) => updateSetting("llmConversationTrigger", Number.parseInt(e.target.value))}
                      className="text-sm"
                    />
                    <p className="text-xs text-gray-500">Speaker turns after which to trigger inference</p>
                  </div>
                </div>
              )}
            </div>

            <Separator />

            {/* General Settings */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-medium text-gray-800">General Settings</h3>
                <Badge variant="outline" className="text-xs">
                  UI/UX
                </Badge>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="api-key" className="text-sm font-medium text-gray-700">
                    OpenAI API Key
                  </Label>
                  <Input
                    id="api-key"
                    type="password"
                    value={settings.apiKey}
                    onChange={(e) => updateSetting("apiKey", e.target.value)}
                    placeholder="sk-..."
                    className="text-sm"
                  />
                  <p className="text-xs text-gray-500">Your API key for OpenAI services</p>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                    <div>
                      <Label htmlFor="auto-scroll" className="text-sm font-medium text-gray-700">
                        Auto-Scroll Transcription
                      </Label>
                      <p className="text-xs text-gray-500">Automatically scroll as text appears</p>
                    </div>
                    <Switch
                      id="auto-scroll"
                      checked={settings.autoScroll}
                      onCheckedChange={(checked) => updateSetting("autoScroll", checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between space-x-2 border p-4 rounded-md">
                    <div>
                      <Label htmlFor="dark-mode" className="text-sm font-medium text-gray-700">
                        Dark Mode
                      </Label>
                      <p className="text-xs text-gray-500">Toggle dark theme</p>
                    </div>
                    <Switch
                      id="dark-mode"
                      checked={settings.darkMode}
                      onCheckedChange={(checked) => updateSetting("darkMode", checked)}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-6 flex justify-end space-x-3">
              <Button variant="outline" onClick={() => window.location.reload()}>
                Reset to Defaults
              </Button>
              <Button onClick={handleSave}>Save Configuration</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
