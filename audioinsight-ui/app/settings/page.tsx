"use client"

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AudioInsightAPI, ConfigurationPreset, LLMStatus, ModelStatus, ProcessingParameters, UploadedFile } from '@/lib/api';
import { AlertCircle, Brain, CheckCircle, FileText, RefreshCw, Settings, TestTube, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function SettingsPage() {
  const [api] = useState(() => new AudioInsightAPI());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // State for different sections
  const [modelsStatus, setModelsStatus] = useState<ModelStatus | null>(null);
  const [processingParams, setProcessingParams] = useState<ProcessingParameters | null>(null);
  const [presets, setPresets] = useState<Record<string, ConfigurationPreset>>({});
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [llmStatus, setLLMStatus] = useState<LLMStatus | null>(null);

  // Load initial data
  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        loadModelsStatus(),
        loadProcessingParameters(),
        loadPresets(),
        loadUploadedFiles(),
        loadLLMStatus(),
      ]);
    } catch (err) {
      setError(`Failed to load data: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const loadModelsStatus = async () => {
    try {
      const status = await api.getModelsStatus();
      setModelsStatus(status);
    } catch (err) {
      console.error('Failed to load models status:', err);
    }
  };

  const loadProcessingParameters = async () => {
    try {
      const params = await api.getProcessingParameters();
      setProcessingParams(params);
    } catch (err) {
      console.error('Failed to load processing parameters:', err);
    }
  };

  const loadPresets = async () => {
    try {
      const presetsData = await api.getConfigurationPresets();
      setPresets(presetsData);
    } catch (err) {
      console.error('Failed to load presets:', err);
    }
  };

  const loadUploadedFiles = async () => {
    try {
      const files = await api.getUploadedFiles();
      setUploadedFiles(files);
    } catch (err) {
      console.error('Failed to load uploaded files:', err);
    }
  };

  const loadLLMStatus = async () => {
    try {
      const status = await api.getLLMStatus();
      setLLMStatus(status);
    } catch (err) {
      console.error('Failed to load LLM status:', err);
    }
  };

  const showSuccess = (message: string) => {
    setSuccess(message);
    setTimeout(() => setSuccess(null), 3000);
  };

  const showError = (message: string) => {
    setError(message);
    setTimeout(() => setError(null), 5000);
  };

  const handleReloadModels = async (modelType: 'all' | 'asr' | 'diarization') => {
    try {
      setLoading(true);
      const reloaded = await api.reloadModels(modelType);
      showSuccess(`Reloaded models: ${reloaded.join(', ')}`);
      await loadModelsStatus();
    } catch (err) {
      showError(`Failed to reload models: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyPreset = async (presetName: string) => {
    try {
      setLoading(true);
      const result = await api.applyConfigurationPreset(presetName);
      showSuccess(`Applied preset: ${result.preset}`);
      await loadProcessingParameters();
    } catch (err) {
      showError(`Failed to apply preset: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFile = async (filePath: string) => {
    try {
      await api.deleteUploadedFile(filePath);
      showSuccess('File deleted successfully');
      await loadUploadedFiles();
    } catch (err) {
      showError(`Failed to delete file: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleCleanupFiles = async () => {
    try {
      setLoading(true);
      const deletedFiles = await api.cleanupOldFiles(24);
      showSuccess(`Cleaned up ${deletedFiles.length} old files`);
      await loadUploadedFiles();
    } catch (err) {
      showError(`Failed to cleanup files: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTestLLM = async () => {
    try {
      setLoading(true);
      const result = await api.testLLMConnection();
      showSuccess(`LLM test successful: ${result.model} (${result.response_time.toFixed(2)}s)`);
    } catch (err) {
      showError(`LLM test failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: boolean, readyStatus?: boolean) => {
    if (readyStatus === false) {
      return <Badge variant="destructive">Not Ready</Badge>;
    }
    return status ? <Badge variant="default">Loaded</Badge> : <Badge variant="secondary">Not Loaded</Badge>;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AudioInsight Settings</h1>
          <p className="text-muted-foreground">Manage models, processing parameters, and system configuration</p>
        </div>
        <Button onClick={loadAllData} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh All
        </Button>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2 text-destructive">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {success && (
        <Card className="border-green-500">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2 text-green-600">
              <CheckCircle className="w-4 h-4" />
              <span>{success}</span>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="models" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="presets">Presets</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="llm">LLM</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>Model Status</span>
              </CardTitle>
              <CardDescription>Current status of all loaded models</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {modelsStatus && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm">ASR Model</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Status:</span>
                          {getStatusBadge(modelsStatus.asr.loaded, modelsStatus.asr.ready)}
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Model:</span>
                          <span className="text-sm font-mono">{modelsStatus.asr.model_name}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Backend:</span>
                          <span className="text-sm">{modelsStatus.asr.backend}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Language:</span>
                          <span className="text-sm">{modelsStatus.asr.language}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm">Diarization</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Status:</span>
                          {getStatusBadge(modelsStatus.diarization.loaded, modelsStatus.diarization.ready)}
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Enabled:</span>
                          <span className="text-sm">{modelsStatus.diarization.enabled ? 'Yes' : 'No'}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm">LLM Models</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Inference:</span>
                          <span className="text-sm">{modelsStatus.llm.inference_enabled ? 'Enabled' : 'Disabled'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Fast Model:</span>
                          <span className="text-xs font-mono">{modelsStatus.llm.fast_model}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Base Model:</span>
                          <span className="text-xs font-mono">{modelsStatus.llm.base_model}</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="flex space-x-2">
                    <Button 
                      onClick={() => handleReloadModels('all')} 
                      disabled={loading}
                      variant="outline"
                    >
                      Reload All Models
                    </Button>
                    <Button 
                      onClick={() => handleReloadModels('asr')} 
                      disabled={loading}
                      variant="outline"
                    >
                      Reload ASR
                    </Button>
                    <Button 
                      onClick={() => handleReloadModels('diarization')} 
                      disabled={loading}
                      variant="outline"
                    >
                      Reload Diarization
                    </Button>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>Processing Parameters</span>
              </CardTitle>
              <CardDescription>Configure audio processing settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {processingParams && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="min_chunk_size">Min Chunk Size (seconds)</Label>
                    <Input
                      id="min_chunk_size"
                      type="number"
                      step="0.1"
                      value={processingParams.min_chunk_size}
                      onChange={(e) => setProcessingParams(prev => prev ? { ...prev, min_chunk_size: parseFloat(e.target.value) } : null)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="buffer_trimming_sec">Buffer Trimming (seconds)</Label>
                    <Input
                      id="buffer_trimming_sec"
                      type="number"
                      step="0.1"
                      value={processingParams.buffer_trimming_sec}
                      onChange={(e) => setProcessingParams(prev => prev ? { ...prev, buffer_trimming_sec: parseFloat(e.target.value) } : null)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="language">Language</Label>
                    <Select value={processingParams.language} onValueChange={(value) => setProcessingParams(prev => prev ? { ...prev, language: value } : null)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select language" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">Auto Detect</SelectItem>
                        <SelectItem value="en">English</SelectItem>
                        <SelectItem value="es">Spanish</SelectItem>
                        <SelectItem value="fr">French</SelectItem>
                        <SelectItem value="de">German</SelectItem>
                        <SelectItem value="zh">Chinese</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="task">Task</Label>
                    <Select value={processingParams.task} onValueChange={(value) => setProcessingParams(prev => prev ? { ...prev, task: value } : null)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select task" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="transcribe">Transcribe</SelectItem>
                        <SelectItem value="translate">Translate</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="vad_enabled"
                      checked={processingParams.vad_enabled}
                      onCheckedChange={(checked) => setProcessingParams(prev => prev ? { ...prev, vad_enabled: checked } : null)}
                    />
                    <Label htmlFor="vad_enabled">Voice Activity Detection</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="vac_enabled"
                      checked={processingParams.vac_enabled}
                      onCheckedChange={(checked) => setProcessingParams(prev => prev ? { ...prev, vac_enabled: checked } : null)}
                    />
                    <Label htmlFor="vac_enabled">Voice Activity Controller</Label>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="presets" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>Configuration Presets</span>
              </CardTitle>
              <CardDescription>Quick configuration presets for different use cases</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(presets).map(([key, preset]) => (
                  <Card key={key} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">{preset.name}</CardTitle>
                      <CardDescription>{preset.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button 
                        onClick={() => handleApplyPreset(key)}
                        disabled={loading}
                        className="w-full"
                      >
                        Apply Preset
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="files" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>File Management</span>
                </div>
                <Button onClick={handleCleanupFiles} disabled={loading} variant="outline">
                  <Trash2 className="w-4 h-4 mr-2" />
                  Cleanup Old Files
                </Button>
              </CardTitle>
              <CardDescription>Manage uploaded audio files</CardDescription>
            </CardHeader>
            <CardContent>
              {uploadedFiles.length === 0 ? (
                <p className="text-muted-foreground text-center py-8">No uploaded files found</p>
              ) : (
                <div className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <div className="font-medium">{file.filename}</div>
                        <div className="text-sm text-muted-foreground">
                          {api.formatBytes(file.size)} • {api.formatTimestamp(file.created)}
                        </div>
                      </div>
                      <Button
                        onClick={() => handleDeleteFile(file.path)}
                        size="sm"
                        variant="destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="llm" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Brain className="w-5 h-5" />
                  <span>LLM Status</span>
                </div>
                <Button onClick={handleTestLLM} disabled={loading} variant="outline">
                  <TestTube className="w-4 h-4 mr-2" />
                  Test Connection
                </Button>
              </CardTitle>
              <CardDescription>LLM processing status and configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {llmStatus && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Display Parser</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Status:</span>
                        {llmStatus.display_parser.enabled ? 
                          <Badge variant="default">Enabled</Badge> : 
                          <Badge variant="secondary">Disabled</Badge>
                        }
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Model:</span>
                        <span className="text-xs font-mono">{llmStatus.display_parser.model}</span>
                      </div>
                      {llmStatus.display_parser.stats && (
                        <>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Requests:</span>
                            <span className="text-sm">{llmStatus.display_parser.stats.total_requests || 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Cache Hit Rate:</span>
                            <span className="text-sm">
                              {((llmStatus.display_parser.stats.cache_hit_rate || 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Inference</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Status:</span>
                        {llmStatus.inference.enabled ? 
                          <Badge variant="default">Enabled</Badge> : 
                          <Badge variant="secondary">Disabled</Badge>
                        }
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Fast Model:</span>
                        <span className="text-xs font-mono">{llmStatus.inference.fast_model}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Base Model:</span>
                        <span className="text-xs font-mono">{llmStatus.inference.base_model}</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
