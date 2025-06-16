"use client"

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { AudioInsightAPI, ProcessingParameters } from '@/lib/api';
import { AlertCircle, Brain, CheckCircle, RefreshCw, Save, Settings } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

export default function SettingsPage() {
  const [api] = useState(() => new AudioInsightAPI());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  // State for different sections
  const [processingParams, setProcessingParams] = useState<ProcessingParameters | null>(null);

  // Auto-save functionality
  const [hasChanges, setHasChanges] = useState(false);
  const [lastSaved, setLastSaved] = useState<ProcessingParameters | null>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const autoSaveIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const saveProcessingParameters = useCallback(async () => {
    if (!processingParams || !hasChanges) return;

    try {
      setSaveStatus('saving');
      await api.updateProcessingParameters(processingParams);
      setLastSaved({ ...processingParams });
      setHasChanges(false);
      setSaveStatus('saved');
      
      // Clear saved status after 2 seconds
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (err) {
      setSaveStatus('error');
      console.error('Failed to save processing parameters:', err);
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  }, [processingParams, hasChanges, api]);

  const updateProcessingParam = useCallback(<K extends keyof ProcessingParameters>(
    key: K, 
    value: ProcessingParameters[K]
  ) => {
    setProcessingParams(prev => {
      if (!prev) return null;
      const updated = { ...prev, [key]: value };
      
      // Check if anything actually changed
      const changed = JSON.stringify(updated) !== JSON.stringify(lastSaved);
      setHasChanges(changed);
      
      return updated;
    });
  }, [lastSaved]);

  // Load initial data
  useEffect(() => {
    loadAllData();
  }, []);

  // Auto-save effect
  useEffect(() => {
    if (hasChanges && processingParams) {
      // Clear existing timeout
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      // Set new timeout for 1 second
      saveTimeoutRef.current = setTimeout(() => {
        saveProcessingParameters();
      }, 1000);
    }

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [processingParams, hasChanges, saveProcessingParameters]);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      if (autoSaveIntervalRef.current) {
        clearTimeout(autoSaveIntervalRef.current);
      }
    };
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        loadProcessingParameters(),
      ]);
    } catch (err) {
      setError(`Failed to load data: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const loadProcessingParameters = async () => {
    try {
      const params = await api.getProcessingParameters();
      setProcessingParams(params);
      setLastSaved(params);
      setHasChanges(false);
    } catch (err) {
      console.error('Failed to load processing parameters:', err);
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

  const getSaveStatusBadge = () => {
    switch (saveStatus) {
      case 'saving':
        return <Badge variant="secondary" className="flex items-center gap-1"><Save className="w-3 h-3 animate-spin" />Saving...</Badge>;
      case 'saved':
        return <Badge variant="default" className="flex items-center gap-1"><CheckCircle className="w-3 h-3" />Saved</Badge>;
      case 'error':
        return <Badge variant="destructive" className="flex items-center gap-1"><AlertCircle className="w-3 h-3" />Save Error</Badge>;
      default:
        return hasChanges ? <Badge variant="outline">Unsaved Changes</Badge> : null;
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
        </div>
        <div className="flex items-center gap-2">
          {getSaveStatusBadge()}
          <Button onClick={loadAllData} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh All
          </Button>
        </div>
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

      {processingParams && (
        <div className="space-y-4">
                    {/* Feature Configuration */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>Feature Configuration</span>
              </CardTitle>
              <CardDescription>Enable or disable audio processing features</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="transcription"
                    checked={processingParams.transcription ?? true}
                    onCheckedChange={(checked) => updateProcessingParam('transcription', checked)}
                  />
                  <Label htmlFor="transcription">Transcription</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="diarization"
                    checked={processingParams.diarization ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('diarization', checked)}
                  />
                  <Label htmlFor="diarization">Speaker Diarization</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="vad_enabled"
                    checked={processingParams.vad_enabled ?? true}
                    onCheckedChange={(checked) => updateProcessingParam('vad_enabled', checked)}
                  />
                  <Label htmlFor="vad_enabled">Voice Activity Detection</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="vac_enabled"
                    checked={processingParams.vac_enabled ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('vac_enabled', checked)}
                  />
                  <Label htmlFor="vac_enabled">Voice Activity Controller</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="confidence_validation"
                    checked={processingParams.confidence_validation ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('confidence_validation', checked)}
                  />
                  <Label htmlFor="confidence_validation">Confidence Validation</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="llm_inference"
                    checked={processingParams.llm_inference ?? true}
                    onCheckedChange={(checked) => updateProcessingParam('llm_inference', checked)}
                  />
                  <Label htmlFor="llm_inference">LLM Inference</Label>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* UI Configuration */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>UI Configuration</span>
              </CardTitle>
              <CardDescription>User interface display settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="show_lag_info"
                    checked={processingParams.show_lag_info ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('show_lag_info', checked)}
                  />
                  <Label htmlFor="show_lag_info">Show Lag Information</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="show_speakers"
                    checked={processingParams.show_speakers ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('show_speakers', checked)}
                  />
                  <Label htmlFor="show_speakers">Show Speaker Labels</Label>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Whisper Configuration */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2">
                <Brain className="w-5 h-5" />
                <span>Whisper Configuration</span>
              </CardTitle>
              <CardDescription>Whisper model and backend settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label htmlFor="model">Model</Label>
                  <Select 
                    value={processingParams.model || 'large-v3-turbo'} 
                    onValueChange={(value) => updateProcessingParam('model', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="large-v3-turbo">large-v3-turbo</SelectItem>
                      <SelectItem value="medium">medium</SelectItem>
                      <SelectItem value="small">small</SelectItem>
                      <SelectItem value="base">base</SelectItem>
                      <SelectItem value="tiny">tiny</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="backend">Backend</Label>
                  <Select 
                    value={processingParams.backend || 'faster-whisper'} 
                    onValueChange={(value) => updateProcessingParam('backend', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="faster-whisper">Faster Whisper</SelectItem>
                      <SelectItem value="openai-api">OpenAI API</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="language">Language</Label>
                  <Select 
                    value={processingParams.language || 'auto'} 
                    onValueChange={(value) => updateProcessingParam('language', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto Detect</SelectItem>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                      <SelectItem value="fr">French</SelectItem>
                      <SelectItem value="de">German</SelectItem>
                      <SelectItem value="zh">Chinese</SelectItem>
                      <SelectItem value="ja">Japanese</SelectItem>
                      <SelectItem value="ko">Korean</SelectItem>
                      <SelectItem value="ru">Russian</SelectItem>
                      <SelectItem value="pt">Portuguese</SelectItem>
                      <SelectItem value="it">Italian</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="task">Task</Label>
                  <Select 
                    value={processingParams.task || 'transcribe'} 
                    onValueChange={(value) => updateProcessingParam('task', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="transcribe">Transcribe</SelectItem>
                      <SelectItem value="translate">Translate</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="model_cache_dir">Model Cache Directory (Optional)</Label>
                  <Input
                    id="model_cache_dir"
                    type="text"
                    value={processingParams.model_cache_dir || ''}
                    onChange={(e) => updateProcessingParam('model_cache_dir', e.target.value || undefined)}
                    placeholder="/path/to/model/cache"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="model_dir">Model Directory (Optional)</Label>
                  <Input
                    id="model_dir"
                    type="text"
                    value={processingParams.model_dir || ''}
                    onChange={(e) => updateProcessingParam('model_dir', e.target.value || undefined)}
                    placeholder="/path/to/model/directory"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* LLM Configuration */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2">
                <Brain className="w-5 h-5" />
                <span>LLM Configuration</span>
              </CardTitle>
              <CardDescription>Language model settings for inference</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label htmlFor="fast_llm">Fast LLM Model</Label>
                  <Input
                    id="fast_llm"
                    type="text"
                    value={processingParams.fast_llm || 'openai/gpt-4.1-nano'}
                    onChange={(e) => updateProcessingParam('fast_llm', e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Model for quick text parsing operations
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="base_llm">Base LLM Model</Label>
                  <Input
                    id="base_llm"
                    type="text"
                    value={processingParams.base_llm || 'openai/gpt-4.1-mini'}
                    onChange={(e) => updateProcessingParam('base_llm', e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Model for complex operations and analysis
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="llm_analysis_interval">Analysis Trigger Interval (seconds)</Label>
                  <Input
                    id="llm_analysis_interval"
                    type="number"
                    step="0.1"
                    min="1.0"
                    value={processingParams.llm_analysis_interval ?? 15.0}
                    onChange={(e) => updateProcessingParam('llm_analysis_interval', parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">
                    Minimum time between comprehensive analyses
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="llm_new_text_trigger">New Text Trigger (characters)</Label>
                  <Input
                    id="llm_new_text_trigger"
                    type="number"
                    min="50"
                    value={processingParams.llm_new_text_trigger ?? 50}
                    onChange={(e) => updateProcessingParam('llm_new_text_trigger', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">
                    Characters of new text to trigger analysis (~50 words = 300 chars)
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="parser_enabled"
                    checked={processingParams.parser_enabled ?? false}
                    onCheckedChange={(checked) => updateProcessingParam('parser_enabled', checked)}
                  />
                  <Label htmlFor="parser_enabled">Enable Transcript Parser</Label>
                  <p className="text-xs text-muted-foreground ml-2">
                    Automatic text correction and formatting
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="parser_trigger_interval">Parser Trigger Interval (seconds)</Label>
                  <Input
                    id="parser_trigger_interval"
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={processingParams.parser_trigger_interval ?? 1.0}
                    onChange={(e) => updateProcessingParam('parser_trigger_interval', parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">
                    Time interval between transcript parser triggers
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="parser_output_tokens">Parser Output Tokens Limit</Label>
                  <Input
                    id="parser_output_tokens"
                    type="number"
                    min="1000"
                    max="100000"
                    step="1000"
                    value={processingParams.parser_output_tokens ?? 33000}
                    onChange={(e) => updateProcessingParam('parser_output_tokens', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">
                    <strong>Maximum OUTPUT tokens</strong> the parser model can generate. Parser will automatically chunk text to stay within this limit. Higher values = fewer API calls but require more capable models.
                  </p>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="parser_window">Parser Character Window</Label>
                  <Input
                    id="parser_window"
                    type="number"
                    min="10"
                    max="1000"
                    step="10"
                    value={processingParams.parser_window ?? 100}
                    onChange={(e) => updateProcessingParam('parser_window', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">
                    Character window size for sentence processing. Parser processes sentences that fall within the last N characters of the transcript. Larger values = more sentences processed.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Processing Configuration */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span>Processing Configuration</span>
              </CardTitle>
              <CardDescription>Audio processing parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label htmlFor="min_chunk_size">Min Chunk Size (seconds)</Label>
                  <Input
                    id="min_chunk_size"
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={processingParams.min_chunk_size ?? 0.5}
                    onChange={(e) => updateProcessingParam('min_chunk_size', parseFloat(e.target.value))}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="buffer_trimming_sec">Buffer Trimming (seconds)</Label>
                  <Input
                    id="buffer_trimming_sec"
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={processingParams.buffer_trimming_sec ?? 15.0}
                    onChange={(e) => updateProcessingParam('buffer_trimming_sec', parseFloat(e.target.value))}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="buffer_trimming">Buffer Trimming Strategy</Label>
                  <Select 
                    value={processingParams.buffer_trimming || 'segment'} 
                    onValueChange={(value) => updateProcessingParam('buffer_trimming', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sentence">Sentence</SelectItem>
                      <SelectItem value="segment">Segment</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="vac_chunk_size">VAC Chunk Size (seconds)</Label>
                  <Input
                    id="vac_chunk_size"
                    type="number"
                    step="0.01"
                    min="0.01"
                    value={processingParams.vac_chunk_size ?? 0.04}
                    onChange={(e) => updateProcessingParam('vac_chunk_size', parseFloat(e.target.value))}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
