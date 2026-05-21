import {
  Clock3,
  Eraser,
  Film,
  Loader2,
  Play,
  Settings2,
  Upload
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type ModelOption = {
  label: string;
  modelId: string;
};

type ModelsResponse = {
  models: ModelOption[];
  defaults: CognitionSettings;
};

type InferenceResponse = {
  runId: string;
  outputText: string;
  metrics: string;
  overlayVideoUrl: string;
};

type CreateJobResponse = {
  jobId: string;
};

type JobStatusResponse = {
  jobId: string;
  status: "queued" | "running" | "complete" | "error";
  progress: number;
  stage: string;
  result?: InferenceResponse;
  error?: string;
};

const fallbackSettings: CognitionSettings = {
  modelLabel: "RynnBrain-2B",
  frameFps: 2,
  promptFrameCount: 5,
  windowStride: 1,
  prompt: "How many people are in the video?",
  maxNewTokens: 128,
  temperature: 0,
  topP: 0.95,
  topK: 50
};

function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function absoluteApiUrl(apiBaseUrl: string, pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  return `${apiBaseUrl}${pathOrUrl}`;
}

export function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [settings, setSettings] = useState<CognitionSettings>(fallbackSettings);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState("");
  const [outputText, setOutputText] = useState("");
  const [metrics, setMetrics] = useState("");
  const [overlayVideoUrl, setOverlayVideoUrl] = useState("");
  const [recentRuns, setRecentRuns] = useState<RecentRun[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStage, setProgressStage] = useState("");
  const [error, setError] = useState("");

  const selectedVideoLabel = useMemo(() => videoFile?.name ?? "No video selected", [videoFile]);

  useEffect(() => {
    let isMounted = true;
    async function boot() {
      try {
        const baseUrl = await window.cognition.getApiBaseUrl();
        const [modelResponse, runs] = await Promise.all([
          fetch(`${baseUrl}/api/models`),
          window.cognition.listRecentRuns()
        ]);
        if (!modelResponse.ok) {
          throw new Error(`Failed to load models: HTTP ${modelResponse.status}`);
        }
        const modelData = (await modelResponse.json()) as ModelsResponse;
        if (isMounted) {
          setApiBaseUrl(baseUrl);
          setModels(modelData.models);
          setSettings(modelData.defaults);
          setRecentRuns(runs);
        }
      } catch (bootError) {
        if (isMounted) {
          setError(asErrorMessage(bootError));
        }
      }
    }
    void boot();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!videoFile) {
      setVideoPreviewUrl("");
      return;
    }
    const objectUrl = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [videoFile]);

  function updateSetting<K extends keyof CognitionSettings>(key: K, value: CognitionSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function onVideoChange(event: ChangeEvent<HTMLInputElement>) {
    setError("");
    setVideoFile(event.target.files?.[0] ?? null);
  }

  async function parseError(response: Response): Promise<string> {
    try {
      const body = (await response.json()) as { detail?: string };
      return body.detail ?? `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiBaseUrl) {
      setError("Backend is not ready.");
      return;
    }
    if (!videoFile) {
      setError("Select a .mp4 or .webm video.");
      return;
    }

    setIsRunning(true);
    setProgress(5);
    setProgressStage("Uploading");
    setError("");
    try {
      const formData = new FormData();
      formData.append("video", videoFile);
      formData.append("modelLabel", settings.modelLabel);
      formData.append("frameFps", String(settings.frameFps));
      formData.append("promptFrameCount", String(settings.promptFrameCount));
      formData.append("windowStride", String(settings.windowStride));
      formData.append("prompt", settings.prompt);
      formData.append("maxNewTokens", String(settings.maxNewTokens));
      formData.append("temperature", String(settings.temperature));
      formData.append("topP", String(settings.topP));
      formData.append("topK", String(settings.topK));

      const response = await fetch(`${apiBaseUrl}/api/inference-jobs`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const { jobId } = (await response.json()) as CreateJobResponse;
      const result = await waitForJob(jobId);
      const resolvedOverlayUrl = `${absoluteApiUrl(apiBaseUrl, result.overlayVideoUrl)}?t=${Date.now()}`;
      setOutputText(result.outputText);
      setMetrics(result.metrics);
      setOverlayVideoUrl(resolvedOverlayUrl);

      const nextRuns = await window.cognition.addRecentRun({
        id: result.runId,
        createdAt: new Date().toISOString(),
        inputName: videoFile.name,
        settings,
        outputText: result.outputText,
        metrics: result.metrics,
        overlayVideoUrl: resolvedOverlayUrl
      });
      setRecentRuns(nextRuns);
    } catch (runError) {
      setError(asErrorMessage(runError));
    } finally {
      setIsRunning(false);
      setProgressStage("");
    }
  }

  async function waitForJob(jobId: string): Promise<InferenceResponse> {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 700));
      const response = await fetch(`${apiBaseUrl}/api/inference-jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      const job = (await response.json()) as JobStatusResponse;
      setProgress(Math.max(0, Math.min(100, Math.round(job.progress))));
      setProgressStage(job.stage);

      if (job.status === "complete" && job.result) {
        return job.result;
      }
      if (job.status === "error") {
        throw new Error(job.error ?? "Inference failed.");
      }
    }
  }

  function loadRecentRun(run: RecentRun) {
    setSettings({
      ...fallbackSettings,
      ...run.settings,
      windowStride: run.settings.windowStride ?? run.settings.promptFrameCount ?? fallbackSettings.windowStride
    });
    setOutputText(run.outputText);
    setMetrics(run.metrics);
    setOverlayVideoUrl(run.overlayVideoUrl);
    setError("");
  }

  async function clearRecentRuns() {
    setRecentRuns(await window.cognition.clearRecentRuns());
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="title-row">
            <h1>RynnBrain Cognition</h1>
            <button type="button" className="quit-button" onClick={() => void window.cognition.quit()} aria-label="Quit application">
              Exit
            </button>
          </div>
          <p className="topbar-backend">{apiBaseUrl ? `Backend ${apiBaseUrl}` : "Starting backend"}</p>
        </div>
        <div className="status-pill" data-state={isRunning ? "busy" : "ready"}>
          {isRunning ? <Loader2 className="spin" size={16} /> : <Clock3 size={16} />}
          <span>{isRunning ? "Running" : "Ready"}</span>
        </div>
      </header>

      <div className="workspace">
        <form className="control-panel" onSubmit={onSubmit}>
          <section className="panel-section">
            <label className="file-picker">
              <input type="file" accept=".mp4,.webm,video/mp4,video/webm" onChange={onVideoChange} />
              <Upload size={18} />
              <span>{selectedVideoLabel}</span>
            </label>
            {videoPreviewUrl ? (
              <video className="input-preview" src={videoPreviewUrl} controls muted />
            ) : (
              <div className="empty-preview">
                <Film size={28} />
              </div>
            )}
          </section>

          <section className="panel-section form-grid">
            <label>
              <span>Model</span>
              <select
                value={settings.modelLabel}
                onChange={(event) => updateSetting("modelLabel", event.target.value)}
              >
                {(models.length ? models : [{ label: fallbackSettings.modelLabel, modelId: "" }]).map(
                  (model) => (
                    <option key={model.label} value={model.label}>
                      {model.label}
                    </option>
                  )
                )}
              </select>
            </label>

            <label>
              <span>Frame extraction FPS</span>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={settings.frameFps}
                onChange={(event) => updateSetting("frameFps", Number(event.target.value))}
              />
            </label>

            <label>
              <span>Images passed to prompt</span>
              <input
                type="number"
                min="1"
                max="128"
                step="1"
                value={settings.promptFrameCount}
                onChange={(event) => updateSetting("promptFrameCount", Number(event.target.value))}
              />
            </label>

            <label>
              <span>Window stride</span>
              <input
                type="number"
                min="1"
                max="128"
                step="1"
                value={settings.windowStride}
                onChange={(event) => updateSetting("windowStride", Number(event.target.value))}
              />
            </label>
          </section>

          <section className="panel-section">
            <label>
              <span>Instruction prompt</span>
              <textarea
                rows={6}
                value={settings.prompt}
                onChange={(event) => updateSetting("prompt", event.target.value)}
              />
            </label>
          </section>

          <section className="panel-section">
            <div className="section-title">
              <Settings2 size={17} />
              <span>Generation settings</span>
            </div>
            <div className="form-grid compact">
              <label>
                <span>Max new tokens</span>
                <input
                  type="number"
                  min="16"
                  max="1024"
                  step="1"
                  value={settings.maxNewTokens}
                  onChange={(event) => updateSetting("maxNewTokens", Number(event.target.value))}
                />
              </label>
              <label>
                <span>Temperature</span>
                <input
                  type="number"
                  min="0"
                  max="1.5"
                  step="0.01"
                  value={settings.temperature}
                  onChange={(event) => updateSetting("temperature", Number(event.target.value))}
                />
              </label>
              <label>
                <span>Top-p</span>
                <input
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={settings.topP}
                  onChange={(event) => updateSetting("topP", Number(event.target.value))}
                />
              </label>
              <label>
                <span>Top-k</span>
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="1"
                  value={settings.topK}
                  onChange={(event) => updateSetting("topK", Number(event.target.value))}
                />
              </label>
            </div>
          </section>

          {error ? <div className="error-banner">{error}</div> : null}

          {isRunning ? (
            <div className="progress-block" aria-live="polite">
              <div className="progress-label">
                <span>{progressStage || "Running"}</span>
                <span>{progress}%</span>
              </div>
              <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          ) : null}

          <button className="run-button" type="submit" disabled={isRunning}>
            {isRunning ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            <span>{isRunning ? "Running inference" : "Run inference"}</span>
          </button>
        </form>

        <section className="result-panel">
          <div className="result-video">
            {overlayVideoUrl ? (
              <video key={overlayVideoUrl} src={overlayVideoUrl} controls />
            ) : (
              <div className="empty-output">
                <Film size={34} />
              </div>
            )}
          </div>
        </section>

        <section className="result-grid">
          <label>
            <span>Model output</span>
            <textarea readOnly rows={10} value={outputText} />
          </label>
          <label>
            <span>Metrics</span>
            <textarea readOnly rows={10} value={metrics} />
          </label>
        </section>

        <aside className="history-panel">
          <div className="history-header">
            <h2>Recent runs</h2>
            <button type="button" className="icon-button" onClick={clearRecentRuns} aria-label="Clear recent runs">
              <Eraser size={16} />
            </button>
          </div>
          <div className="history-list">
            {recentRuns.length === 0 ? (
              <div className="empty-history">No recent runs</div>
            ) : (
              recentRuns.map((run) => (
                <button key={run.id} type="button" className="history-item" onClick={() => loadRecentRun(run)}>
                  <span>{run.inputName}</span>
                  <time>{new Date(run.createdAt).toLocaleString()}</time>
                  <small>{run.settings.modelLabel}</small>
                </button>
              ))
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
