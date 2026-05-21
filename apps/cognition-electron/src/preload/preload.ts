import { contextBridge, ipcRenderer } from "electron";

type RecentRun = {
  id: string;
  createdAt: string;
  inputName: string;
  settings: {
    modelLabel: string;
    frameFps: number;
    promptFrameCount: number;
    windowStride: number;
    prompt: string;
    maxNewTokens: number;
    temperature: number;
    topP: number;
    topK: number;
  };
  outputText: string;
  metrics: string;
  overlayVideoUrl: string;
};

contextBridge.exposeInMainWorld("cognition", {
  getApiBaseUrl: (): Promise<string> => ipcRenderer.invoke("cognition:apiBaseUrl"),
  quit: (): Promise<void> => ipcRenderer.invoke("cognition:quit"),
  listRecentRuns: (): Promise<RecentRun[]> => ipcRenderer.invoke("cognition:recentRuns:list"),
  addRecentRun: (run: RecentRun): Promise<RecentRun[]> =>
    ipcRenderer.invoke("cognition:recentRuns:add", run),
  clearRecentRuns: (): Promise<RecentRun[]> => ipcRenderer.invoke("cognition:recentRuns:clear")
});
