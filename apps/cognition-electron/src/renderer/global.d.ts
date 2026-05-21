export {};

declare global {
  type CognitionSettings = {
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

  type RecentRun = {
    id: string;
    createdAt: string;
    inputName: string;
    settings: CognitionSettings;
    outputText: string;
    metrics: string;
    overlayVideoUrl: string;
  };

  interface Window {
    cognition: {
      getApiBaseUrl: () => Promise<string>;
      quit: () => Promise<void>;
      listRecentRuns: () => Promise<RecentRun[]>;
      addRecentRun: (run: RecentRun) => Promise<RecentRun[]>;
      clearRecentRuns: () => Promise<RecentRun[]>;
    };
  }
}
