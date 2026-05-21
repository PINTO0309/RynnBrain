import { app, BrowserWindow, ipcMain } from "electron";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";

const backendHost = process.env.RYNNBRAIN_COGNITION_HOST ?? "127.0.0.1";
const backendPort = Number(process.env.RYNNBRAIN_COGNITION_PORT ?? "8765");
const apiBaseUrl = `http://${backendHost}:${backendPort}`;

let backendProcess: ChildProcessWithoutNullStreams | null = null;
let mainWindow: BrowserWindow | null = null;

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

function repoRoot(): string {
  return path.resolve(app.getAppPath(), "../..");
}

function recentRunsPath(): string {
  return path.join(app.getPath("userData"), "recent-runs.json");
}

async function readRecentRuns(): Promise<RecentRun[]> {
  try {
    const raw = await fs.readFile(recentRunsPath(), "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function writeRecentRuns(runs: RecentRun[]): Promise<void> {
  await fs.mkdir(path.dirname(recentRunsPath()), { recursive: true });
  await fs.writeFile(recentRunsPath(), JSON.stringify(runs.slice(0, 20), null, 2), "utf8");
}

function startBackend(): void {
  if (backendProcess) {
    return;
  }

  const scriptPath = path.join(repoRoot(), "cookbooks", "cognition_api.py");
  backendProcess = spawn(
    "uv",
    ["run", "python", scriptPath, "--host", backendHost, "--port", String(backendPort)],
    {
      cwd: repoRoot(),
      env: {
        ...process.env,
        FORCE_QWENVL_VIDEO_READER: process.env.FORCE_QWENVL_VIDEO_READER ?? "torchcodec",
        TORCHCODEC_NUM_THREADS: process.env.TORCHCODEC_NUM_THREADS ?? "1"
      }
    }
  );

  backendProcess.stdout.on("data", (chunk) => {
    console.log(`[cognition-api] ${chunk.toString().trimEnd()}`);
  });
  backendProcess.stderr.on("data", (chunk) => {
    console.error(`[cognition-api] ${chunk.toString().trimEnd()}`);
  });
  backendProcess.on("exit", (code, signal) => {
    console.log(`[cognition-api] exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    backendProcess = null;
  });
}

async function waitForBackend(): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${apiBaseUrl}/api/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // Backend is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Cognition API did not become ready at ${apiBaseUrl}.`);
}

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 1120,
    minWidth: 1040,
    minHeight: 980,
    backgroundColor: "#f6f7f9",
    title: "RynnBrain Cognition",
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  const devServerUrl = process.env.ELECTRON_RENDERER_URL ?? "http://127.0.0.1:5173";
  if (!app.isPackaged) {
    await mainWindow.loadURL(devServerUrl);
  } else {
    await mainWindow.loadFile(path.join(app.getAppPath(), "dist", "index.html"));
  }
}

ipcMain.handle("cognition:apiBaseUrl", () => apiBaseUrl);
ipcMain.handle("cognition:quit", () => {
  app.quit();
});
ipcMain.handle("cognition:recentRuns:list", () => readRecentRuns());
ipcMain.handle("cognition:recentRuns:add", async (_event, run: RecentRun) => {
  const existing = await readRecentRuns();
  const next = [run, ...existing.filter((item) => item.id !== run.id)];
  await writeRecentRuns(next);
  return next.slice(0, 20);
});
ipcMain.handle("cognition:recentRuns:clear", async () => {
  await writeRecentRuns([]);
  return [];
});

app.whenReady().then(async () => {
  startBackend();
  await waitForBackend();
  await createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
});
