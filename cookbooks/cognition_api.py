import argparse
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cognition_core import (
    DEFAULT_FRAME_FPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_LABEL,
    DEFAULT_PROMPT,
    DEFAULT_PROMPT_FRAME_COUNT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_WINDOW_STRIDE,
    MODEL_CHOICES,
    SUPPORTED_VIDEO_EXTS,
    CognitionError,
    run_inference,
    run_inference_sliding,
)


RUNS_DIR = Path(tempfile.gettempdir()) / "rynnbrain_cognition_runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
OVERLAYS_DIR = RUNS_DIR / "overlays"

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    os.environ.setdefault("TORCHCODEC_NUM_THREADS", "1")
    _ensure_dirs()
    yield


app = FastAPI(title="RynnBrain Cognition API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTS:
        raise CognitionError("Only .mp4 and .webm videos are supported.")
    return suffix


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        current = JOBS.setdefault(job_id, {})
        current.update(updates)


def _run_inference_job(
    *,
    job_id: str,
    upload_path: Path,
    overlay_dir: Path,
    model_label: str,
    frame_fps: float,
    prompt_frame_count: int,
    window_stride: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    try:
        _set_job(job_id, status="running", progress=20, stage="Preparing frames")
        def on_window(window_idx: int, window_count: int) -> None:
            progress = 25 + round((window_idx / max(1, window_count)) * 60)
            _set_job(
                job_id,
                status="running",
                progress=progress,
                stage=f"Running window {window_idx + 1}/{window_count}",
            )

        result = run_inference_sliding(
            video_path=str(upload_path),
            model_label=model_label,
            frame_fps=frame_fps,
            prompt_frame_count=prompt_frame_count,
            window_stride=window_stride,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            output_dir=str(overlay_dir),
            progress_callback=on_window,
        )

        _set_job(job_id, progress=90, stage="Saving overlay")
        overlay_path = overlay_dir / "overlay.mp4"
        shutil.move(result.overlay_path, overlay_path)
        _set_job(
            job_id,
            status="complete",
            progress=100,
            stage="Complete",
            result={
                "runId": job_id,
                "outputText": result.output_text,
                "metrics": result.metrics,
                "overlayVideoUrl": f"/api/runs/{job_id}/overlay.mp4",
            },
        )
    except CognitionError as exc:
        _set_job(job_id, status="error", progress=100, stage="Error", error=str(exc))
    except Exception as exc:
        _set_job(job_id, status="error", progress=100, stage="Error", error=f"Inference failed: {exc}")
    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, object]:
    return {
        "models": [
            {"label": label, "modelId": model_id}
            for label, model_id in MODEL_CHOICES.items()
        ],
        "defaults": {
            "modelLabel": DEFAULT_MODEL_LABEL,
            "frameFps": DEFAULT_FRAME_FPS,
            "promptFrameCount": DEFAULT_PROMPT_FRAME_COUNT,
            "windowStride": DEFAULT_WINDOW_STRIDE,
            "prompt": DEFAULT_PROMPT,
            "maxNewTokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "topP": DEFAULT_TOP_P,
            "topK": DEFAULT_TOP_K,
        },
    }


@app.post("/api/inference")
async def inference(
    video: UploadFile = File(...),
    modelLabel: str = Form(DEFAULT_MODEL_LABEL),
    frameFps: float = Form(DEFAULT_FRAME_FPS),
    promptFrameCount: int = Form(DEFAULT_PROMPT_FRAME_COUNT),
    windowStride: int = Form(DEFAULT_WINDOW_STRIDE),
    prompt: str = Form(DEFAULT_PROMPT),
    maxNewTokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    topP: float = Form(DEFAULT_TOP_P),
    topK: int = Form(DEFAULT_TOP_K),
) -> dict[str, str]:
    _ensure_dirs()
    run_id = str(uuid.uuid4())

    try:
        upload_path = UPLOADS_DIR / f"{run_id}{_safe_suffix(video.filename)}"
        overlay_dir = OVERLAYS_DIR / run_id
        overlay_dir.mkdir(parents=True, exist_ok=True)

        with upload_path.open("wb") as output:
            shutil.copyfileobj(video.file, output)

        result = run_inference(
            video_path=str(upload_path),
            model_label=modelLabel,
            frame_fps=frameFps,
            prompt_frame_count=promptFrameCount,
            window_stride=windowStride,
            prompt=prompt,
            max_new_tokens=maxNewTokens,
            temperature=temperature,
            top_p=topP,
            top_k=topK,
            output_dir=str(overlay_dir),
        )
    except CognitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        video.file.close()
        if "upload_path" in locals() and upload_path.exists():
            upload_path.unlink()

    overlay_path = overlay_dir / "overlay.mp4"
    shutil.move(result.overlay_path, overlay_path)
    return {
        "runId": run_id,
        "outputText": result.output_text,
        "metrics": result.metrics,
        "overlayVideoUrl": f"/api/runs/{run_id}/overlay.mp4",
    }


@app.post("/api/inference-jobs")
async def create_inference_job(
    video: UploadFile = File(...),
    modelLabel: str = Form(DEFAULT_MODEL_LABEL),
    frameFps: float = Form(DEFAULT_FRAME_FPS),
    promptFrameCount: int = Form(DEFAULT_PROMPT_FRAME_COUNT),
    windowStride: int = Form(DEFAULT_WINDOW_STRIDE),
    prompt: str = Form(DEFAULT_PROMPT),
    maxNewTokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    topP: float = Form(DEFAULT_TOP_P),
    topK: int = Form(DEFAULT_TOP_K),
) -> dict[str, str]:
    _ensure_dirs()
    job_id = str(uuid.uuid4())

    try:
        upload_path = UPLOADS_DIR / f"{job_id}{_safe_suffix(video.filename)}"
        overlay_dir = OVERLAYS_DIR / job_id
        overlay_dir.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as output:
            shutil.copyfileobj(video.file, output)
    except CognitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create inference job: {exc}") from exc
    finally:
        video.file.close()

    _set_job(job_id, status="queued", progress=5, stage="Queued")
    thread = threading.Thread(
        target=_run_inference_job,
        kwargs={
            "job_id": job_id,
            "upload_path": upload_path,
            "overlay_dir": overlay_dir,
            "model_label": modelLabel,
            "frame_fps": frameFps,
            "prompt_frame_count": promptFrameCount,
            "window_stride": windowStride,
            "prompt": prompt,
            "max_new_tokens": maxNewTokens,
            "temperature": temperature,
            "top_p": topP,
            "top_k": topK,
        },
        daemon=True,
    )
    thread.start()
    return {"jobId": job_id}


@app.get("/api/inference-jobs/{job_id}")
def get_inference_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Inference job not found.")
        return {"jobId": job_id, **job}


@app.get("/api/runs/{run_id}/overlay.mp4")
def overlay(run_id: str) -> FileResponse:
    overlay_path = OVERLAYS_DIR / run_id / "overlay.mp4"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay video not found.")
    return FileResponse(
        overlay_path,
        media_type="video/mp4",
        filename=f"rynnbrain-cognition-{run_id}.mp4",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RynnBrain Cognition API")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
