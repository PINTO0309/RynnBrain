import gc
import os
import re
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_CHOICES = {
    "RynnBrain-2B": "Alibaba-DAMO-Academy/RynnBrain-2B",
    "RynnBrain-4B": "Alibaba-DAMO-Academy/RynnBrain-4B",
    "RynnBrain-8B": "Alibaba-DAMO-Academy/RynnBrain-8B",
    "RynnBrain-30B-A3B": "Alibaba-DAMO-Academy/RynnBrain-30B-A3B",
}
DEFAULT_MODEL_LABEL = "RynnBrain-2B"
DEFAULT_PROMPT = "How many people are in the video?"
DEFAULT_FRAME_FPS = 2.0
DEFAULT_PROMPT_FRAME_COUNT = 5
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
SUPPORTED_VIDEO_EXTS = {".mp4", ".webm"}
MAX_DECODED_FRAMES = 2048


class CognitionError(Exception):
    """User-facing validation or inference error."""


@dataclass
class LoadedModel:
    label: str
    model_id: str
    model: Any
    processor: Any


@dataclass
class InferenceResult:
    output_text: str
    metrics: str
    overlay_path: str


_loaded_model: LoadedModel | None = None
_model_lock = threading.Lock()


def resolve_model_id(model_label: str) -> str:
    return MODEL_CHOICES.get(model_label, model_label)


def unload_model() -> None:
    global _loaded_model
    _loaded_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(model_label: str) -> LoadedModel:
    global _loaded_model
    model_id = resolve_model_id(model_label)

    with _model_lock:
        if _loaded_model and _loaded_model.model_id == model_id:
            return _loaded_model

        unload_model()
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        model.eval()
        _loaded_model = LoadedModel(
            label=model_label,
            model_id=model_id,
            model=model,
            processor=processor,
        )
        return _loaded_model


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def validate_video_path(video_path: str | None) -> str:
    if not video_path:
        raise CognitionError("Please upload a .mp4 or .webm video.")

    suffix = Path(video_path).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTS:
        raise CognitionError("Only .mp4 and .webm videos are supported.")

    if not os.path.exists(video_path):
        raise CognitionError(f"Video file does not exist: {video_path}")

    return video_path


def _select_evenly(items: list[Any], max_count: int) -> list[Any]:
    if max_count <= 0:
        raise CognitionError("The number of prompt images must be greater than zero.")
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count).round().astype(int)
    return [items[int(i)] for i in indices]


def _probe_video(video_path: str) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(video_path)
    try:
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
            if width > 0 and height > 0:
                return width, height, duration
    finally:
        cap.release()

    if not hasattr(ffmpeg, "probe"):
        raise CognitionError("Failed to inspect video with OpenCV, and ffmpeg-python is unavailable.")

    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            stream for stream in probe["streams"] if stream.get("codec_type") == "video"
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(
            video_stream.get("duration") or probe.get("format", {}).get("duration") or 0
        )
        return width, height, duration
    except Exception as exc:
        raise CognitionError(f"Failed to inspect video: {exc}") from exc


def _read_video_with_opencv(video_path: str, sample_fps: float, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    step = max(1, round(src_fps / sample_fps)) if sample_fps > 0 else 1
    frames: list[np.ndarray] = []
    frame_idx = 0

    try:
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    return frames


def extract_frames(
    video_path: str,
    sample_fps: float,
    prompt_frame_count: int,
    select_prompt_frames: bool = True,
) -> tuple[list[np.ndarray], int, int]:
    if sample_fps <= 0:
        raise CognitionError("Frame extraction FPS must be greater than zero.")

    width, height, duration = _probe_video(video_path)
    expected_frames = int(np.ceil(duration * sample_fps)) if duration > 0 else prompt_frame_count
    decode_limit = max(prompt_frame_count, min(MAX_DECODED_FRAMES, expected_frames or MAX_DECODED_FRAMES))

    decoded_frames = []
    if hasattr(ffmpeg, "input"):
        try:
            out, _ = (
                ffmpeg.input(video_path, threads=1)
                .filter("fps", fps=sample_fps)
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    vframes=decode_limit,
                )
                .global_args("-hide_banner", "-loglevel", "error", "-nostdin")
                .run(capture_stdout=True, capture_stderr=True)
            )
            frame_size = width * height * 3
            if len(out) < frame_size or len(out) % frame_size != 0:
                raise RuntimeError(f"Unexpected decoded byte size: {len(out)}")
            frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            decoded_frames = [frame.copy() for frame in frames]
        except Exception:
            decoded_frames = []

    if not decoded_frames:
        decoded_frames = _read_video_with_opencv(video_path, sample_fps, decode_limit)

    if not decoded_frames:
        raise CognitionError("No frames could be decoded from the video.")

    selected_frames = (
        _select_evenly(decoded_frames, int(prompt_frame_count))
        if select_prompt_frames
        else decoded_frames
    )
    return selected_frames, len(decoded_frames), expected_frames


def _sliding_windows(items: list[Any], window_size: int, stride: int) -> list[list[Any]]:
    if window_size <= 0:
        raise CognitionError("The sliding window size must be greater than zero.")
    if stride <= 0:
        raise CognitionError("The sliding window stride must be greater than zero.")
    if not items:
        return []

    windows = []
    for start in range(0, len(items), stride):
        window = items[start:start + window_size]
        if window:
            windows.append(window)
        if start + window_size >= len(items):
            break
    return windows


def _pil_frames(frames: Iterable[np.ndarray]) -> list[Image.Image]:
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _build_messages(frames: list[Image.Image], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_idx, frame in enumerate(frames):
        content.append({"type": "text", "text": f"<frame {frame_idx}>: "})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _vram_markdown() -> str:
    if not torch.cuda.is_available():
        return "VRAM: N/A (CUDA is not available)"

    lines = ["VRAM:"]
    for device_idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
        reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
        total = props.total_memory / 1024**3
        lines.append(
            f"- cuda:{device_idx} {props.name}: "
            f"allocated {allocated:.2f} GB / reserved {reserved:.2f} GB / total {total:.2f} GB"
        )
    return "\n".join(lines)


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        str(Path(__file__).parent / "assets" / "arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, width: int, max_lines: int = 8) -> str:
    if not text:
        return ""
    max_chars = max(24, width // 12)
    chunks = []
    for line in text.splitlines():
        chunks.extend(textwrap.wrap(line, max_chars) or [""])
    return "\n".join(chunks[:max_lines])


def _parse_spatial_tags(output_text: str) -> list[dict[str, Any]]:
    tag_pattern = re.compile(
        r"<(?P<tag>object|area|affordance|trajectory)>(?P<body>.*?)</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    )
    frame_pattern = re.compile(r"<frame\s*(?P<frame>\d+)>", re.IGNORECASE)
    coord_pattern = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")

    annotations = []
    for match in tag_pattern.finditer(output_text):
        body = match.group("body")
        frame_match = frame_pattern.search(body)
        frame_idx = int(frame_match.group("frame")) if frame_match else None
        points = [
            (float(x), float(y))
            for x, y in coord_pattern.findall(body)
        ]
        if points:
            annotations.append(
                {
                    "tag": match.group("tag").lower(),
                    "frame_idx": frame_idx,
                    "points": points,
                }
            )
    return annotations


def _to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    px = int(round(np.clip(x, 0, 1000) / 1000 * (width - 1)))
    py = int(round(np.clip(y, 0, 1000) / 1000 * (height - 1)))
    return px, py


def _draw_annotation(
    draw: ImageDraw.ImageDraw,
    annotation: dict[str, Any],
    width: int,
    height: int,
) -> None:
    tag = annotation["tag"]
    points = [_to_pixel(point, width, height) for point in annotation["points"]]
    radius = max(4, max(width, height) // 120)
    line_width = max(2, max(width, height) // 200)

    if tag == "object" and len(points) >= 2:
        x_values = [points[0][0], points[1][0]]
        y_values = [points[0][1], points[1][1]]
        draw.rectangle(
            [min(x_values), min(y_values), max(x_values), max(y_values)],
            outline=(255, 48, 48),
            width=line_width,
        )
        return

    if tag in {"area", "trajectory"} and len(points) >= 2:
        if tag == "area" and len(points) >= 3:
            draw.polygon(points, outline=(255, 214, 10))
        else:
            draw.line(points, fill=(58, 134, 255), width=line_width)

    color = (255, 214, 10) if tag == "area" else (58, 134, 255)
    if tag == "affordance":
        color = (255, 48, 48)
    for x, y in points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def _draw_text_panel(
    image: Image.Image,
    prompt: str,
    output_text: str,
) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    font = _find_font(max(13, min(width, height) // 38))
    small_font = _find_font(max(12, min(width, height) // 44))

    output_wrapped = _wrap_text(f"Output: {output_text}", width, max_lines=7)
    text_bbox = draw.multiline_textbbox((0, 0), output_wrapped, font=font, spacing=4)
    panel_height = min(height // 2, text_bbox[3] - text_bbox[1] + 28)
    draw.rectangle([0, height - panel_height, width, height], fill=(0, 0, 0, 176))
    draw.multiline_text(
        (14, height - panel_height + 12),
        output_wrapped,
        fill=(255, 255, 255, 255),
        font=font,
        spacing=4,
    )

    label = "RynnBrain Cognition"
    label_bbox = draw.textbbox((0, 0), label, font=small_font)
    draw.rectangle(
        [10, 10, label_bbox[2] + 24, label_bbox[3] + 20],
        fill=(0, 0, 0, 150),
    )
    draw.text((17, 14), label, fill=(255, 255, 255, 255), font=small_font)

    instruction_text = _wrap_text(f"Instruction: {prompt}", width, max_lines=3)
    if instruction_text:
        instruction_top = label_bbox[3] + 28
        instruction_bbox = draw.multiline_textbbox(
            (0, 0),
            instruction_text,
            font=small_font,
            spacing=3,
        )
        instruction_height = instruction_bbox[3] - instruction_bbox[1] + 18
        instruction_bottom = min(height - panel_height - 8, instruction_top + instruction_height)
        if instruction_bottom > instruction_top + 8:
            draw.rectangle(
                [10, instruction_top, width - 10, instruction_bottom],
                fill=(0, 0, 0, 150),
            )
            draw.multiline_text(
                (17, instruction_top + 9),
                instruction_text,
                fill=(255, 255, 255, 255),
                font=small_font,
                spacing=3,
            )
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_overlay_video(
    frames: list[np.ndarray],
    prompt: str,
    output_text: str,
    fps: float,
    output_dir: str | None = None,
    frame_texts: list[dict[str, Any]] | None = None,
) -> str:
    annotations = _parse_spatial_tags(output_text)
    rendered_frames: list[np.ndarray] = []

    for frame_idx, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for annotation in annotations:
            target_frame = annotation["frame_idx"]
            if target_frame is None or target_frame == frame_idx:
                _draw_annotation(draw, annotation, width, height)
        text_for_frame = output_text
        if frame_texts:
            for item in reversed(frame_texts):
                if int(item["start_frame"]) <= frame_idx <= int(item["end_frame"]):
                    text_for_frame = str(item["text"])
                    break
        image = _draw_text_panel(image, prompt, text_for_frame)
        rendered_frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        prefix="rynnbrain_cognition_overlay_",
        dir=output_dir,
        delete=False,
    )
    output_file.close()

    height, width = rendered_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise CognitionError("Failed to create overlay preview video.")
    try:
        for frame in rendered_frames:
            writer.write(frame)
    finally:
        writer.release()

    return _transcode_for_browser(output_file.name, output_dir)


def _transcode_for_browser(input_path: str, output_dir: str | None = None) -> str:
    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        prefix="rynnbrain_cognition_overlay_h264_",
        dir=output_dir,
        delete=False,
    )
    output_file.close()

    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_file.name,
                vcodec="libx264",
                pix_fmt="yuv420p",
                movflags="+faststart",
                **{"an": None},
            )
            .global_args("-hide_banner", "-loglevel", "error", "-nostdin")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        if os.path.exists(output_file.name):
            os.unlink(output_file.name)
        return input_path

    if os.path.exists(input_path):
        os.unlink(input_path)
    return output_file.name


@torch.inference_mode()
def run_inference(
    video_path: str | None,
    model_label: str,
    frame_fps: float,
    prompt_frame_count: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_dir: str | None = None,
) -> InferenceResult:
    return run_inference_sliding(
        video_path=video_path,
        model_label=model_label,
        frame_fps=frame_fps,
        prompt_frame_count=prompt_frame_count,
        window_stride=prompt_frame_count,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        output_dir=output_dir,
        progress_callback=None,
    )


@torch.inference_mode()
def run_inference_sliding(
    video_path: str | None,
    model_label: str,
    frame_fps: float,
    prompt_frame_count: int,
    window_stride: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_dir: str | None = None,
    progress_callback: Any | None = None,
) -> InferenceResult:
    video_path = validate_video_path(video_path)
    if not prompt.strip():
        raise CognitionError("Please enter an instruction prompt.")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    frames, decoded_count, expected_count = extract_frames(
        video_path,
        float(frame_fps),
        int(prompt_frame_count),
        select_prompt_frames=False,
    )
    windows = _sliding_windows(frames, int(prompt_frame_count), int(window_stride))
    timings["preprocess"] = time.perf_counter() - preprocess_start

    loaded = _load_model(model_label)
    processor = loaded.processor
    model = loaded.model
    device = _model_device(model)

    if temperature <= 0:
        sampling_params = {"do_sample": False}
    else:
        sampling_params = {
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
        }

    generation_start = time.perf_counter()
    window_outputs = []
    overlay_frame_texts = []
    for window_idx, window_frames in enumerate(windows):
        if progress_callback:
            progress_callback(window_idx, len(windows))

        pil_frames = _pil_frames(window_frames)
        messages = _build_messages(pil_frames, prompt.strip())
        model_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(device)

        _cuda_sync()
        generated_ids = model.generate(
            **model_inputs,
            **sampling_params,
            max_new_tokens=int(max_new_tokens),
        )
        _cuda_sync()

        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        window_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        start_frame = window_idx * int(window_stride)
        end_frame = min(start_frame + len(window_frames) - 1, len(frames) - 1)
        window_label = f"Window {window_idx + 1}/{len(windows)} (sampled frames {start_frame}-{end_frame})"
        window_output = f"{window_label}:\n{window_text}"
        window_outputs.append(window_output)
        overlay_frame_texts.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "text": f"{window_label}:\n{window_text}",
            }
        )

    timings["generation"] = time.perf_counter() - generation_start

    postprocess_start = time.perf_counter()
    output_text = "\n\n".join(window_outputs)
    overlay_path = render_overlay_video(
        frames,
        prompt.strip(),
        output_text,
        float(frame_fps),
        output_dir=output_dir,
        frame_texts=overlay_frame_texts,
    )
    timings["postprocess"] = time.perf_counter() - postprocess_start
    timings["total"] = time.perf_counter() - total_start

    expected_note = (
        f"{expected_count} expected at requested FPS, capped at {MAX_DECODED_FRAMES}"
        if expected_count > MAX_DECODED_FRAMES
        else f"{expected_count} expected at requested FPS"
    )
    metrics = "\n".join(
        [
            f"Model: `{loaded.model_id}`",
            f"Frames: decoded {decoded_count} ({expected_note}), candidate {len(frames)}",
            f"Sliding windows: {len(windows)} windows, size {int(prompt_frame_count)}, stride {int(window_stride)}",
            "",
            "Timing:",
            f"- preprocess: {timings['preprocess']:.2f} sec",
            f"- generation: {timings['generation']:.2f} sec",
            f"- postprocess: {timings['postprocess']:.2f} sec",
            f"- total: {timings['total']:.2f} sec",
            "",
            _vram_markdown(),
        ]
    )
    return InferenceResult(output_text=output_text, metrics=metrics, overlay_path=overlay_path)
