import argparse
import os

import gradio as gr

from cognition_core import (
    DEFAULT_FRAME_FPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_LABEL,
    DEFAULT_PROMPT,
    DEFAULT_PROMPT_FRAME_COUNT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MODEL_CHOICES,
    CognitionError,
    run_inference as run_core_inference,
)


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
) -> tuple[str, str, str]:
    try:
        result = run_core_inference(
            video_path=video_path,
            model_label=model_label,
            frame_fps=frame_fps,
            prompt_frame_count=prompt_frame_count,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except CognitionError as exc:
        raise gr.Error(str(exc)) from exc
    return result.output_text, result.metrics, result.overlay_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RynnBrain Cognition GUI") as app:
        gr.Markdown("# RynnBrain Cognition GUI")
        gr.Markdown(
            "Upload a `.mp4` or `.webm` video, choose a RynnBrain model, "
            "sample frames, and render the instruction plus model output on a preview video."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(
                    label="Input video (.mp4 / .webm)",
                    sources=["upload"],
                    include_audio=False,
                )
                model = gr.Dropdown(
                    choices=list(MODEL_CHOICES.keys()),
                    value=DEFAULT_MODEL_LABEL,
                    label="Model",
                )
                frame_fps = gr.Number(
                    value=DEFAULT_FRAME_FPS,
                    precision=2,
                    label="Frame extraction FPS",
                )
                prompt_frame_count = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=DEFAULT_PROMPT_FRAME_COUNT,
                    step=1,
                    label="Images passed to prompt",
                )
                prompt = gr.Textbox(
                    value=DEFAULT_PROMPT,
                    lines=5,
                    label="Instruction prompt",
                )

                with gr.Accordion("Generation settings", open=False):
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=1024,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        step=1,
                        label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=DEFAULT_TEMPERATURE,
                        step=0.01,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=DEFAULT_TOP_P,
                        step=0.01,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=DEFAULT_TOP_K,
                        step=1,
                        label="Top-k",
                    )

                submit = gr.Button("Run inference", variant="primary")

            with gr.Column(scale=1):
                overlay_video = gr.Video(
                    label="Overlay preview",
                    include_audio=False,
                    autoplay=False,
                )
                output = gr.Textbox(
                    label="Model output",
                    lines=8,
                )
                metrics = gr.Markdown(label="Metrics")

        submit.click(
            run_inference,
            inputs=[
                video,
                model,
                frame_fps,
                prompt_frame_count,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=[output, metrics, overlay_video],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RynnBrain Cognition Gradio GUI")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8061, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    os.environ.setdefault("TORCHCODEC_NUM_THREADS", "1")
    args = parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
