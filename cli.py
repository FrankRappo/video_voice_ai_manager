"""
VVAM CLI — command-line entry point for Video Voice AI Manager.

Usage:
    vvam video <source> [options]
    vvam voice <source> [options]
    vvam dictate [options]
    vvam screenshot <source> [options]
    vvam server [options]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from config import Config


def _is_pipe() -> bool:
    """True when stdout is not a terminal (piped or redirected)."""
    return not sys.stdout.isatty()


def _parse_time(ts: str) -> float:
    """Parse a time string like '00:05', '00:05:30', or '5.0' into seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


# ── Subcommand handlers ─────────────────────────────────────────

async def cmd_video(args: argparse.Namespace, cfg: Config) -> None:
    """Analyze a video file or URL."""
    from core.video import VideoAnalyzer
    from transcribers import get_transcriber
    from vision import get_vision

    is_client_feedback = args.format == "client-feedback"

    transcriber = get_transcriber(cfg)
    vision_backend = get_vision(cfg) if not args.audio_only else None
    analyzer = VideoAnalyzer(transcriber=transcriber, vision=vision_backend, config=cfg)

    time_from = _parse_time(args.time_from) if args.time_from else None
    time_to = _parse_time(args.time_to) if args.time_to else None

    result = await analyzer.analyze(
        source=args.source,
        audio_only=args.audio_only,
        time_from=time_from,
        time_to=time_to,
    )

    metadata = {
        "source": args.source,
        "transcriber": cfg.transcriber,
        "vision": cfg.vision,
    }
    if result.duration:
        metadata["duration"] = result.duration

    if is_client_feedback:
        # Run correlation pipeline using direct multimodal mode
        # (sends images + transcription in a single API call — much more efficient)
        from core.correlator import Correlator
        from output.client_feedback import format_client_feedback

        correlator = Correlator(api_key=cfg.gemini_api_key, model=cfg.gemini_model)

        if result.frames:
            # Collect frame paths for direct image mode
            frame_paths = [
                (fa.timestamp, Path(fa.frame_path))
                for fa in result.frames
                if Path(fa.frame_path).exists()
            ]
            if frame_paths:
                correlation = await correlator.correlate_with_images(
                    result.transcription, frame_paths, max_frames=15,
                )
            else:
                correlation = await correlator.correlate(result.transcription, result.frames)
        else:
            correlation = await correlator.correlate(result.transcription, [])

        text = format_client_feedback(correlation, metadata)
    else:
        from output.formatters import get_formatter
        fmt = get_formatter(args.format)

        transcript = result.transcription
        frames = result.frames or None

        if args.format == "srt":
            text = fmt(transcript)
        else:
            text = fmt(transcript, frames, metadata)

    _output(text, args.output)


async def cmd_voice(args: argparse.Namespace, cfg: Config) -> None:
    """Analyze voice messages."""
    from core.voice import VoiceAnalyzer
    from transcribers import get_transcriber

    transcriber = get_transcriber(cfg)
    analyzer = VoiceAnalyzer(transcriber=transcriber, config=cfg)

    results = await analyzer.analyze(args.source)

    from output.formatters import get_formatter
    from transcribers.base import TranscriptionResult
    fmt = get_formatter(args.format)

    metadata = {
        "source": args.source,
        "transcriber": cfg.transcriber,
    }

    # Merge all voice results into a single transcription
    all_segments = []
    language = ""
    for vr in results:
        all_segments.extend(vr.transcription.segments)
        if vr.transcription.language and not language:
            language = vr.transcription.language

    transcript = TranscriptionResult(segments=all_segments, language=language) if all_segments else None

    if args.format == "srt":
        text = fmt(transcript)
    else:
        text = fmt(transcript, None, metadata)

    _output(text, args.output)


async def cmd_dictate(args: argparse.Namespace, cfg: Config) -> None:
    """Dictation mode — transcribe from file or microphone."""
    from core.dictate import Dictator
    from transcribers import get_transcriber

    transcriber = get_transcriber(cfg)
    dictator = Dictator(transcriber=transcriber, config=cfg)

    if args.file:
        text_result = await dictator.dictate(args.file)
    elif args.mic:
        text_result = await dictator.record_and_dictate()
    else:
        print("Error: provide --file or --mic", file=sys.stderr)
        sys.exit(1)

    # For dictate, just output the plain text (pipe-friendly)
    _output(text_result, args.output)


async def cmd_screenshot(args: argparse.Namespace, cfg: Config) -> None:
    """Extract and analyze frames from a video."""
    from core.screenshot import ScreenshotExtractor

    extractor = ScreenshotExtractor(config=cfg)
    output_dir = cfg.output_dir

    if args.time:
        ts = _parse_time(args.time)
        path = await extractor.extract_at(args.source, ts, output_dir=output_dir)
        print(f"Screenshot saved: {path}")
    elif args.time_from and args.time_to:
        t_from = _parse_time(args.time_from)
        t_to = _parse_time(args.time_to)
        fps = args.fps if args.fps else cfg.video_fps
        paths = await extractor.extract_range(
            args.source, t_from, t_to, fps=fps, output_dir=output_dir,
        )
        for p in paths:
            print(f"Screenshot saved: {p}")
    else:
        print("Error: provide --time or --from/--to", file=sys.stderr)
        sys.exit(1)


def cmd_server(args: argparse.Namespace, cfg: Config) -> None:
    """Start the web server."""
    from web.server import main as run_server
    run_server()


def _output(text: str, output_path: str | None) -> None:
    """Write result to file or stdout."""
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        if not _is_pipe():
            print(f"Saved to {output_path}")
    else:
        print(text)


# ── Argument parser ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vvam",
        description="Video Voice AI Manager — universal AI-powered analyzer",
    )

    # Global options
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--gemini-key", help="Gemini API key")
    parser.add_argument("--openai-key", help="OpenAI API key")

    sub = parser.add_subparsers(dest="command")

    # ── video ────────────────────────────────────────────────
    p_video = sub.add_parser("video", help="Analyze video (file or URL)")
    p_video.add_argument("source", help="Video file path or URL")
    p_video.add_argument("--transcriber", choices=["gemini", "whisper-local", "openai"])
    p_video.add_argument("--vision", choices=["gemini", "openai", "ollama"])
    p_video.add_argument("-o", "--output", help="Output file path")
    p_video.add_argument("-f", "--format", default="markdown",
                         choices=["markdown", "json", "srt", "client-feedback"])
    p_video.add_argument("--chunk", type=int, help="Chunk duration in minutes")
    p_video.add_argument("--audio-only", action="store_true",
                         help="Transcribe audio only, skip vision")
    p_video.add_argument("--from", dest="time_from", help="Start time (e.g. 00:05:00)")
    p_video.add_argument("--to", dest="time_to", help="End time (e.g. 00:15:00)")
    p_video.add_argument("--fps", type=float, help="Frame extraction rate")
    p_video.add_argument("--prompt", help="Custom vision prompt")

    # ── voice ────────────────────────────────────────────────
    p_voice = sub.add_parser("voice", help="Analyze voice messages")
    p_voice.add_argument("source", help="Audio file or directory path")
    p_voice.add_argument("--transcriber", choices=["gemini", "whisper-local", "openai"])
    p_voice.add_argument("-f", "--format", default="markdown",
                         choices=["markdown", "json", "srt"])
    p_voice.add_argument("-o", "--output", help="Output file path")
    p_voice.add_argument("--all", action="store_true",
                         help="Batch process all files in directory")

    # ── dictate ──────────────────────────────────────────────
    p_dictate = sub.add_parser("dictate", help="Dictation mode")
    p_dictate.add_argument("--file", help="Audio file to transcribe")
    p_dictate.add_argument("--mic", action="store_true",
                           help="Record from microphone")
    p_dictate.add_argument("--transcriber", choices=["gemini", "whisper-local", "openai"])
    p_dictate.add_argument("-f", "--format", default="markdown",
                           choices=["markdown", "json", "srt"])
    p_dictate.add_argument("-o", "--output", help="Output file path")

    # ── screenshot ───────────────────────────────────────────
    p_screen = sub.add_parser("screenshot", help="Extract and analyze frames")
    p_screen.add_argument("source", help="Video file path")
    p_screen.add_argument("--time", help="Single timestamp to extract")
    p_screen.add_argument("--from", dest="time_from", help="Start time")
    p_screen.add_argument("--to", dest="time_to", help="End time")
    p_screen.add_argument("--fps", type=float, help="Frame extraction rate")
    p_screen.add_argument("-f", "--format", default="markdown",
                          choices=["markdown", "json"])
    p_screen.add_argument("-o", "--output", help="Output file path")

    # ── server ───────────────────────────────────────────────
    p_server = sub.add_parser("server", help="Start web server")
    p_server.add_argument("--host", default=None, help="Bind host")
    p_server.add_argument("--port", type=int, default=None, help="Bind port")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load config and apply CLI overrides
    cfg = Config.load(args.config)

    if args.gemini_key:
        cfg.gemini_api_key = args.gemini_key
    if args.openai_key:
        cfg.openai_api_key = args.openai_key

    # Apply backend overrides from subcommand options
    if hasattr(args, "transcriber") and args.transcriber:
        cfg.transcriber = args.transcriber
    if hasattr(args, "vision") and args.vision:
        cfg.vision = args.vision
    if hasattr(args, "chunk") and args.chunk:
        cfg.video_chunk_minutes = args.chunk
    if hasattr(args, "fps") and args.fps:
        cfg.video_fps = args.fps
    if hasattr(args, "prompt") and args.prompt:
        cfg.vision_prompt = args.prompt

    # Dispatch
    handlers = {
        "video": cmd_video,
        "voice": cmd_voice,
        "dictate": cmd_dictate,
        "screenshot": cmd_screenshot,
        "server": cmd_server,
    }
    handler = handlers[args.command]

    if args.command == "server":
        if args.host:
            cfg.web_host = args.host
        if args.port:
            cfg.web_port = args.port
        handler(args, cfg)
    else:
        asyncio.run(handler(args, cfg))


if __name__ == "__main__":
    main()
