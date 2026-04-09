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


# ── Subcommand handlers ─────────────────────────────────────────

async def cmd_video(args: argparse.Namespace, cfg: Config) -> None:
    """Analyze a video file or URL."""
    from core.video import process_video  # type: ignore[import-not-found]

    result = await process_video(
        source=args.source,
        config=cfg,
        audio_only=args.audio_only,
        time_from=args.time_from,
        time_to=args.time_to,
    )

    from output.formatters import get_formatter
    fmt = get_formatter(args.format)

    metadata = {
        "source": args.source,
        "transcriber": cfg.transcriber,
        "vision": cfg.vision,
    }
    if hasattr(result, "duration"):
        metadata["duration"] = result.duration

    transcript = getattr(result, "transcript", None)
    frames = getattr(result, "frames", None)

    if args.format == "srt":
        text = fmt(transcript)
    else:
        text = fmt(transcript, frames, metadata)

    _output(text, args.output)


async def cmd_voice(args: argparse.Namespace, cfg: Config) -> None:
    """Analyze voice messages."""
    from core.voice import process_voice  # type: ignore[import-not-found]

    result = await process_voice(
        source=args.source,
        config=cfg,
        batch_all=args.all,
    )

    from output.formatters import get_formatter
    fmt = get_formatter(args.format)

    metadata = {
        "source": args.source,
        "transcriber": cfg.transcriber,
    }

    transcript = getattr(result, "transcript", None)

    if args.format == "srt":
        text = fmt(transcript)
    else:
        text = fmt(transcript, None, metadata)

    _output(text, args.output)


async def cmd_dictate(args: argparse.Namespace, cfg: Config) -> None:
    """Dictation mode — transcribe from file or microphone."""
    from core.dictate import process_dictate  # type: ignore[import-not-found]

    source = args.file or args.mic
    result = await process_dictate(source=source, config=cfg)

    from output.formatters import get_formatter
    fmt = get_formatter(args.format)

    transcript = getattr(result, "transcript", None)
    metadata = {"source": source or "microphone", "transcriber": cfg.transcriber}

    if args.format == "srt":
        text = fmt(transcript)
    else:
        text = fmt(transcript, None, metadata)

    _output(text, args.output)


async def cmd_screenshot(args: argparse.Namespace, cfg: Config) -> None:
    """Extract and analyze frames from a video."""
    from core.screenshot import process_screenshot  # type: ignore[import-not-found]

    result = await process_screenshot(
        source=args.source,
        config=cfg,
        time=args.time,
        time_from=args.time_from,
        time_to=args.time_to,
    )

    from output.formatters import get_formatter
    fmt = get_formatter(args.format)

    frames = getattr(result, "frames", None)
    metadata = {"source": args.source, "vision": cfg.vision}

    text = fmt(None, frames, metadata)
    _output(text, args.output)


def cmd_server(args: argparse.Namespace, cfg: Config) -> None:
    """Start the web server."""
    from web.app import run_server  # type: ignore[import-not-found]
    run_server(host=args.host, port=args.port, config=cfg)


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
                         choices=["markdown", "json", "srt"])
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
