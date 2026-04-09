"""FastAPI server for video_voice_ai_manager."""

import asyncio
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import Config

app = FastAPI(title="Video Voice AI Manager", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_jobs: dict[str, dict] = {}
_ws_clients: list[WebSocket] = []
_tmp_dir = Path(tempfile.mkdtemp(prefix="vvam_"))

WEB_DIR = Path(__file__).parent
_static_dir = WEB_DIR / "static"
_static_dir.mkdir(exist_ok=True)
# Create a placeholder so StaticFiles doesn't fail on empty dir
_placeholder = _static_dir / ".gitkeep"
if not any(_static_dir.iterdir()):
    _placeholder.touch(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


async def _broadcast(job_id: str, data: dict):
    """Send progress update to all connected WebSocket clients."""
    message = {"job_id": job_id, **data}
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _get_transcriber(cfg: Config):
    """Instantiate transcriber based on config."""
    if cfg.transcriber == "gemini":
        from transcribers.gemini import GeminiTranscriber
        return GeminiTranscriber(api_key=cfg.gemini_api_key, model=cfg.gemini_audio_model)
    elif cfg.transcriber == "openai":
        from transcribers.openai_api import OpenAITranscriber
        return OpenAITranscriber(api_key=cfg.openai_api_key, model=cfg.openai_whisper_model)
    elif cfg.transcriber == "whisper-local":
        from transcribers.whisper_local import WhisperLocalTranscriber
        return WhisperLocalTranscriber(model=cfg.whisper_model, language=cfg.whisper_language)
    raise ValueError(f"Unknown transcriber: {cfg.transcriber}")


def _get_vision(cfg: Config):
    """Instantiate vision backend based on config."""
    if cfg.vision == "gemini":
        from vision.gemini import GeminiVision
        return GeminiVision(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
    elif cfg.vision == "openai":
        from vision.openai_api import OpenAIVision
        return OpenAIVision(api_key=cfg.openai_api_key, model=cfg.openai_model)
    elif cfg.vision == "ollama":
        from vision.ollama import OllamaVision
        return OllamaVision(url=cfg.ollama_url, model=cfg.ollama_model)
    raise ValueError(f"Unknown vision backend: {cfg.vision}")


async def _save_upload(file: UploadFile) -> Path:
    """Save uploaded file to temp directory."""
    suffix = Path(file.filename).suffix if file.filename else ""
    path = _tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
    return path


async def _process_video(job_id: str, video_path: Path, cfg: Config):
    """Background task: process a video file."""
    try:
        _jobs[job_id]["status"] = "processing"
        await _broadcast(job_id, {"status": "processing", "progress": 0, "step": "Starting..."})

        from core.video import VideoAnalyzer

        transcriber = _get_transcriber(cfg)
        vision = None
        try:
            vision = _get_vision(cfg)
        except Exception:
            pass

        analyzer = VideoAnalyzer(transcriber=transcriber, vision=vision, config=cfg)

        await _broadcast(job_id, {"status": "processing", "progress": 20, "step": "Analyzing video..."})
        video_result = await analyzer.analyze(source=str(video_path))

        result = {"transcription": None, "frames": [], "metadata": video_result.metadata}

        if video_result.transcription:
            t = video_result.transcription
            result["transcription"] = {
                "full_text": t.full_text,
                "language": t.language,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text,
                     "start_ts": s.start_ts, "end_ts": s.end_ts}
                    for s in t.segments
                ],
            }

        if video_result.frames:
            result["frames"] = [
                {"timestamp": a.timestamp, "timestamp_str": a.timestamp_str,
                 "description": a.description, "frame_path": a.frame_path}
                for a in video_result.frames
            ]

        # Generate report
        await _broadcast(job_id, {"status": "processing", "progress": 90, "step": "Generating report..."})
        try:
            from output.markdown import format_markdown
            result["report_md"] = format_markdown(
                transcript=video_result.transcription,
                frame_analyses=video_result.frames or None,
                metadata={"source": video_path.name},
            )
        except Exception:
            pass

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
        await _broadcast(job_id, {"status": "done", "progress": 100, "step": "Complete"})

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        await _broadcast(job_id, {"status": "error", "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    html_path = WEB_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/video")
async def api_video(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    transcriber: Optional[str] = Form(None),
    vision: Optional[str] = Form(None),
):
    """Upload a video file or provide URL for analysis."""
    cfg = Config.load()
    if transcriber:
        cfg.transcriber = transcriber
    if vision:
        cfg.vision = vision

    video_path = None

    if file and file.filename:
        video_path = await _save_upload(file)
    elif url:
        try:
            import yt_dlp
            out_path = _tmp_dir / f"{uuid.uuid4().hex}.mp4"
            ydl_opts = {"outtmpl": str(out_path), "format": cfg.ytdlp_format or "best[ext=mp4]/best", "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            # yt-dlp may add extension
            candidates = list(_tmp_dir.glob(f"{out_path.stem}.*"))
            video_path = candidates[0] if candidates else out_path
        except ImportError:
            raise HTTPException(status_code=400, detail="yt-dlp not installed. Install with: pip install yt-dlp")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide a file or url")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    asyncio.create_task(_process_video(job_id, video_path, cfg))

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/voice")
async def api_voice(
    file: UploadFile = File(...),
    transcriber: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """Upload an audio file for transcription."""
    cfg = Config.load()
    if transcriber:
        cfg.transcriber = transcriber

    audio_path = await _save_upload(file)

    try:
        t = _get_transcriber(cfg)
        lang = language or cfg.whisper_language or ""
        result = await t.transcribe(audio_path, language=lang)
        return {
            "full_text": result.full_text,
            "language": result.language,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text,
                 "start_ts": s.start_ts, "end_ts": s.end_ts}
                for s in result.segments
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/screenshot")
async def api_screenshot(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    timecode: str = Form("0"),
):
    """Extract a frame from a video at a given timecode."""
    if file and file.filename:
        vpath = await _save_upload(file)
    elif video_path:
        vpath = Path(video_path)
        if not vpath.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
    else:
        raise HTTPException(status_code=400, detail="Provide a file or video_path")

    # Parse timecode (supports seconds or HH:MM:SS)
    try:
        parts = timecode.split(":")
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            seconds = int(parts[0]) * 60 + float(parts[1])
        else:
            seconds = float(timecode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timecode: {timecode}")

    # Extract frame using ScreenshotExtractor
    out_dir = str(_tmp_dir)
    try:
        from core.screenshot import ScreenshotExtractor
        extractor = ScreenshotExtractor()
        out_path = await extractor.extract_at(str(vpath), seconds, output_dir=out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract frame: {e}")

    if not out_path.exists():
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    mime = "image/png" if str(out_path).endswith(".png") else "image/jpeg"
    return FileResponse(str(out_path), media_type=mime)


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    """Get the status of a background job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        resp["result"] = job["result"]
    elif job["status"] == "error":
        resp["error"] = job["error"]
    return resp


@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.get("/api/config")
async def api_config():
    """Return current configuration (safe fields only)."""
    cfg = Config.load()
    return {
        "transcriber": cfg.transcriber,
        "vision": cfg.vision,
        "video_fps": cfg.video_fps,
        "output_format": cfg.output_format,
        "whisper_language": cfg.whisper_language,
    }


def main():
    """Run the server."""
    import uvicorn
    cfg = Config.load()
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port)


if __name__ == "__main__":
    main()
