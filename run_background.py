#!/usr/bin/env python3
"""Launch existing VVAM file-processing commands without waiting in a terminal."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "vvam_output" / "background"
PUBLIC_ENV = ("VVAM_TRANSCRIBER", "VVAM_GEMINI_AUDIO_MODEL", "VVAM_WHISPER_MODEL")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def alive(record):
    try:
        fields = Path("/proc", str(record["pid"]), "stat").read_text().split(") ", 1)[1].split()
        return fields[0] != "Z" and fields[19] == record["starttime"]
    except (OSError, KeyError):
        return False


def inspect_job(directory):
    record = json.loads((directory / "job.json").read_text())
    result = directory / "result.json"
    record["status"] = "running" if alive(record) else "stopped_without_result"
    if result.exists():
        record.update(json.loads(result.read_text()))
    return record


def worker(directory):
    with (directory / "start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        record = json.loads((directory / "job.json").read_text())
    env = dict(os.environ, TMPDIR=str(directory / "tmp"), PYTHONUNBUFFERED="1")
    try:
        with (directory / "run.log").open("ab") as log:
            result = subprocess.run(
                [sys.executable, str(ROOT / "cli.py"), *record["argv"]],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
        data = {"exit_code": result.returncode,
                "status": "completed" if result.returncode == 0 else "failed"}
    except Exception as error:
        data = {"status": "failed", "error_type": type(error).__name__}
    output = Path(record["output"])
    data.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                output_exists=output.is_file(),
                output_bytes=output.stat().st_size if output.is_file() else 0)
    write_json(directory / "result.json", data)


def main(argv):
    os.umask(0o077)
    if argv[:1] == ["--worker"]:
        directory = Path(argv[1]).resolve()
        if directory.parent != JOBS or directory.is_symlink():
            raise ValueError("Invalid job directory")
        worker(directory)
        return
    if argv[:1] == ["status"]:
        directory = Path(argv[1]).resolve()
        if directory.parent != JOBS:
            raise ValueError("Status requires a VVAM background job directory")
        print(json.dumps(inspect_job(directory), ensure_ascii=False))
        return
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python3 run_background.py voice|video|screenshot FILE [CLI options]\n"
              "       python3 run_background.py status JOB_DIRECTORY")
        return
    if argv[0] not in {"voice", "video", "screenshot"}:
        raise ValueError("Only noninteractive file-processing commands are supported")
    if len(argv) < 2 or argv[1].startswith("-"):
        raise ValueError("Provide the input file or URL")
    source = Path(argv[1]).expanduser()
    source_sha256 = None
    if source.exists():
        source = source.resolve()
        argv[1] = str(source)
        if source.is_file():
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            source_sha256 = digest.hexdigest()
    requested_output = None
    for i, arg in enumerate(argv):
        if arg in {"-o", "--output"}:
            requested_output = str(Path(argv[i + 1]).expanduser().resolve())
            argv[i + 1] = requested_output
        elif arg.startswith("--output="):
            requested_output = str(Path(arg.split("=", 1)[1]).expanduser().resolve())
            argv[i] = "--output=" + requested_output
    identity = {"argv": argv, "source_sha256": source_sha256,
                "model_overrides": {k: os.environ.get(k) for k in PUBLIC_ENV}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for prior in sorted(JOBS.glob("*/job.json")):
            record = json.loads(prior.read_text())
            if record.get("fingerprint") == fingerprint:
                status = inspect_job(prior.parent)
                if status["status"] == "running" or (
                    status["status"] == "completed" and status.get("output_exists")
                ):
                    print(json.dumps(dict(status, reused=True), ensure_ascii=False))
                    return
        if requested_output and Path(requested_output).exists():
            raise FileExistsError("Output exists; choose another output path")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = JOBS / (stamp + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(mode=0o700)
        (directory / "tmp").mkdir(mode=0o700)
        output = requested_output or str(directory / "transcript.md")
        if not requested_output:
            argv = [*argv, "-o", output]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        record = dict(identity, argv=argv, fingerprint=fingerprint, output=output,
                      job_dir=str(directory), log=str(directory / "run.log"),
                      started_at=dt.datetime.now(dt.timezone.utc).isoformat())
        # The worker waits on this lock until its PID record has been published.
        with (directory / "start.lock").open("a") as start_lock:
            fcntl.flock(start_lock, fcntl.LOCK_EX)
            write_json(directory / "job.json", record)
            with (directory / "worker.log").open("ab") as log:
                child = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, start_new_session=True,
                )
            fields = Path("/proc", str(child.pid), "stat").read_text().split(") ", 1)[1].split()
            record.update(pid=child.pid, starttime=fields[19])
            write_json(directory / "job.json", record)
        print(json.dumps(dict(record, status="started", reused=False), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
