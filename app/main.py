from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from itertools import chain
from typing import Iterable, Iterator, Mapping

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from openai import OpenAI
from pydub import AudioSegment
from starlette.background import BackgroundTask

from .config import get_settings
from .srt_utils import build_single_segment, iter_srt_blocks, segments_to_srt

app = FastAPI(title="AI Subtitle Generator", version="1.0.0")

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MiB service-side ceiling
MIN_CHUNK_DURATION_MS = 1_000  # 1 second lower bound when splitting audio
CHUNK_EXPORT_BITRATE = "128k"
NORMALIZED_SAMPLE_RATE = 16_000

UPLOAD_PAGE_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>AI Subtitle Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 720px; margin: 0 auto; padding: 24px; background: #f8f9fb; color: #1f2933; }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        p { margin-bottom: 1rem; }
        .uploader { border: 2px dashed #6c5ce7; padding: 32px; text-align: center; background: #fff; border-radius: 12px; transition: border-color 0.2s, background 0.2s; }
        .uploader.dragover { border-color: #0984e3; background: #ecf5ff; }
        .controls { margin-top: 16px; display: flex; gap: 12px; align-items: center; justify-content: center; flex-wrap: wrap; }
        input[type=\"number\"] { width: 80px; padding: 6px; }
        button { background: #6c5ce7; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
        button.secondary { background: #d63031; }
        button:disabled { background: #a5a5a5; cursor: not-allowed; }
        #status { margin-top: 16px; min-height: 24px; font-size: 0.95rem; }
        #download-link { display: none; margin-top: 16px; }
        #live-output { display: none; background: #fff; border: 1px solid #dcdfe6; padding: 12px; margin-top: 16px; max-height: 240px; overflow-y: auto; white-space: pre-wrap; font-family: "Courier New", monospace; font-size: 0.9rem; border-radius: 8px; }
        footer { margin-top: 48px; font-size: 0.85rem; color: #555; }
        a { color: #0984e3; }
    </style>
</head>
<body>
    <h1>AI Subtitle Generator</h1>
    <p>Drop an audio or video file below to receive translated English subtitles in SRT format.</p>
    <div id=\"drop-zone\" class=\"uploader\">
        <strong>Drag &amp; drop</strong> an MP3/MP4 file here or
        <br />
        <label for=\"file-input\" style=\"cursor:pointer;color:#0984e3;text-decoration:underline;\">browse</label>
        <input id=\"file-input\" type=\"file\" accept=\"audio/*,video/*\" style=\"display:none\" />
        <div class=\"controls\">
            <label>Max minutes
                <input id=\"minutes\" type=\"number\" min=\"1\" max=\"240\" placeholder=\"optional\" />
            </label>
            <button id=\"upload-btn\">Generate Subtitles</button>
            <button id=\"stop-btn\" class=\"secondary\" type=\"button\" disabled>Cancel</button>
        </div>
    </div>
    <div id=\"status\"></div>
    <pre id=\"live-output\"></pre>
    <a id=\"download-link\" href=\"#\" download=\"subtitles.srt\">Download subtitles</a>

    <footer>
        <p>Powered by OpenAI Whisper via FastAPI. Audio is normalised to mono 16&nbsp;kHz before transcription.</p>
    </footer>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');
        const stopBtn = document.getElementById('stop-btn');
        const statusEl = document.getElementById('status');
        const downloadLink = document.getElementById('download-link');
        const minutesInput = document.getElementById('minutes');
        const liveOutput = document.getElementById('live-output');

        let selectedFile = null;
        let processingTimer = null;
        let currentRequest = null;
        let requestAborted = false;

        function stopProcessingStatus() {
            if (processingTimer) {
                clearInterval(processingTimer);
                processingTimer = null;
            }
        }

        function setStatus(message, isError = false) {
            stopProcessingStatus();
            statusEl.textContent = message;
            statusEl.style.color = isError ? '#d63031' : '#1f2933';
        }

        function startProcessingStatus() {
            stopProcessingStatus();
            const started = Date.now();
            processingTimer = setInterval(() => {
                const elapsed = ((Date.now() - started) / 1000).toFixed(1);
                const dots = '.'.repeat((Math.floor((Date.now() - started) / 700) % 4) + 1);
                statusEl.style.color = '#1f2933';
                statusEl.textContent = `Processing on server${dots} (${elapsed}s)`;
            }, 700);
            statusEl.textContent = 'Processing on server…';
            statusEl.style.color = '#1f2933';
        }

        function resetDownload() {
            downloadLink.style.display = 'none';
            if (downloadLink.href) {
                URL.revokeObjectURL(downloadLink.href);
            }
            downloadLink.removeAttribute('href');
        }

        function resetLiveOutput() {
            liveOutput.textContent = '';
            liveOutput.style.display = 'none';
        }

        function appendLiveText(text) {
            if (!text) {
                return;
            }
            if (liveOutput.style.display !== 'block') {
                liveOutput.style.display = 'block';
            }
            liveOutput.textContent += text;
            liveOutput.scrollTop = liveOutput.scrollHeight;
        }

        function handleFiles(files) {
            if (!files || !files.length) {
                return;
            }
            selectedFile = files[0];
            setStatus(`Selected: ${selectedFile.name}`);
            resetDownload();
            resetLiveOutput();
        }

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, event => {
                event.preventDefault();
                event.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
        });

        dropZone.addEventListener('drop', event => {
            handleFiles(event.dataTransfer.files);
        });

        fileInput.addEventListener('change', event => {
            handleFiles(event.target.files);
        });

        uploadBtn.addEventListener('click', () => {
            if (!selectedFile) {
                fileInput.click();
                return;
            }

            uploadBtn.disabled = true;
            stopBtn.disabled = false;
            resetDownload();
            resetLiveOutput();
            const formData = new FormData();
            formData.append('file', selectedFile);

            const params = new URLSearchParams();
            params.set('stream', 'true');
            const minutes = minutesInput.value.trim();
            if (minutes) {
                params.set('max_duration_minutes', minutes);
            }

            let url = '/generate-subtitles';
            const query = params.toString();
            if (query) {
                url += `?${query}`;
            }

            const xhr = new XMLHttpRequest();
            xhr.open('POST', url, true);
            xhr.responseType = 'text';
            currentRequest = xhr;
            requestAborted = false;

            let uploadStarted = null;
            let lastLength = 0;

            xhr.upload.onloadstart = () => {
                uploadStarted = Date.now();
                setStatus('Upload started…');
            };

            xhr.upload.onprogress = event => {
                if (event.lengthComputable) {
                    const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
                    statusEl.textContent = `Uploading… ${percent}%`;
                } else {
                    statusEl.textContent = 'Uploading…';
                }
            };

            xhr.upload.onload = () => {
                if (uploadStarted) {
                    const seconds = ((Date.now() - uploadStarted) / 1000).toFixed(1);
                    statusEl.textContent = `Upload complete (${seconds}s). Preparing transcription…`;
                }
                startProcessingStatus();
            };

            xhr.onprogress = () => {
                const response = xhr.responseText || '';
                if (!response || response.length <= lastLength) {
                    return;
                }
                appendLiveText(response.substring(lastLength));
                lastLength = response.length;
            };

            xhr.onreadystatechange = () => {
                if (xhr.readyState !== XMLHttpRequest.DONE) {
                    return;
                }

                if (requestAborted) {
                    return;
                }

                stopProcessingStatus();
                uploadBtn.disabled = false;
                stopBtn.disabled = true;

                if (xhr.status >= 200 && xhr.status < 300) {
                    const response = xhr.responseText || '';
                    if (response.length > lastLength) {
                        appendLiveText(response.substring(lastLength));
                        lastLength = response.length;
                    }
                    const srtText = xhr.responseText || '';
                    const blob = new Blob([srtText], { type: 'text/plain;charset=utf-8' });
                    const objectUrl = URL.createObjectURL(blob);

                    downloadLink.href = objectUrl;
                    const baseName = selectedFile.name.replace(/[.][^.]+$/, '') || 'subtitles';
                    downloadLink.download = `${baseName}.srt`;
                    downloadLink.style.display = 'inline-block';
                    setStatus('Completed. Click to download your subtitles.');
                } else {
                    const message = xhr.responseText || `Request failed with status ${xhr.status}`;
                    setStatus(message, true);
                }

                currentRequest = null;
                selectedFile = null;
                fileInput.value = '';
            };

            xhr.onerror = () => {
                stopProcessingStatus();
                uploadBtn.disabled = false;
                stopBtn.disabled = true;
                currentRequest = null;
                selectedFile = null;
                fileInput.value = '';
                setStatus('Network error while uploading.', true);
            };

            xhr.ontimeout = () => {
                stopProcessingStatus();
                uploadBtn.disabled = false;
                stopBtn.disabled = true;
                currentRequest = null;
                selectedFile = null;
                fileInput.value = '';
                setStatus('Request timed out.', true);
            };

            xhr.onabort = () => {
                stopProcessingStatus();
                uploadBtn.disabled = false;
                stopBtn.disabled = true;
                currentRequest = null;
                resetDownload();
                selectedFile = null;
                fileInput.value = '';
                setStatus('Upload cancelled.');
            };

            xhr.send(formData);
        });

        stopBtn.addEventListener('click', () => {
            if (currentRequest) {
                requestAborted = true;
                setStatus('Cancelling…');
                currentRequest.abort();
            }
        });

        dropZone.addEventListener('click', event => {
            if (event.target === uploadBtn) {
                return;
            }
            fileInput.click();
        });
    </script>
</body>
</html>
"""


def _cleanup_paths(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _segment_iter(segments: Iterable[object] | None) -> list[Mapping[str, object]]:
    cleaned: list[Mapping[str, object]] = []
    if not segments:
        return cleaned

    for segment in segments:
        if isinstance(segment, Mapping):
            cleaned.append(segment)
            continue
        cleaned.append(
            {
                "start": getattr(segment, "start", 0.0),
                "end": getattr(segment, "end", 0.0),
                "text": getattr(segment, "text", ""),
            }
        )
    return cleaned


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Upload an audio or video file to /generate-subtitles"}


@app.get("/ui", response_class=HTMLResponse)
async def upload_interface() -> str:
    return UPLOAD_PAGE_HTML


@app.post("/generate-subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    max_duration_minutes: int | None = Query(
        default=None,
        ge=1,
        le=240,
        description="Limit transcription to the first N minutes of audio",
    ),
    stream: bool = Query(
        default=False,
        description="Stream subtitle output as it is generated",
    ),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename")

    allowed_fallback_types = {"application/octet-stream"}
    if file.content_type and not (
        file.content_type.startswith("audio/")
        or file.content_type.startswith("video/")
        or file.content_type in allowed_fallback_types
    ):
        raise HTTPException(
            status_code=400,
            detail="Only audio or video uploads are supported",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp3"
    ) as temp:
        temp.write(data)
        temp_path = temp.name

    cleanup_paths: list[str] = [temp_path]

    try:
        normalized_path = _ensure_mono_mp3(temp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to normalize audio: {exc}") from exc

    cleanup_paths.append(normalized_path)

    trimmed_path = normalized_path
    if max_duration_minutes is not None:
        try:
            trimmed_path = _trim_audio_to_duration(
                normalized_path, max_duration_minutes * 60 * 1000
            )
            cleanup_paths.append(trimmed_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to trim audio: {exc}") from exc

    basename, _ = os.path.splitext(os.path.basename(file.filename))
    safe_name = basename or "transcription"

    headers = {
        "Content-Disposition": f"attachment; filename=\"{safe_name}.srt\"",
    }
    segment_iter = _transcribe_with_chunking(
        client, settings.openai_model, trimmed_path
    )

    if stream:
        try:
            first_segment = next(segment_iter)
        except StopIteration as exc:
            _cleanup_paths(cleanup_paths)
            raise HTTPException(
                status_code=502, detail="No transcription segments returned by OpenAI"
            ) from exc

        streaming_iter = iter_srt_blocks(chain([first_segment], segment_iter))
        background = BackgroundTask(_cleanup_paths, cleanup_paths)
        return StreamingResponse(
            streaming_iter,
            media_type="application/x-subrip",
            headers=headers,
            background=background,
        )

    try:
        segments = list(segment_iter)
    finally:
        _cleanup_paths(cleanup_paths)

    if not segments:
        raise HTTPException(status_code=502, detail="No transcription segments returned by OpenAI")

    srt_payload = segments_to_srt(segments)
    if not srt_payload:
        raise HTTPException(status_code=502, detail="Unable to create SRT output")

    return PlainTextResponse(
        content=srt_payload,
        media_type="application/x-subrip",
        headers=headers,
    )


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def _transcribe_with_chunking(
    client: OpenAI, model: str, audio_path: str
) -> Iterator[dict[str, object]]:
    file_size = os.path.getsize(audio_path)
    emitted = False

    def _emit_segments(
        raw_segments: Iterable[Mapping[str, object]], *, offset_seconds: float
    ) -> Iterator[dict[str, object]]:
        nonlocal emitted
        for seg in _segment_iter(raw_segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0) or 0.0) + offset_seconds
            end = float(seg.get("end", start) or start) + offset_seconds
            emitted = True
            yield {"start": start, "end": end, "text": text}

    if file_size <= MAX_AUDIO_BYTES:
        segments, transcript_text = _translate_and_normalize(
            client, model, audio_path, offset_seconds=0.0
        )
        yield from _emit_segments(segments, offset_seconds=0.0)
        if not emitted:
            yield from _emit_segments(
                build_single_segment(transcript_text), offset_seconds=0.0
            )
        return

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as exc:  # pragma: no cover - decode failures are user errors
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    duration_ms = len(audio)
    if duration_ms <= 0:
        raise HTTPException(status_code=400, detail="Uploaded audio has zero duration")

    bytes_per_ms = max(file_size / duration_ms, 1e-6)
    max_chunk_ms = max(int(MAX_AUDIO_BYTES / bytes_per_ms), MIN_CHUNK_DURATION_MS)

    collected_text: list[str] = []

    start_ms = 0
    while start_ms < duration_ms:
        current_chunk_ms = max(
            MIN_CHUNK_DURATION_MS, min(max_chunk_ms, duration_ms - start_ms)
        )
        chunk_path = None
        end_ms_snapshot = start_ms

        while current_chunk_ms >= MIN_CHUNK_DURATION_MS:
            end_ms = min(duration_ms, start_ms + current_chunk_ms)
            chunk = audio[start_ms:end_ms]
            if len(chunk) == 0:
                break

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as chunk_file:
                chunk.export(chunk_file.name, format="mp3", bitrate=CHUNK_EXPORT_BITRATE)
                chunk_path = chunk_file.name

            chunk_size = os.path.getsize(chunk_path)
            if chunk_size <= MAX_AUDIO_BYTES:
                end_ms_snapshot = end_ms
                break

            os.remove(chunk_path)
            chunk_path = None

            if current_chunk_ms <= MIN_CHUNK_DURATION_MS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Unable to split audio below OpenAI size limit. Reduce bitrate or duration and try again."
                    ),
                )

            current_chunk_ms = max(MIN_CHUNK_DURATION_MS, current_chunk_ms // 2)

        if not chunk_path:
            start_ms = duration_ms
            continue

        try:
            chunk_segments, chunk_text = _translate_and_normalize(
                client,
                model,
                chunk_path,
                offset_seconds=start_ms / 1000.0,
            )
        finally:
            try:
                os.remove(chunk_path)
            except OSError:
                pass

        yield from _emit_segments(
            chunk_segments, offset_seconds=start_ms / 1000.0
        )
        if chunk_text:
            collected_text.append(chunk_text)

        start_ms = min(duration_ms, end_ms_snapshot)

    if not emitted:
        fallback_text = " ".join(
            text.strip() for text in collected_text if text and text.strip()
        )
        yield from _emit_segments(
            build_single_segment(fallback_text), offset_seconds=0.0
        )


def _translate_and_normalize(
    client: OpenAI, model: str, audio_path: str, *, offset_seconds: float
) -> tuple[list[dict[str, object]], str]:
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
            )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc

    segment_dicts = []
    for seg in _segment_iter(getattr(transcription, "segments", None)):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0) + offset_seconds
        end = float(seg.get("end", start) or start) + offset_seconds
        segment_dicts.append({"start": start, "end": end, "text": text})

    if not segment_dicts:
        segment_dicts = build_single_segment(getattr(transcription, "text", ""))

        for seg in segment_dicts:
            seg["start"] = float(seg.get("start", 0.0) or 0.0) + offset_seconds
            seg["end"] = float(seg.get("end", seg["start"]) or seg["start"]) + offset_seconds

    return segment_dicts, getattr(transcription, "text", "")


def _ensure_mono_mp3(source_path: str) -> str:
    ffmpeg_executable = shutil.which("ffmpeg")
    if not ffmpeg_executable:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is required for audio normalization but was not found on PATH.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        output_path = temp.name

    cmd = [
        ffmpeg_executable,
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(NORMALIZED_SAMPLE_RATE),
        "-c:a",
        "libmp3lame",
        "-b:a",
        CHUNK_EXPORT_BITRATE,
        output_path,
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if completed.returncode != 0:
        try:
            os.remove(output_path)
        except OSError:
            pass
        stderr = completed.stderr.decode(errors="ignore").strip()
        stdout = completed.stdout.decode(errors="ignore").strip()
        detail = stderr or stdout or "Unknown FFmpeg error"
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {detail}")

    return output_path


def _trim_audio_to_duration(source_path: str, limit_ms: int) -> str:
    if limit_ms <= 0:
        raise HTTPException(status_code=400, detail="Duration limit must be greater than zero")

    audio = AudioSegment.from_file(source_path)
    if len(audio) <= limit_ms:
        return source_path

    trimmed = audio[:limit_ms]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        trimmed.export(temp.name, format="mp3", bitrate=CHUNK_EXPORT_BITRATE)
        return temp.name
