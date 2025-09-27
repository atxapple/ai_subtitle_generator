# AI Subtitle Generator

FastAPI web service that transcribes uploaded audio or video into translated English subtitles (SRT) using OpenAI Whisper models.

## Features
- Accepts audio (`audio/*`) and video (`video/*`) uploads via `/generate-subtitles`
- Normalises media locally to mono 16 kHz MP3 (128 kbps) before transcription
- Translates speech to English and returns timestamped SubRip (`.srt`) output
- Automatically chunks large sources to stay within OpenAI's 25 MiB limit
- Optional web UI at `/ui` with drag-and-drop support
- Provides `/healthz` endpoint for basic health checks

## Requirements
- Python 3.9+
- OpenAI API key with Whisper transcription access
- [FFmpeg](https://ffmpeg.org/) available on `PATH` (required by pydub for chunked processing)
- Python 3.13 users: the `audioop-lts` compatibility shim is bundled in `requirements.txt`; reinstall dependencies after pulling updates

## Installation
~~~bash
python -m venv .venv
. .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
~~~

Copy .env.example to .env and add your OpenAI key:
~~~bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
~~~

## Running the Server
~~~bash
uvicorn app.main:app --reload
~~~

The service listens on http://127.0.0.1:8000 by default.

## API Usage
### Generate Subtitles
POST /generate-subtitles

- Body: multipart form with field 'file' containing an audio or video file.
- Response: 200 OK with SRT text payload plus Content-Disposition header prompting download.

Example using curl (use `curl.exe` on Windows PowerShell to bypass the alias):
~~~bash
curl.exe -X POST   -F "file=@speech.mp3"   http://127.0.0.1:8000/generate-subtitles   -o subtitles.srt
~~~

Switch `speech.mp3` for a video file and add `?max_duration_minutes=3` when you only need the opening minutes:
~~~bash
curl.exe -X POST   -F "file=@talk.mp4"   "http://127.0.0.1:8000/generate-subtitles?max_duration_minutes=3"   -o subtitles.srt
~~~

Add `?max_duration_minutes=3` to trim processing to the first three minutes (helpful for quick smoke tests).

Large uploads are automatically split into multiple 128 kbps MP3 chunks to stay within OpenAI's 25 MiB limit, and the resulting subtitles are stitched back together. If the service cannot shrink a chunk under the limit it returns HTTP 413; lower the audio bitrate and retry.

### Health Check
GET /healthz → { "status": "ok" }

### Web UI
Visit http://127.0.0.1:8000/ui for a drag-and-drop interface that supports optional duration limits and automatic download of the resulting `.srt` file.

## Configuration
Adjust the default transcription model by setting OPENAI_MODEL in .env or environment variables. Defaults to whisper-1.

## Development Tips
- Use uvicorn app.main:app --reload for hot reload while developing.
- HTTP 502 responses include the OpenAI failure reason; surface or log them for easier troubleshooting.
