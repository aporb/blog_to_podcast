# Blog → Podcast Agent (Local + Cloud)

Convert any blog/article URL into a short, engaging podcast you can play and download — all from a simple Streamlit app.

- Local mode: Runs fully on your machine using Ollama (Qwen) for summarization and Piper or macOS say for TTS.
- Cloud mode: Uses OpenAI for both summarization and text-to-speech. Scraping is done locally.

## Features
- Paste a URL → scrape → summarize → edit → synthesize → play/download WAV
- Local mode (privacy-first): Ollama + Piper/macOS say, no API keys required
- Cloud mode (quality/variety): OpenAI Responses API + OpenAI TTS
- Character limit clamp to keep TTS within safe bounds
- Length controls in characters or tokens (via `tiktoken`)
- Target by words or time (minutes): choose Words or Time, and the app will guide the LLM to that target and clamp the output accordingly. It also shows words, ~tokens, and estimated duration at your selected WPM.
- Timestamped filenames saved to `audio_generations/`
- Clear error messages for bad URLs, scraping failures, or TTS issues
- Optional TTS chunking: split long scripts by tokens/chars and stitch into a single WAV (uses `ffmpeg` when available)
- Download format options: WAV, MP3, or M4A/AAC (requires `ffmpeg` for conversion)

## Screenshots
- Streamlit single-page UI with sidebar controls (mode, models, voices, length)

## Architecture
- UI: Streamlit single page (`app.py`)
- Scraper: Trafilatura (local) extracts article text
- Summarizers:
  - Local: Ollama model (default `qwen3:4b`)
  - Cloud: OpenAI (e.g., `gpt-5`, `gpt-5-mini`, `gpt-4o`, `gpt-4o-mini`) with optional reasoning effort and verbosity controls
- TTS:
  - Local: Piper (recommended) or macOS `say`
  - Cloud: OpenAI TTS (`gpt-4o-mini-tts` or `tts-1`)

## Requirements
- Python 3.9+
- Streamlit and Python deps: `pip install -r requirements.txt`
- Trafilatura (installed via requirements) for scraping
- For Local mode:
  - Ollama installed, with a pulled model (e.g., `qwen3:4b`)
  - One of:
    - Piper (plus a voice .onnx model file)
    - macOS `say` (built into macOS)
  - Recommended: `ffmpeg` for audio conversions (macOS: `brew install ffmpeg`)
- For Cloud mode:
  - OpenAI API key set as `OPENAI_API_KEY`

## Installation
1) Clone this repo and install Python dependencies:
```
pip install -r requirements.txt
```

2) Optional: Install system tools depending on mode.
- Ollama (macOS via Homebrew):
```
brew install ollama
ollama pull qwen3:4b
```
- Piper (macOS via Homebrew):
```
brew install piper
# Download a voice model (example voice):
mkdir -p ~/.local/share/piper-voices
cd ~/.local/share/piper-voices
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-amy-medium.onnx
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-amy-medium.onnx.json
```
- ffmpeg (recommended):
```
brew install ffmpeg
```

3) Cloud mode (optional): Set your OpenAI key in the environment:
```
export OPENAI_API_KEY=sk-...
```
Alternatively, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key there.

## Run
```
streamlit run app.py
```
Then open the local URL shown by Streamlit (usually http://localhost:8501).

## Usage
1) Choose mode in the sidebar:
   - Local (Ollama): set the Ollama model (default `qwen3:4b`), choose TTS engine (Piper/macOS say), and set Piper voice path if using Piper.
   - Cloud (OpenAI): choose the OpenAI model (supports GPT‑5 family) and TTS voice. You can also set reasoning effort and verbosity. Ensure `OPENAI_API_KEY` is set.
2) Paste a blog/article URL into the input field.
3) Choose length control:
   - Characters or Tokens: hard caps with clamping.
   - Words: pick a target word count (the app shows the approximate duration at 160 wpm).
   - Time: choose minutes and speaking rate (wpm); the app converts to an approximate word target.
4) Click “📝 Generate Script” to scrape and summarize.
5) Optionally edit the script in the text area. You’ll see live metrics (Words, ~Tokens, Estimated Duration).
6) Click “🎧 Generate Audio” to synthesize and play/download the WAV.

## Configuration (Sidebar)
- Execution mode: Local (Ollama) or Cloud (OpenAI)
- Length mode: Characters or Tokens
- Max summary length: Clamp in characters (default 1900) or tokens (default 4000)
- Local mode options:
  - Ollama model (e.g., `qwen3:4b`, `llama3.1:8b`, etc.)
  - TTS engine: `piper`, `mac_say`, or `none`
  - Piper voice path (path to `.onnx` voice file)
- Cloud mode options:
  - OpenAI model (e.g., `gpt-5`, `gpt-5-mini`, `gpt-4o`, `gpt-4o-mini`)
  - Reasoning effort (minimal/low/medium/high) and text verbosity (low/medium/high)
  - TTS model (e.g., `gpt-4o-mini-tts`, `tts-1`)
    - Note: `gpt-4o-mini-tts` accepts up to 2000 input tokens. The app automatically clamps to this limit.
  - Voice (`alloy`, `verse`, `aria`, `sage`)

### TTS Chunking & Stitching
- Enable “Split long scripts for TTS and stitch” in the sidebar to render long scripts reliably.
- Choose chunk units (Tokens or Characters) and a per‑chunk limit.
- The app renders each chunk to WAV, then stitches them:
  - If `ffmpeg` is installed, it uses the concat demuxer and re‑encodes to a common PCM format for robustness.
  - Without `ffmpeg`, it attempts a basic WAV concatenation (requires matching audio params across chunks).

### Long articles and chunking
- Toggle “Use chunked (map–reduce) summarization” to handle very long articles.
- Configure chunk size/overlap (in tokens). Each chunk is summarized, then the partials are combined and summarized again into the final script.

## File Structure
```
.
├── app.py                        # Streamlit app
├── requirements.txt              # Python dependencies
├── PRD.md                        # Product requirements document
├── audio_generations/            # Output WAV files (created at runtime)
└── .streamlit/
    └── secrets.toml.example      # Example secrets file for OpenAI key
```

## Notes & Tips
- Scraping is always local via Trafilatura. Some sites may block scraping or return little readable content.
- Local model quality varies; if the summary is weak or too short, try a larger Ollama model (e.g., 7B/8B).
- Character vs token limits: Token mode uses `tiktoken` (falls back to a rough estimate if unavailable). GPT‑5 supports large context; use chunking only if needed or if you hit context limits.
- Piper requires a valid voice `.onnx` file; ensure the path in the sidebar points to your installed voice.
- macOS `say` is quick to try, but voice quality is basic. `ffmpeg` is used to convert AIFF to WAV automatically if present.
- Generated audio is saved under `audio_generations/` with a timestamped filename. You can choose a smaller MP3 or M4A download; the app will transcode from WAV using `ffmpeg`.

## Troubleshooting
- “Please enter a valid blog URL” → Ensure the URL starts with http/https and is reachable.
- “Could not extract readable content” → The page may block scraping or isn’t an article-like page.
- “Model output insufficient” → Increase max characters, choose a larger model, or try cloud mode.
- Piper errors → Verify the `piper` binary is installed and the voice path exists; try a different voice.
- OpenAI errors → Ensure `OPENAI_API_KEY` is set; check selected model names are available to your account.

## Security & Privacy
- Local mode runs entirely on-device once dependencies are installed.
- Cloud mode sends the article-derived text and script to OpenAI for processing.
- API keys are read from environment variables or Streamlit secrets and are not written to disk by the app.

## Roadmap (Potential Enhancements)
- MP3 output option and loudness normalization (pydub/ffmpeg)
- Simple caching per URL to avoid re-scraping/re-summarizing
- Additional TTS engines (e.g., Coqui XTTS) and voice controls
- Batch mode for multiple URLs (CSV)
- Dockerfiles for local-only and cloud-enabled variants

## Acknowledgements
- Trafilatura for robust web text extraction
- Ollama for local LLM orchestration
- Piper for fast, high-quality local TTS
- OpenAI for summarization and TTS in cloud mode

---

If you run into issues or want specific enhancements (e.g., MP3 output, caching, or Docker), open an issue or request and we’ll add the pieces.
