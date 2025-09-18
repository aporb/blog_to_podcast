**Product Requirements Document (PRD)**
**Blog → Podcast Agent (Cloud + Local)**

**Overview**
- Convert any blog/article URL into a short, engaging podcast episode via a Streamlit UI.
- Two execution modes:
  - Cloud Mode: OpenAI (LLM + optional TTS), Firecrawl (scraping), ElevenLabs (TTS).
  - Local Mode: Ollama (Qwen3-4B or similar) for summarization, Trafilatura for scraping, Piper/Coqui/macOS say for TTS.
- Output a playable and downloadable audio file (WAV, optionally MP3) and show the editable script.

**Goals**
- One-click conversion from URL to audio, consistently under ~1–3 minutes of narration.
- Reliable extraction and summarization with guardrails (max character limit) to prevent TTS failures.
- Local-first option with no external API keys; cloud option for higher quality voices and models.
- Lightweight, simple UI with sensible defaults and transparent errors.

**Non‑Goals**
- Full podcast production (music beds, multitrack mixing, hosting/publishing feeds).
- Multi-page crawling beyond a single article URL.
- Advanced SSML authoring and prosody tuning beyond light hints.

**Personas**
- Content Repurposer: Wants to quickly turn blog posts into short audio for distribution.
- Developer/Analyst: Needs a local, privacy-preserving option without cloud vendors.
- Casual User: Wants a simple tool; defaults just work.

**User Stories**
- As a user, I paste a blog URL and get a summarized audio I can play/download.
- As a user, I can edit the generated script before audio synthesis.
- As a user, I can choose between cloud/local engines and select model/voice.
- As a user, I get clear error messages if scraping fails or audio generation fails.
- As a user, I can re-run on the same URL without paying again (cache).

**Key Flows**
1) Input Keys/Settings → Paste URL → Generate → Play/Download.
2) Local mode: No keys; select Ollama model and TTS engine → Generate → Edit → TTS → Play/Download.
3) Optional: Preview/edit script between summarize and TTS.

**Functional Requirements**
- Input
  - Accept `http`/`https` URLs; validate format before running.
  - Optional model/voice selectors depending on mode.
- Scraping
  - Cloud: Firecrawl via tool or API. Local: Trafilatura fetch+extract with recall bias.
  - Provide clear errors when page blocks scraping or extraction fails.
- Summarization
  - Prompt designed for 1–2 minute conversational script with intro, key points, outro.
  - Enforce max character length (default 1900–2000 chars) post‑generation.
  - Expose the script as text in the UI for optional edits.
- TTS
  - Cloud: ElevenLabs or OpenAI TTS; select voice; output WAV.
  - Local: Piper by default; configurable voice model (.onnx path). Optionally Coqui XTTS/macOS say.
  - Fail gracefully with actionable errors (missing model, unsupported voice, etc.).
- Output
  - Save audio to `audio_generations/` with timestamp+id in filename.
  - Render player; offer download button.
  - Persist last script and audio path per URL in a simple cache (JSON or sqlite) to avoid rework.
- Telemetry (local only, opt-in)
  - Basic counters for runs, failures, average summary length (no content logging by default).

**Quality & Performance Targets**
- Summarization returns within ~10–20s on cloud; ~5–30s on local small models.
- TTS generation within ~5–20s for ~2k chars with Piper/ElevenLabs.
- Summary length clamped to configured limit; no TTS request over limit.

**Security & Privacy**
- Cloud keys loaded via Streamlit secrets or env. Do not persist keys to disk.
- Local mode should function without network once dependencies are installed.
- No article contents stored beyond ephemeral cache unless user enables caching.
- Avoid logging full article or script content in production logs by default.

**Error Handling**
- Invalid URL → inline warning with example format.
- Scrape fail → “Could not extract readable content. The site may block scraping.”
- LLM empty/too short → “Model output insufficient; try increasing max length or a larger model.”
- TTS fail → “No audio generated. Check TTS setup/voice/model path.”
- Add retry/backoff for networked operations (cloud mode).

**Dependencies**
- Cloud Mode
  - OpenAI (Responses/TTS), Firecrawl, ElevenLabs (optional if using OpenAI TTS instead).
- Local Mode
  - Ollama (e.g., `qwen3:4b`, recommend optional larger models for quality).
  - Trafilatura (HTML extraction), Piper (TTS) with voice `.onnx`.
  - Optional: ffmpeg for format conversions; Coqui XTTS or macOS `say`.

**Configuration**
- Mode toggle: Cloud | Local.
- Model pickers: OpenAI model id or Ollama model name.
- TTS engine dropdown: ElevenLabs | OpenAI TTS | Piper | Coqui | macOS say.
- Voice settings: ID (cloud) or model path (local) and speaking rate (if supported).
- Max summary characters: default 1900; adjustable 500–4000.

**UI/UX**
- Simple single-page Streamlit layout.
- Sidebar: mode, models, voices, character limit.
- Main: URL input, Generate button, spinner, editable script, TTS button (local flow), audio player, download.
- Consistent success/error messaging; disable buttons when prerequisites unmet.

**Data & Files**
- Cache directory: `.cache/blog_to_podcast/` with mapping `{url_hash: {article_digest, script, audio_path, ts}}`.
- Audio output: `audio_generations/podcast_{YYYYMMDD_HHMMSS}_{id8}.wav`.

**Acceptance Criteria**
- Given a valid URL, the app produces an audio file and playable widget in both Cloud and Local modes.
- The script shown to the user is ≤ configured character limit (enforced post‑generation).
- Invalid URL or missing dependencies yields clear, blocking error states with guidance.
- User can edit script and re‑synthesize locally without re-scraping/summarizing.
- Audio file is saved with timestamped filename and is downloadable.

**Testing Plan**
- Unit
  - URL validation helper.
  - Clamp function enforces max length.
  - File naming utility produces unique timestamped filenames.
- Integration
  - Trafilatura extraction on a few known articles.
  - Ollama summarization call mocked to return deterministic text.
  - Piper TTS mocked (simulate success/fail) and real if CI environment allows.
- E2E (manual)
  - Cloud and Local happy paths, TTS failure injection, scrape failure injection.

**Risks & Mitigations**
- Scrape blocking → fallback parser, user guidance, cache input text if user pastes.
- Weak summaries on tiny local models → recommend larger local model; expose length and temperature.
- TTS voice model availability (Piper) → link to curated model list; validate path pre-run.
- Vendor changes (cloud) → thin adapters to swap providers.

**Architecture Overview**
- Layers
  - UI: Streamlit components, state management.
  - Pipeline: scrape → summarize → script edit → tts → save/play.
  - Engines: adapters for Cloud vs Local (Strategy pattern).
  - Storage: output dir + lightweight cache.
- Module Interfaces
  - Scraper: `extract(url) -> str`
  - Summarizer: `summarize(article: str, max_chars: int) -> str`
  - TTS: `synthesize(text: str, out_path: str) -> Path`
  - Cache: `get(url) -> Optional[Record]`, `put(url, record)`

**Prompts (Baseline)**
- “Summarize the article in an engaging, conversational 1–2 minute podcast script. Max {max_chars} characters. Clear intro, key points, short outro. Avoid long quotes and fluff.”

**Open Questions**
- Do we need MP3 output in addition to WAV by default?
- Should we offer length target in words/time rather than characters?
- What languages must be supported initially (TTS voice availability)?

**Action Build Plan (for Coding Agent)**
- Repo Setup
  - Create `app.py`, `PRD.md`, `requirements.txt`, and `audio_generations/` dir.
  - Add `.streamlit/secrets.toml.example` documenting cloud keys.
- Utilities
  - Add `utils/validation.py` with `is_valid_url`, `clamp`.
  - Add `utils/io.py` with filename builder and safe read/write helpers.
  - Add `cache/simple_cache.py` (JSON-backed) with `get/put` by URL hash.
- Engines
  - Scraper: `engines/scrape_trafilatura.py` implementing `extract(url)`.
  - Summarizers: `engines/sum_ollama.py` (local), `engines/sum_openai.py` (cloud).
  - TTS: `engines/tts_piper.py` (local), `engines/tts_openai.py`, `engines/tts_elevenlabs.py`, `engines/tts_macsay.py`.
  - Engine registry/factory reading mode and settings from UI state.
- Streamlit UI
  - Sidebar: mode selector, model/voice selectors, char limit.
  - Main: URL input, Generate, editable script area, “Generate Audio” button, player, download.
  - Disable actions when prerequisites unmet; show spinners and precise errors.
- Pipeline Wiring
  - Implement `run_pipeline(url, mode, settings)` → `{script, audio_path}`.
  - Enforce character clamp post-summarization and pre‑TTS.
  - Cache article and script keyed by URL; reuse if present unless user forces refresh.
- Tests
  - Unit tests for validation, clamp, filename, cache.
  - Integration tests with mocked engines; optional real local tests behind flag.
- Docs
  - Update `README.md` with Local and Cloud setup, voices, troubleshooting.
  - Add a “Voices” section with Piper model links and paths.
- Packaging
  - `requirements.txt` for both modes; optional extras `[cloud]`, `[local]`.
  - Optional Dockerfile variants: local-only and cloud-enabled.

