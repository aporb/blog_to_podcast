import os
import shutil
import subprocess
from uuid import uuid4
from datetime import datetime
from urllib.parse import urlparse
import re

import streamlit as st
import trafilatura

# Optional imports; guarded for environments where they aren't installed
try:
    import ollama  # Local LLM (Ollama)
except Exception:  # pragma: no cover
    ollama = None

try:
    from openai import OpenAI  # Cloud (OpenAI)
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import tiktoken  # Tokenizer for token-based length control
except Exception:  # pragma: no cover
    tiktoken = None


APP_TITLE = "📰 ➡️ 🎙️ Blog to Podcast"
AUDIO_DIR = "audio_generations"
os.makedirs(AUDIO_DIR, exist_ok=True)


# ----------------------- Helpers -----------------------
def is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def clamp(s: str, max_len: int = 2000) -> str:
    return (s or "")[:max_len]


def count_tokens(text: str, model_hint: str = "cl100k_base") -> int:
    if not text:
        return 0
    if tiktoken is None:
        # Rough fallback: ~1 token per 4 chars
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.get_encoding(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def clamp_tokens(text: str, max_tokens: int, model_hint: str = "cl100k_base") -> str:
    if not text:
        return ""
    if tiktoken is None:
        # Fallback: approximate by characters (4 chars ≈ 1 token)
        return clamp(text, max_tokens * 4)
    try:
        enc = tiktoken.get_encoding(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    ids = ids[:max_tokens]
    return enc.decode(ids)


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w[\w'\-]*\b", text))


def clamp_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = re.findall(r"\S+", text)
    return " ".join(words[: max(0, int(max_words))])


def words_to_tokens_est(words: int) -> int:
    return int(round(words / 0.75))


def tokens_to_words_est(tokens: int) -> int:
    return int(round(tokens * 0.75))


def words_to_chars_est(words: int) -> int:
    return int(round(words * 6))


def estimate_duration_seconds(words: int, wpm: int) -> int:
    if wpm <= 0:
        return 0
    return int(round((words / float(wpm)) * 60))


def ts_filename(prefix: str, ext: str = "wav") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = uuid4().hex[:8]
    return f"{AUDIO_DIR}/{prefix}_{ts}_{safe_id}.{ext}"


def fetch_and_clean(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_images=False,
        favor_recall=True,
        with_metadata=False,
    )
    return (text or "").strip()


def make_summary_prompt(article: str, constraints_lines: list[str]) -> str:
    header = (
        "You are a podcast script writer. Summarize the article in an engaging, "
        "conversational script.\n"
    )
    constraints = "Constraints:\n" + "\n".join(f"- {line}" for line in constraints_lines) + "\n\n"
    article_block = f"Article:\n\"\"\"{article}\"\"\"\n\n"
    tail = "Now write the script:"
    return header + constraints + article_block + tail


def summarize_with_ollama(model: str, prompt: str) -> str:
    if not ollama:
        raise RuntimeError("Ollama Python client not available. Install `ollama`." )
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    text = resp.get("message", {}).get("content", "").strip()
    return text


def summarize_with_openai(
    model: str,
    prompt: str,
    api_key: str = "",
    reasoning_effort: str = None,
    verbosity: str = None,
) -> str:
    if not OpenAI:
        raise RuntimeError("OpenAI SDK not available. Install `openai`." )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if verbosity:
        kwargs["text"] = {"verbosity": verbosity}
    r = client.responses.create(model=model, input=prompt, **kwargs)
    text = getattr(r, "output_text", None) or ""
    return text.strip()


def chunk_text_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int = 100) -> list[str]:
    if not text:
        return []
    if tiktoken is None:
        # Fallback: approximate characters
        approx_chars = chunk_tokens * 4
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + approx_chars])
            i += approx_chars
        return chunks
    enc = None
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        pass
    if enc is None:
        return [text]
    ids = enc.encode(text)
    chunks = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + chunk_tokens)
        chunk_ids = ids[start:end]
        chunks.append(enc.decode(chunk_ids))
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks


def summarize_hierarchical(
    summarize_fn,
    article: str,
    chunk_tokens: int = 8000,
    overlap_tokens: int = 200,
):
    # Map: summarize chunks
    chunks = chunk_text_by_tokens(article, chunk_tokens, overlap_tokens)
    if len(chunks) <= 1:
        return summarize_fn(article)
    partials = [summarize_fn(c) for c in chunks]
    # Reduce: combine partial summaries and summarize again
    combined = "\n\n".join(partials)
    return summarize_fn(combined)


def tts_with_piper(text: str, voice_path: str, wav_path: str) -> bool:
    if not os.path.isfile(voice_path):
        raise FileNotFoundError(f"Piper voice not found at {voice_path}")
    cmd = ["piper", "--model", voice_path, "--output_file", wav_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.communicate(input=(text or "").encode("utf-8"), timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        return False
    return proc.returncode == 0 and os.path.exists(wav_path)


def tts_with_mac_say(text: str, wav_path: str) -> bool:
    # macOS say outputs AIFF; convert to WAV if ffmpeg is present
    aiff_path = wav_path.replace(".wav", ".aiff")
    try:
        subprocess.check_call(["say", "-o", aiff_path, text])
        if shutil.which("ffmpeg"):
            subprocess.check_call(["ffmpeg", "-y", "-i", aiff_path, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(aiff_path)
        else:
            # Fall back: keep AIFF extension but move to WAV path
            os.rename(aiff_path, wav_path)
        return True
    except Exception:
        return False


def tts_with_openai(text: str, wav_path: str, model: str, voice: str, api_key: str) -> bool:
    if not OpenAI:
        raise RuntimeError("OpenAI SDK not available. Install `openai`.")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    def _write_bytes_to(path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    # Try preferred WAV output first; fall back gracefully if SDK rejects 'format'
    raw = None
    try:
        resp = client.audio.speech.create(model=model, voice=voice, input=text, format="wav")
        # Most recent SDKs return bytes directly
        if isinstance(resp, (bytes, bytearray)):
            raw = bytes(resp)
        elif hasattr(resp, "read"):
            raw = resp.read()
        else:
            raw = getattr(resp, "content", None)
        if not raw:
            raise RuntimeError("OpenAI TTS returned empty content")
        _write_bytes_to(wav_path, raw)
        return os.path.exists(wav_path)
    except TypeError as e:
        # Older/newer SDKs may not accept 'format'; retry without it (likely MP3)
        if "unexpected keyword argument 'format'" not in str(e):
            raise
        resp = client.audio.speech.create(model=model, voice=voice, input=text)
        if isinstance(resp, (bytes, bytearray)):
            raw = bytes(resp)
        elif hasattr(resp, "read"):
            raw = resp.read()
        else:
            raw = getattr(resp, "content", None)
        if not raw:
            raise RuntimeError("OpenAI TTS returned empty content")
        tmp_mp3 = wav_path[:-4] + ".mp3"
        _write_bytes_to(tmp_mp3, raw)
        # Convert to WAV if ffmpeg is available; else rename with .wav extension
        if shutil.which("ffmpeg"):
            try:
                subprocess.check_call(["ffmpeg", "-y", "-i", tmp_mp3, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(tmp_mp3)
                return os.path.exists(wav_path)
            except Exception:
                # fallback to leaving MP3 renamed as WAV
                os.rename(tmp_mp3, wav_path)
                return os.path.exists(wav_path)
        else:
            os.rename(tmp_mp3, wav_path)
            return os.path.exists(wav_path)


# ----------------------- TTS Chunking Utilities -----------------------
def split_text_by_sentences(text: str) -> list[str]:
    import re
    # Split on sentence enders while keeping delimiters
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def split_text_for_tts(text: str, units: str, limit: int, model_hint: str = "cl100k_base") -> list[str]:
    # Prefer sentence-aware grouping under the limit
    if units == "Tokens":
        sentences = split_text_by_sentences(text)
        chunks: list[str] = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if count_tokens(candidate, model_hint=model_hint) <= limit:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur)
                    cur = s
                else:
                    # Single sentence longer than limit: hard split by tokens
                    if tiktoken is None:
                        chunks.append(clamp(s, limit * 4))
                    else:
                        enc = tiktoken.get_encoding("cl100k_base")
                        ids = enc.encode(s)
                        for start in range(0, len(ids), limit):
                            chunks.append(enc.decode(ids[start:start+limit]))
                        cur = ""
        if cur:
            chunks.append(cur)
        return chunks
    else:
        # Characters
        sentences = split_text_by_sentences(text)
        chunks: list[str] = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) <= limit:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur)
                    cur = s
                else:
                    # Hard split by chars
                    for i in range(0, len(s), limit):
                        chunks.append(s[i:i+limit])
                    cur = ""
        if cur:
            chunks.append(cur)
        return chunks


def stitch_wavs_ffmpeg(chunk_paths: list[str], output_path: str) -> bool:
    # Use concat demuxer with re-encode to a common PCM format to avoid mismatches
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in chunk_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:a", "pcm_s16le",
            "-ar", "22050",
            output_path,
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception:
        return False
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


def stitch_wavs_naive(chunk_paths: list[str], output_path: str) -> bool:
    import wave
    try:
        with wave.open(chunk_paths[0], 'rb') as w0:
            params = w0.getparams()
            frames = [w0.readframes(w0.getnframes())]
        for p in chunk_paths[1:]:
            with wave.open(p, 'rb') as wi:
                if wi.getparams() != params:
                    return False
                frames.append(wi.readframes(wi.getnframes()))
        with wave.open(output_path, 'wb') as wo:
            wo.setparams(params)
            for fr in frames:
                wo.writeframes(fr)
        return os.path.exists(output_path)
    except Exception:
        return False


def convert_audio_ffmpeg(src_path: str, dst_path: str, codec: str, bitrate_kbps: int | None = None) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    args = ["ffmpeg", "-y", "-i", src_path]
    if codec:
        args += ["-c:a", codec]
    if bitrate_kbps:
        args += ["-b:a", f"{int(bitrate_kbps)}k"]
    args.append(dst_path)
    try:
        subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(dst_path)
    except Exception:
        return False


# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title=f"{APP_TITLE} (Local/Cloud)", page_icon="🎙️")
st.title(f"{APP_TITLE} (Local / Cloud)")

with st.sidebar:
    st.header("Mode & Settings")
    mode = st.selectbox("Execution mode", ["Local (Ollama)", "Cloud (OpenAI)"])

    # Length control
    st.markdown("**Length control**")
    length_mode = st.radio("Length control", ["Characters", "Tokens", "Words", "Time"], index=0, key="length_mode")
    target_words = None
    target_minutes = None
    wpm = None
    if length_mode == "Characters":
        max_chars = st.slider("Max summary length (chars)", 500, 20000, 1900, step=100, key="max_chars")
        max_tokens = None
    elif length_mode == "Tokens":
        max_tokens = st.number_input("Max summary length (tokens)", min_value=200, max_value=200000, value=4000, step=200, key="max_tokens")
        max_chars = None
    elif length_mode == "Words":
        target_words = st.number_input("Target words", min_value=50, max_value=20000, value=350, step=25, key="target_words")
        st.caption(f"≈ {estimate_duration_seconds(int(target_words), 160)//60} min at 160 wpm")
        max_chars = None
        max_tokens = None
    else:
        target_minutes = st.number_input("Target duration (minutes)", min_value=0.5, max_value=60.0, value=2.0, step=0.5, key="target_minutes")
        wpm = st.slider("Speaking rate (wpm)", 100, 220, 160, step=10, key="wpm")
        target_words = int(round((st.session_state.get("target_minutes", 2.0)) * (st.session_state.get("wpm", 160))))
        st.caption(f"Target words ≈ {target_words}")
        max_chars = None
        max_tokens = None

    # Presets removed per request

    # Advanced Summarization
    with st.expander("Advanced Summarization", expanded=False):
        chunking = st.checkbox("Use chunked (map–reduce) summarization for long articles", value=False)
        if chunking:
            chunk_tokens = st.number_input("Chunk size (tokens)", min_value=2000, max_value=200000, value=8000, step=1000)
            overlap_tokens = st.number_input("Overlap (tokens)", min_value=0, max_value=5000, value=200, step=50)
        else:
            chunk_tokens = None
            overlap_tokens = None

    # Engine-specific settings
    if mode.startswith("Local"):
        with st.expander("Local Engine", expanded=True):
            ollama_model = st.text_input("Ollama model", value="qwen3:4b")
            tts_engine = st.selectbox("TTS engine", ["piper", "mac_say", "none"], index=0)
            piper_voice_path = st.text_input(
                "Piper voice (.onnx)",
                value=os.path.expanduser("~/.local/share/piper-voices/en_US-amy-medium.onnx"),
            )
            if ollama is None:
                st.warning("Python client for Ollama not installed. `pip install ollama`.")
    else:
        with st.expander("Cloud Engine (OpenAI)", expanded=True):
            openai_model = st.selectbox("OpenAI model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini"], index=0)
            reasoning_effort = st.selectbox("Reasoning effort", ["minimal", "low", "medium", "high"], index=2)
            verbosity = st.selectbox("Verbosity", ["low", "medium", "high"], index=1)
            tts_model = st.selectbox("OpenAI TTS model", ["gpt-4o-mini-tts", "tts-1"], index=0)
            voice = st.selectbox("Voice", ["alloy", "verse", "aria", "sage"], index=0)
            openai_key_present = bool(os.environ.get("OPENAI_API_KEY"))
            st.caption("OPENAI_API_KEY " + ("✅ detected" if openai_key_present else "❌ not set"))
            if tts_model == "gpt-4o-mini-tts":
                st.caption("Note: gpt-4o-mini-tts has a 2000-token input limit for TTS.")

    # Advanced TTS
    with st.expander("Advanced TTS", expanded=False):
        tts_chunking = st.checkbox("Split long scripts for TTS and stitch", value=False)
        if tts_chunking:
            tts_units = st.radio("Chunk by", ["Tokens", "Characters"], index=0, horizontal=True)
            if tts_units == "Tokens":
                default_tok = 1800
                tts_limit = st.number_input("Per-chunk token limit", min_value=500, max_value=20000, value=default_tok, step=100)
            else:
                tts_limit = st.number_input("Per-chunk character limit", min_value=500, max_value=20000, value=4000, step=100)
            tts_pause_ms = st.number_input("Pause between chunks (ms)", min_value=0, max_value=2000, value=250, step=50)
        else:
            tts_units = "Tokens"
            tts_limit = None
            tts_pause_ms = 0

    # Output Options
    with st.expander("Output Options", expanded=False):
        dl_format = st.selectbox("Preferred format", ["WAV (no compression)", "MP3 (smaller)", "M4A/AAC (smaller)"], index=0)
        if dl_format != "WAV (no compression)":
            bitrate_kbps = st.slider("Bitrate (kbps)", min_value=64, max_value=192, value=128, step=16)
        else:
            bitrate_kbps = None
        ffmpeg_ok = bool(shutil.which("ffmpeg"))
        st.caption("ffmpeg " + ("✅ available" if ffmpeg_ok else "❌ not found (required for MP3/M4A conversion)"))


tab_script, tab_audio = st.tabs(["📝 Script", "🎧 Audio"]) 

with tab_script:
    with st.form("script_form"):
        url = st.text_input("Enter the Blog URL:", "")
        use_own = st.checkbox("Use my own text instead of scraping")
        own_text = ""
        if use_own:
            own_text = st.text_area("Paste your article text", height=180)
        gen_script_btn = st.form_submit_button("Generate Script", type="primary")

with tab_audio:
    with st.form("audio_form"):
        gen_audio_btn = st.form_submit_button("Generate Audio")

# Session state for script text and last audio path
if "script" not in st.session_state:
    st.session_state.script = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


def ensure_valid_url_or_stop():
    if not is_valid_url(url.strip()):
        st.warning("Please enter a valid blog URL. Example: https://example.com/article")
        st.stop()


# ----------------------- Actions -----------------------
if gen_script_btn:
    if not use_own:
        ensure_valid_url_or_stop()
    with st.spinner("Scraping and summarizing..."):
        article = own_text.strip() if use_own else fetch_and_clean(url)
        if not article:
            st.error("No input text available. Either paste your own text or use a valid URL.")
            st.stop()

        try:
            constraints_lines = [
                "Clear structure with intro, key points, and a short outro",
                "No long quotes or marketing fluff",
            ]
            if length_mode == "Characters" and max_chars:
                constraints_lines.insert(0, f"Max {int(max_chars)} characters")
            elif length_mode == "Tokens" and max_tokens:
                constraints_lines.insert(0, f"Max {int(max_tokens)} tokens")
            elif length_mode == "Words" and target_words:
                constraints_lines.insert(0, f"Target about {int(target_words)} words (±10%)")
            else:
                constraints_lines.insert(0, f"Target about {int(target_words)} words (~{target_minutes} min at {wpm} wpm)")

            def make_fn(engine: str):
                if engine == "local":
                    return lambda text: summarize_with_ollama(
                        ollama_model, make_summary_prompt(text[:180000], constraints_lines)
                    )
                else:
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                    return lambda text: summarize_with_openai(
                        openai_model,
                        make_summary_prompt(text[:180000], constraints_lines),
                        api_key=api_key,
                        reasoning_effort=reasoning_effort,
                        verbosity=verbosity,
                    )

            summarize_fn = make_fn("local" if mode.startswith("Local") else "cloud")

            if chunking and chunk_tokens:
                script = summarize_hierarchical(
                    summarize_fn,
                    article,
                    chunk_tokens=int(chunk_tokens),
                    overlap_tokens=int(overlap_tokens or 0),
                )
            else:
                script = summarize_fn(article)
        except Exception as e:
            st.error(f"Summarization error: {e}")
            st.stop()

        if not script or len(script) < 80:
            st.error("Model produced an empty or too short script. Try increasing max length or a larger model.")
            st.stop()

        if length_mode == "Tokens" and max_tokens:
            st.session_state.script = clamp_tokens(script, int(max_tokens))
        elif length_mode == "Characters" and max_chars:
            st.session_state.script = clamp(script, int(max_chars))
        elif length_mode in ("Words", "Time") and target_words:
            st.session_state.script = clamp_words(script, int(target_words))
        else:
            st.session_state.script = clamp(script, 2000)

        # Article preview
        if not use_own:
            with st.expander("Preview extracted article (read-only)"):
                st.caption(f"Words: {count_words(article)} • ~Tokens: {count_tokens(article)}")
                st.text_area("Extracted article", value=article[:10000], height=200, disabled=True)


with tab_script:
    if st.session_state.script:
        st.subheader("Editable Script")
        st.session_state.script = st.text_area(
            "You can tweak the script before TTS",
            value=st.session_state.script,
            height=260,
        )
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Words", count_words(st.session_state.script))
        with metrics_col2:
            approx_tokens = count_tokens(st.session_state.script)
            st.metric("~Tokens", approx_tokens)
        with metrics_col3:
            display_wpm = (st.session_state.get("wpm") or wpm or 160)
            secs = estimate_duration_seconds(count_words(st.session_state.script), int(display_wpm))
            st.metric("Est. duration", f"{secs//60}:{secs%60:02d} @ {display_wpm} wpm")
        cc1, cc2 = st.columns([1,1])
        with cc1:
            if st.button("Reset Script"):
                st.session_state.script = ""
                st.session_state.audio_path = None
                st.experimental_rerun()
        with cc2:
            if st.button("Clear Audio"):
                st.session_state.audio_path = None


if gen_audio_btn:
    if not st.session_state.script:
        st.warning("Generate a script first.")
        st.stop()

    wav_path = ts_filename("podcast", ext="wav")
    if length_mode == "Tokens":
        text = clamp_tokens(st.session_state.script, int(max_tokens or 4000))
    else:
        text = clamp(st.session_state.script, int(max_chars or 2000))

    # Soft warning for TTS token limits
    if (not tts_chunking) and (not mode.startswith("Local")) and (tts_model == "gpt-4o-mini-tts"):
        current_tokens = count_tokens(text)
        if current_tokens > 1800:
            st.info(f"Your script is ~{current_tokens} tokens. gpt-4o-mini-tts max is 2000 tokens. Consider enabling TTS chunking or reducing length.")

    # Status panel during TTS
    with st.status("Synthesizing speech…", expanded=True) as status:
        ok = False
        try:
            status.update(label="Preparing chunks" if tts_chunking else "Generating audio", state="running")
            if tts_chunking:
                # Prepare chunk directory
                chunk_dir = os.path.join(AUDIO_DIR, f"chunks_{uuid4().hex[:8]}")
                os.makedirs(chunk_dir, exist_ok=True)
                # Split text
                chunks = split_text_for_tts(text, units=(tts_units if tts_chunking else ("Tokens" if length_mode == "Tokens" else "Characters")), limit=int(tts_limit if tts_chunking else (max_tokens or max_chars or 2000)))
                chunk_paths: list[str] = []
                # Render each chunk
                for i, chunk in enumerate(chunks, start=1):
                    status.update(label=f"Rendering chunk {i}/{len(chunks)}", state="running")
                    cpath = os.path.join(chunk_dir, f"chunk_{i:03d}.wav")
                    if mode.startswith("Local"):
                        if 'tts_engine' not in locals():
                            tts_engine = "piper"
                        if tts_engine == "piper":
                            okc = tts_with_piper(chunk, piper_voice_path, cpath)
                        elif tts_engine == "mac_say":
                            okc = tts_with_mac_say(chunk, cpath)
                        else:
                            okc = False
                    else:
                        # Enforce gpt-4o-mini-tts limit per chunk
                        chunk_text = chunk
                        if tts_model == "gpt-4o-mini-tts":
                            tcount = count_tokens(chunk_text)
                            if tcount > 2000:
                                chunk_text = clamp_tokens(chunk_text, 2000)
                        api_key = os.environ.get("OPENAI_API_KEY", "")
                        okc = tts_with_openai(chunk_text, cpath, model=tts_model, voice=voice, api_key=api_key)
                    if not okc:
                        raise RuntimeError(f"TTS failed on chunk {i}")
                    chunk_paths.append(cpath)
                    # Optional pause between chunks (silence handling deferred to stitch)
                # Stitch
                status.update(label="Stitching audio", state="running")
                if shutil.which("ffmpeg"):
                    ok = stitch_wavs_ffmpeg(chunk_paths, wav_path)
                else:
                    ok = stitch_wavs_naive(chunk_paths, wav_path)
            else:
                status.update(label="Generating audio", state="running")
                if mode.startswith("Local"):
                    if 'tts_engine' not in locals():
                        tts_engine = "piper"
                    if tts_engine == "piper":
                        ok = tts_with_piper(text, piper_voice_path, wav_path)
                    elif tts_engine == "mac_say":
                        ok = tts_with_mac_say(text, wav_path)
                    else:
                        st.info("TTS disabled. You can copy the script above.")
                else:
                    # Enforce gpt-4o-mini-tts 2000-token input limit
                    if tts_model == "gpt-4o-mini-tts":
                        tcount = count_tokens(text)
                        if tcount > 2000:
                            text = clamp_tokens(text, 2000)
                            st.info("Clamped TTS input to 2000 tokens for gpt-4o-mini-tts.")
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                    ok = tts_with_openai(text, wav_path, model=tts_model, voice=voice, api_key=api_key)
        except Exception as e:
            st.error(f"TTS error: {e}")
            status.update(label="TTS failed", state="error")
        else:
            status.update(label="TTS complete" if ok else "No audio generated", state=("complete" if ok else "error"))

    if ok and os.path.exists(wav_path):
        st.success("Audio generated.")
        st.session_state.audio_path = wav_path
    elif mode.startswith("Local") and (tts_engine == "none"):
        st.info("TTS skipped. Script is ready above.")
    else:
        st.error("No audio generated. Check TTS setup and settings.")


with tab_audio:
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        primary_path = st.session_state.audio_path
        primary_mime = "audio/wav"
        # Convert to user-selected format if needed
        if 'dl_format' in locals() and dl_format != "WAV (no compression)":
            base, _ = os.path.splitext(primary_path)
            if dl_format.startswith("MP3"):
                dst = base + ".mp3"
                ok_conv = convert_audio_ffmpeg(primary_path, dst, codec="libmp3lame", bitrate_kbps=bitrate_kbps)
                if ok_conv:
                    primary_path = dst
                    primary_mime = "audio/mpeg"
                else:
                    st.info("MP3 conversion unavailable (ffmpeg missing?). Serving WAV instead.")
            elif dl_format.startswith("M4A"):
                dst = base + ".m4a"
                ok_conv = convert_audio_ffmpeg(primary_path, dst, codec="aac", bitrate_kbps=bitrate_kbps)
                if ok_conv:
                    primary_path = dst
                    primary_mime = "audio/mp4"
                else:
                    st.info("M4A/AAC conversion unavailable (ffmpeg missing?). Serving WAV instead.")

        audio_bytes = open(primary_path, "rb").read()
        st.audio(audio_bytes, format=primary_mime)
        st.download_button(
            "Download Podcast",
            audio_bytes,
            file_name=os.path.basename(primary_path),
            mime=primary_mime,
        )
        # Offer original WAV as secondary download if different
        if primary_path != st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            wav_bytes = open(st.session_state.audio_path, "rb").read()
            st.download_button(
                "Download Original WAV",
                wav_bytes,
                file_name=os.path.basename(st.session_state.audio_path),
                mime="audio/wav",
            )


st.caption(
    "Notes: Local mode uses Ollama for summarization and Piper/macOS say for TTS. "
    "Cloud mode uses OpenAI (GPT-4o/GPT-5 family) for summarization and TTS; "
    "scraping is performed locally via Trafilatura. Length can be constrained by characters or tokens. "
    "Optionally split TTS into chunks and stitch for longer scripts or strict TTS limits."
)
