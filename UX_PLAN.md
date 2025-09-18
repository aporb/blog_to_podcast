# Blog → Podcast (Streamlit) – UI/UX Improvement Plan

This document audits the current Streamlit dashboard and lays out a practical, no-regressions UX plan. Scope is presentation, layout, and guidance only — no functional changes.

## Objectives
- Reduce cognitive load and guide users from URL → Script → Audio.
- Make advanced controls discoverable but unobtrusive.
- Provide clear feedback, helpful defaults, and accurate expectations (length/time/limits).
- Keep everything within Streamlit; optional light CSS/theming, no JS.

## Current UX Audit (Snapshot)
- Information Architecture: Single page, many controls; some advanced options visible by default.
- Mode Awareness: Local vs Cloud settings coexist; can overwhelm first-time users.
- Length Controls: Powerful but dense (chars/tokens/words/time + WPM).
- Feedback: Good spinners; could add step-wise status and warnings (e.g., when approaching token/tts limits).
- Error UX: Helpful messages; could offer inline fixes and copyable diagnostics.
- Accessibility: Labels are clear; contrast and focus states depend on default theme.
- Performance: Noisy pages can be slow; chunking option is helpful but can be auto-suggested when needed.

## Recommendations (Non-breaking)

1) Layout & Navigation
- Use tabs to structure the main flow:
  - Tab 1: Script (URL input, generate, preview/edit, metrics)
  - Tab 2: Audio (TTS options, chunking, format, generate, player/download)
- Keep the sidebar for Mode, Length control, and “Advanced” sections using expanders.
- Enable wide layout and consistent section headers; reduce emoji to functional spots.

2) Progressive Disclosure
- Group advanced options under expanders:
  - Advanced Summarization: chunked map–reduce settings
  - Advanced TTS: chunking, pauses, voice-specific notes
  - Output Options: format/bitrate and ffmpeg status
- Show contextual hints only when the related option is enabled (e.g., gpt-4o-mini-tts limit).

3) Visual Hierarchy & Theming
- Adopt a cohesive theme via `.streamlit/config.toml` (primary color, base font size, light/dark parity).
- Add a minimal CSS shim for tighter spacing and clearer section dividers.
- Use consistent iconography and concise microcopy (e.g., “Generate Script”, “Generate Audio”).

4) Forms & Validation
- Wrap “Script” and “Audio” actions in `st.form` blocks to prevent partial submissions and to batch validation.
- Disable submission buttons until required fields are valid (URL or script present). Keep current validation logic.
- Add a “Reset” link/button to clear session state.

5) Length & Time Guidance
- Add presets: 1m, 2m, 3m buttons that set Words/Time and WPM.
- Always show real-time metrics (Words, ~Tokens, Est. duration) under the editor.
- Soft warnings when the script will exceed selected TTS per‑chunk limit.

6) Status & Progress
- Use `st.status` (or a styled placeholder) to display pipeline phases: Scrape → Summarize → Post-process → TTS → Stitch → Convert.
- Keep toasts minimal; rely on a status panel that persists until completion.

7) Content & Editing UX
- Add an expander to preview extracted article text (read-only) with word count.
- Add a “Use my own text” toggle to replace scraped content (fallback when scraping fails).
- Optional: quick actions “Trim to target words” and “Normalize punctuation” before TTS.

8) TTS Options & Output
- Only show TTS chunking controls when script length exceeds a threshold or when the user opts in.
- Add an optional “Pause between chunks (ms)” field (default 200–300ms) to improve pacing.
- In the download section, show predicted file sizes for MP3/M4A bitrates and an ffmpeg availability badge.

9) Accessibility & Internationalization
- Ensure all interactive controls have clear labels and helper text.
- Respect Streamlit’s base contrast; avoid color-only state signaling; include icons/text.
- Plan for multilingual support (voice availability note) and RTL-ready layout (text alignment switches with locale).

10) Diagnostics & Troubleshooting
- Add a collapsible “Diagnostics” panel with:
  - Environment checks: Ollama present, Piper voice exists, ffmpeg present, OPENAI_API_KEY set
  - Copy-to-clipboard of the last error/stack (sanitized)
- Link to README troubleshooting anchors.

## UI Sketch (Structural)
- Sidebar
  - Mode (Local/Cloud)
  - Length control (Chars/Tokens/Words/Time + WPM)
  - Expander: Advanced Summarization (chunk map–reduce)
  - Expander: Advanced TTS (chunking/pause)
  - Expander: Output Options (format/bitrate, ffmpeg badge)
- Main body
  - Tabs: Script | Audio
  - Script tab: URL → Generate Script (form) → Editor + Metrics → Article Preview (expander)
  - Audio tab: TTS controls (contextually minimal) → Generate Audio (form) → Status → Player + Downloads

## Theming & Styling (non-invasive)
- .streamlit/config.toml (example)

```
[theme]
primaryColor = "#4F46E5"
backgroundColor = "#0B0F19"
secondaryBackgroundColor = "#111827"
textColor = "#E5E7EB"
font = "Inter"
```

- Minimal CSS snippet in app (optional)

```
<style>
.small-note { color: #9CA3AF; font-size: 0.9rem; }
.section { padding-top: .25rem; margin-top: .5rem; border-top: 1px solid rgba(148,163,184,.15); }
.tight > div { padding-top: .25rem; padding-bottom: .25rem; }
</style>
```

## Templates & Components to Borrow
- Streamlit patterns
  - st.tabs for primary flow separation
  - st.expander for Advanced sections
  - st.form for grouped submit and atomic validation
  - st.status (or reusable placeholders) for step progress
  - st.toast for ephemeral success/soft warnings
- Community components (optional)
  - streamlit-extras (badges, keyboard shortcuts, card containers)
  - streamlit-aggrid (if you later add history lists)
  - streamlit-tags (for lightweight tag pills if needed)

## Copywriting & Microcopy
- Use instructive, specific labels: “Target duration (min)” + “Speaking rate (wpm)”.
- Contextual hints near controls: e.g., “gpt‑4o‑mini‑tts: 2000 token limit (auto‑clamped)”.
- Avoid technical jargon in primary flow; reserve it for expandable help.

## Accessibility Checklist
- Labels and helper text for all controls.
- Avoid color-only indicators; pair with icons or text.
- Reasonable target sizes (min 40px touch area) and spacing.
- Keyboard-only path for the main flow (forms + buttons).

## Non-Functional Enhancements (No Logic Changes)
- Cache banners: show when cached results are used (if/when caching added).
- Deterministic defaults: prefill fields based on mode with sensible choices.
- File naming consistency surfaced in UI and logs.

## Risks & Mitigations
- Too many controls visible → Use expanders and tabs; hide advanced by default.
- Confusion around length modes → Presets and inline duration estimates.
- Conversion failures (ffmpeg) → Badges and fallback notes near the format picker.

## Phased Action Plan (Checklist)

Phase 1 — Structure & Clarity
- [ ] Switch to tabs: Script | Audio
- [ ] Move advanced controls into expanders
- [ ] Add section headers and concise descriptions

Phase 2 — Guidance & Presets
- [ ] Add time/word presets (1m, 2m, 3m)
- [ ] Add contextual hints for selected modes/models
- [ ] Show ffmpeg availability badge and OPENAI key badge

Phase 3 — Status & Feedback
- [ ] Replace ad-hoc spinners with a step status panel
- [ ] Add soft warnings for near-limit TTS inputs
- [ ] Add Reset and Clear buttons for forms/session state

Phase 4 — Content & Editing Comfort
- [ ] Add article preview (expander) with word count
- [ ] Add “Use my own text” toggle to bypass scraping
- [ ] Optional: “Trim to target” helper before TTS

Phase 5 — Accessibility & Polish
- [ ] Theme via .streamlit/config.toml
- [ ] Minimal CSS for spacing/sections
- [ ] Audit labels/focus/contrast; adjust copy

Phase 6 — Diagnostics & Docs
- [ ] Diagnostics panel (env checks + copy error)
- [ ] Link to README troubleshooting anchors
- [ ] Record small UX notes in PRD/README

## Success Criteria
- Users can complete URL → Script → Audio with fewer visible choices initially.
- Advanced options remain accessible but don’t distract by default.
- Length and time expectations are clear; estimates match output within reasonable bounds.
- Errors are actionable and give a frictionless retry path.

---

If approved, I can start with Phase 1 (tabs + expanders + headers) in a non-breaking PR and iterate using this checklist.
