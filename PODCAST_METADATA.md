# Optional Podcast & Episode Metadata (for Generation and Distribution)

This guide lists optional metadata you can collect from users and embed into the audio files and feed outputs. It groups fields by scope (Podcast-level, Episode-level) and by standards (RSS/iTunes, Podcasting 2.0, Audio Tags). Use this as a checklist when extending the Streamlit UI.

## Why capture metadata?
- Improve listener experience in players (titles, artwork, chapters, links).
- Enable podcast directories and apps to classify and discover your show.
- Support accessibility and compliance (transcripts, content warnings).
- Track contributors and credits; automate intros/outros and sponsor reads.

## Podcast-level (show) metadata
- Core identity
  - Show title
  - Author/owner name
  - Owner email/contact
  - Show website URL
  - Primary language (e.g., en, en-US)
  - Copyright / license (e.g., CC-BY, all rights reserved)
- Presentation
  - Show description (short blurb + long description)
  - Category and subcategory (iTunes categories)
  - Explicit content flag (yes/no/clean)
  - Show artwork (upload or URL, recommended 3000x3000 JPEG/PNG)
- Social & links
  - Homepage, Twitter/X, LinkedIn, YouTube, Mastodon links
  - Funding/sponsor link (Podcasting 2.0: podcast:funding)
- Contributors (Podcasting 2.0)
  - podcast:person entries (name, role, email/url, group)
- Value/Monetization (Podcasting 2.0)
  - podcast:value (if you support value-for-value)
- Blocking & targeting
  - podcast:block (hide from iTunes)
  - podcast:location (optional regional targeting)

## Episode-level metadata
- Identity & scheduling
  - Episode title
  - Subtitle (short tagline)
  - Description / show notes (HTML allowed in many apps)
  - Publish date/time (timezone aware)
  - Season number (int)
  - Episode number (int)
  - Episode type (full, trailer, bonus)
  - GUID (stable unique id)
  - Episode artwork (upload/URL; overrides show art)
  - Keywords / tags (comma‑separated)
- Classification & compliance
  - Explicit flag (yes/no/clean)
  - Content warnings (brief text)
  - Language override (if different from show)
  - License override
- Structure & navigation
  - Chapters (Podcasting 2.0: podcast:chapters; JSON or MP3 ID3+CHAP)
    - Start time, title, URL per chapter
  - Transcript (Podcasting 2.0: podcast:transcript)
    - URL to VTT/SRT/JSON + type, language, rel="captions"
  - Soundbites/Clips (Podcasting 2.0: podcast:soundbite)
- People & credits
  - Hosts, guests (podcast:person for each; roles like host/guest/producer)
  - Guest bios and links
  - Music credits (artist, track, license)
  - Sponsor/affiliate disclosures
- Calls to action
  - Primary CTA copy + link (subscribe, newsletter, product)
  - Secondary links (article source, references)
- Technical
  - Duration (auto-computed; allow manual override)
  - Enclosure MIME (audio/mpeg, audio/mp4, audio/x-m4a, audio/wav)
  - Bitrate and sample rate (auto from file)

## Audio file tagging (on export)
- MP3 (ID3v2)
  - TIT2 (Title), TALB (Album/Show), TPE1 (Artist/Author), TDRC (Year/Date)
  - TRCK (Track/Episode), TCON (Genre), COMM (Comments/Show notes)
  - APIC (Artwork), TSSE (Encoder)
  - iTunes custom frames for podcast fields (e.g., TCAT category)
- M4A/AAC (iTunes-style atoms)
  - ©nam (Title), ©ART (Artist), ©alb (Album), ©day (Date)
  - covr (Cover art), stik (Media type: 10=Podcast), trkn (Track)
  - Custom podcast atoms for episode type, season/episode numbers
- WAV (BWF/INFO)
  - INFO tags (INAM, IART, ICMT), BWF bext chunk for metadata

## Generation-time inputs (feed the model)
- Voice & style
  - Host persona (friendly/analytical/energetic), formality, humor level
  - Pace & emphasis (short sentences, rhetorical questions)
  - Pronunciation/lexicon notes (brand names, people)
- Audience
  - Target audience profile (beginner/pro, age range, region/accent)
  - Prior knowledge assumptions
- Content controls
  - Focus areas (tech, business impact, practical tips)
  - Strictness on citations/quotes and disclaimers
  - CTA (copy + link) and placement (intro/outro/midroll)
- Audio polish (post‑processing plan)
  - Intro/outro music selection (file/upload or stock), volume, fade-in/out
  - Optional midroll slot(s) and sponsor script
  - Silence trimming, normalization target (e.g., −16 LUFS), limiter

## Streamlit UI proposal (non-breaking additions)
- Sidebar → “Podcast Metadata” (expander)
  - Title, Author, Email, Website, Language, Category, Explicit, Show art upload/URL
- Sidebar → “Episode Metadata” (expander)
  - Title, Subtitle, Description (textarea), Episode art, Season, Episode #, Type, Tags
  - Content warnings, Transcript URL, Chapter JSON upload (optional)
  - CTA text + link, Sponsor block (textarea)
- Script Tab
  - “Host persona”, “Tone”, “Audience” selectors to steer the generation
  - Optional “Insert CTA at outro” toggle
- Audio Tab
  - Music bed upload (intro/outro) with level sliders and fade checkboxes
  - Normalize loudness toggle (ffmpeg sox filters) [optional]
- Output
  - After synthesis, write tags for MP3/M4A (if chosen) using mutagen or ffmpeg metadata flags
  - Generate an RSS item XML snippet with the chosen metadata as a copyable block

## Implementation tips
- Keep all new fields optional; default to current behavior if blank.
- Guard audio tagging and XML generation behind toggles to avoid breaking flows.
- Prefer Mutagen for ID3/M4A tagging in Python; fall back to ffmpeg `-metadata` on conversion.
- Validate URLs and file types; show small previews for artwork.
- Store user inputs in `st.session_state` to persist across tabs.

---

Use this checklist to prioritize which fields you want first (e.g., Episode title, artwork, CTA, transcript URL), then expand toward Podcasting 2.0 features like chapters and persons.
