"""
tanglish_script.py — Display + TTS helper

ALL words → Tamil script (UI + ElevenLabs) so speech stays one continuous
Tamil flow with no English mid-sentence speed breaks.
Clitics stay joined: வேணும்னு / பேசிக்கா / ஆப்கு
"""

from __future__ import annotations

import re

# Longer phrases / multi-word first
_PHRASE_MAP = [
    ("purinjiduchu", "புரிஞ்சுடுச்சு"),
    ("purinjidhuchu", "புரிஞ்சுடுச்சு"),
    ("purinjuchu", "புரிஞ்சுச்சு"),
    ("puriyuthu", "புரியுது"),
    ("puriyudhu", "புரியுது"),
    ("theriyala", "தெரியல"),
    ("theriyum", "தெரியும்"),
    ("therinjikka", "தெரிஞ்சுக்க"),
    ("therinjikkalaam", "தெரிஞ்சுக்கலாம்"),
    ("therinjikalaam", "தெரிஞ்சுக்கலாம்"),
    ("irukkeenga", "இருக்கீங்க"),
    ("irukeenga", "இருக்கீங்க"),
    ("irundhu", "இருந்து"),
    ("irundhaa", "இருந்தா"),
    ("irundha", "இருந்தா"),
    ("sollungala", "சொல்லுங்கல"),
    ("sollungalaa", "சொல்லுங்களா"),
    ("sollunga-nga", "சொல்லுங்கங்க"),
    ("sollunga", "சொல்லுங்க"),
    ("solren", "சொல்றேன்"),
    ("solla", "சொல்ல"),
    ("kelunga", "கேளுங்க"),
    ("pannunga", "பண்ணுங்க"),
    ("pannikkaren", "பண்ணிக்கறேன்"),
    ("pannikaren", "பண்ணிக்கறேன்"),
    ("pannuvanga", "பண்ணுவாங்க"),
    ("pannuvom", "பண்ணுவோம்"),
    ("pannalam", "பண்ணலாம்"),
    ("pannala", "பண்ணல"),
    ("pannaala", "பண்ணல"),
    ("vanakkam", "வணக்கம்"),
    ("vaanga", "வாங்க"),
    ("venum nu", "வேணும்னு"),
    ("venumna", "வேணும்னா"),
    ("venumnaa", "வேணும்னா"),
    ("venuma", "வேணுமா"),
    ("venum-aa", "வேணுமா"),
    ("venumaa", "வேணுமா"),
    ("venum-nu", "வேணும்னு"),
    ("venum", "வேணும்"),
    ("vendum", "வேண்டும்"),
    ("mudiyum", "முடியும்"),
    ("mudiyaadhu", "முடியாது"),
    ("paakalaam", "பாக்கலாம்"),
    ("paarkalaam", "பாக்கலாம்"),
    ("edhavadhu", "ஏதாவது"),
    ("edhavathu", "ஏதாவது"),
    ("kooda", "கூட"),
    ("dhaan", "தான்"),
    ("thaan", "தான்"),
    ("adhula", "அதுல"),
    ("idhula", "இதுல"),
    ("edhula", "எதுல"),
    ("podhum", "போதும்"),
    ("vera edhavadhu", "வேற ஏதாவது"),
    ("vera", "வேற"),
    ("enna nu", "என்னனு"),
    ("enna-nu", "என்னனு"),
    ("ungalukku", "உங்களுக்கு"),
    ("ungaluku", "உங்களுக்கு"),
    ("unga", "உங்க"),
    ("ennai", "என்னை"),
    ("enna", "என்ன"),
    ("idhu", "இது"),
    ("ithu", "இது"),
    ("athu", "அது"),
    ("adhu", "அது"),
    ("seri-nga", "சரிங்க"),
    ("seringaa", "சரிங்க"),
    ("seringa", "சரிங்க"),
    ("seri", "சரி"),
    ("sari", "சரி"),
    ("konjam", "கொஞ்சம்"),
    ("romba", "ரொம்ப"),
    ("nalla", "நல்ல"),
    ("illa", "இல்ல"),
    ("illai", "இல்லை"),
    ("aama", "ஆமா"),
    ("aamaa", "ஆமா"),
    ("appa", "அப்ப"),
    ("ippo", "இப்ப"),
    ("ippothu", "இப்பொழுது"),
    ("innum", "இன்னும்"),
    ("innu", "இன்னு"),
    ("pathi", "பத்தி"),
    ("patthi", "பத்தி"),
    ("mattum", "மட்டும்"),
    ("epdi", "எப்படி"),
    ("eppadi", "எப்படி"),
    ("evlo", "எவ்வளவு"),
    ("evvalavu", "எவ்வளவு"),
    ("enge", "எங்கே"),
    ("engae", "எங்கே"),
    ("yaru", "யாரு"),
    ("yaaru", "யாரு"),
    ("naan", "நான்"),
    ("naanga", "நாங்க"),
    ("neenga", "நீங்க"),
    ("avanga", "அவங்க"),
    ("pesama", "பேசாம"),
    ("pesaama", "பேசாம"),
    ("pesuven", "பேசுவேன்"),
    ("pesuren", "பேசுறேன்"),
    ("sirichu", "சிரிச்சு"),
    ("siricha", "சிரிச்சா"),
    ("ketkala", "கேக்கல"),
    ("kekkala", "கேக்கல"),
    ("kandippa", "கண்டிப்பா"),
    ("note panniten", "நோட் பண்ணிட்டேன்"),
    ("note pannitten", "நோட் பண்ணிட்டேன்"),
    ("note-panniten", "நோட் பண்ணிட்டேன்"),
    ("save panniten", "சேவ் பண்ணிட்டேன்"),
    ("save pannitten", "சேவ் பண்ணிட்டேன்"),
    ("save-panniten", "சேவ் பண்ணிட்டேன்"),
    ("panniten", "பண்ணிட்டேன்"),
    ("pannitten", "பண்ணிட்டேன்"),
    ("panren", "பண்றேன்"),
    ("pannuren", "பண்ணுறேன்"),
    ("panrom", "பண்றோம்"),
    ("note", "நோட்"),
    ("save", "சேவ்"),
    ("saved", "சேவ்ட்"),
    ("aagala", "ஆகல"),
    ("aagalaa", "ஆகல"),
    ("irukka", "இருக்கு"),
    ("irukku", "இருக்கு"),
    ("wait pannunga", "வேட் பண்ணுங்க"),
    ("follow-up", "ஃபாலோஅப்"),
    ("follow up", "ஃபாலோ அப்"),
    ("followup", "ஃபாலோஅப்"),
    ("thank you", "தேங்க்யூ"),
    ("thanks", "தேங்க்ஸ்"),
    ("welcome", "வெல்கம்"),
    ("okay-nga", "ஓகேங்க"),
    ("ok-nga", "ஓகேங்க"),
    ("okaynga", "ஓகேங்க"),
    ("okay", "ஓகே"),
    ("ok", "ஓகே"),
    ("hmm", "ஹ்ம்ம்"),
    ("hm", "ஹ்ம்ம்"),
    ("sure", "ஷூர்"),
    ("please", "ப்ளீஸ்"),
    ("sorry", "சாரி"),
    ("yes", "எஸ்"),
    ("decide", "டிசைட்"),
    ("confirm", "கன்ஃபர்ம்"),
    ("share", "ஷேர்"),
    ("help", "ஹெல்ப்"),
    ("develop", "டெவலப்"),
    ("start", "ஸ்டார்ட்"),
    ("call", "கால்"),
    ("project", "ப்ராஜெக்ட்"),
    ("company", "கம்பெனி"),
    ("contact", "காண்டாக்ட்"),
    ("office", "ஆபீஸ்"),
    ("salem", "சேலம்"),
    ("feature", "ஃபீச்சர்"),
    ("features", "ஃபீச்சர்ஸ்"),
    ("cost", "காஸ்ட்"),
    ("charge", "சார்ஜ்"),
    ("time", "டைம்"),
    ("ready", "ரெடி"),
    ("send", "சென்ட்"),
    ("check", "செக்"),
    ("link", "லிங்க்"),
    ("message", "மெசேஜ்"),
    ("demo", "டெமோ"),
    ("sample", "சாம்பிள்"),
    ("basic-aa", "பேசிக்கா"),
    ("basicaa", "பேசிக்கா"),
    ("basic", "பேசிக்"),
    ("correct-aa", "கரெக்டா"),
    ("correctaa", "கரெக்டா"),
    ("correct", "கரெக்ட்"),
    ("clear-aa", "கிளியரா"),
    ("clearaa", "கிளியரா"),
    ("clear", "கிளியர்"),
    ("unnecessary", "அனெசஸரி"),
    ("development", "டெவலப்மென்ட்"),
    ("developer", "டெவலப்பர்"),
    ("software", "சாஃப்ட்வேர்"),
    ("hardware", "ஹார்ட்வேர்"),
    ("ecommerce", "ஈகாமர்ஸ்"),
    ("e-commerce", "ஈகாமர்ஸ்"),
    ("digital", "டிஜிட்டல்"),
    ("marketing", "மார்க்கெட்டிங்"),
    ("internet", "இன்டர்நெட்"),
    ("website", "வெப்சைட்"),
    ("websites", "வெப்சைட்ஸ்"),
    ("web", "வெப்"),
    ("email", "இமெயில்"),
    ("mail", "மெயில்"),
    ("whatsapp", "வாட்ஸ்அப்"),
    ("instagram", "இன்ஸ்டாகிராம்"),
    ("facebook", "பேஸ்புக்"),
    ("google", "கூகிள்"),
    ("quotation", "கோட்டேஷன்"),
    ("quote", "கோட்"),
    ("requirement", "ரிக்வயர்மென்ட்"),
    ("requirements", "ரிக்வயர்மென்ட்ஸ்"),
    ("service", "சர்வீஸ்"),
    ("services", "சர்வீஸஸ்"),
    ("price", "பிரைஸ்"),
    ("pricing", "பிரைசிங்"),
    ("budget", "பட்ஜெட்"),
    ("team", "டீம்"),
    ("detail", "டீடேல்"),
    ("details", "டீடேல்ஸ்"),
    ("number", "நம்பர்"),
    ("phone", "போன்"),
    ("mobile", "மொபைல்"),
    ("business", "பிசினஸ்"),
    ("name", "நேம்"),
    ("type", "டைப்"),
    ("app-ku", "ஆப்கு"),
    ("appku", "ஆப்கு"),
    ("app", "ஆப்"),
    ("apps", "ஆப்ஸ்"),
    ("android", "ஆண்ட்ராய்ட்"),
    ("ios", "ஐஓஎஸ்"),
    ("design", "டிசைன்"),
    ("designer", "டிசைனர்"),
    ("customer", "கஸ்டமர்"),
    ("care", "கேர்"),
    ("support", "சப்போர்ட்"),
    ("package", "பேக்கேஜ்"),
    ("plan", "பிளான்"),
    ("update", "அப்டேட்"),
    ("updates", "அப்டேட்ஸ்"),
    ("online", "ஆன்லைன்"),
    ("offline", "ஆஃப்லைன்"),
    ("domain", "டொமைன்"),
    ("hosting", "ஹோஸ்டிங்"),
    ("server", "சர்வர்"),
    ("database", "டேட்டாபேஸ்"),
    ("login", "லாகின்"),
    ("password", "பாஸ்வேர்ட்"),
    ("sir", "சர்"),
    ("madam", "மேடம்"),
]

_PARTICLE_MAP = [
    (r"\bnu\b", "னு"),
    (r"\bnaa\b", "னா"),
    (r"\bna\b", "னா"),
    (r"\baa\b", "ஆ"),
    (r"\bla\b", "ல"),
    (r"\bda\b", "டா"),
    (r"\bdi\b", "டி"),
    (r"\bku\b", "கு"),
]

# Hyphen clitics: basic-aa, app-ku → attach after conversion
_HYPHEN_PARTICLE_MAP = [
    (r"-aa\b", "ஆ"),
    (r"-a\b", "அ"),
    (r"-ku\b", "கு"),
    (r"-la\b", "ல"),
    (r"-nga\b", "ங்க"),
    (r"-nu\b", "னு"),
    (r"-na\b", "னா"),
]

_GLUE_PARTICLES = ("னு", "னா", "ஆ", "ல", "டா", "டி", "கு", "ங்க", "அ")

# Rough Latin → Tamil letters for any leftover English token (zero Latin on screen)
_LATIN_TO_TAMIL = [
    ("zh", "ழ்"), ("ng", "ங்"), ("nj", "ஞ்"), ("nd", "ண்ட்"), ("nt", "ன்ட்"),
    ("th", "த்"), ("dh", "த்"), ("sh", "ஷ்"), ("ch", "ச்"), ("kh", "க்"),
    ("ph", "ப்"), ("bh", "ப்"), ("gh", "க்"), ("ck", "க்"), ("qu", "க்வ்"),
    ("aa", "ா"), ("ee", "ீ"), ("oo", "ூ"), ("ai", "ை"), ("au", "ௌ"),
    ("a", "அ"), ("e", "எ"), ("i", "இ"), ("o", "ஒ"), ("u", "உ"),
    ("b", "ப்"), ("c", "க்"), ("d", "ட்"), ("f", "ஃப்"), ("g", "க்"),
    ("h", "ஹ்"), ("j", "ஜ்"), ("k", "க்"), ("l", "ல்"), ("m", "ம்"),
    ("n", "ன்"), ("p", "ப்"), ("q", "க்"), ("r", "ர்"), ("s", "ஸ்"),
    ("t", "ட்"), ("v", "வ்"), ("w", "வ்"), ("x", "க்ஸ்"), ("y", "ய்"), ("z", "ஸ்"),
]


def _latin_word_to_tamil(word: str) -> str:
    """Phonetic fallback so leftover English never stays as Latin letters."""
    w = word.lower()
    if not w.isascii() or not w.isalpha():
        return word
    # Prefer known phrase map hits already applied; this is last resort
    out = []
    i = 0
    while i < len(w):
        matched = False
        for src, dst in _LATIN_TO_TAMIL:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out) or word


def _scrub_remaining_latin(text: str) -> str:
    """Replace any remaining A–Z tokens with Tamil phonetic letters."""

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        # Keep pure numbers
        if tok.isdigit():
            return tok
        return _latin_word_to_tamil(tok)

    return re.sub(r"[A-Za-z][A-Za-z']*", _sub, text)


def _replace_word_ci(text: str, src: str, dst: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(src)}(?![A-Za-z0-9])", re.IGNORECASE)
    return pattern.sub(dst, text)


def _glue_tamil_particles(text: str) -> str:
    if not text:
        return text
    parts = "|".join(re.escape(p) for p in _GLUE_PARTICLES)
    out = re.sub(
        rf"([\u0B80-\u0BFF]+)\s+({parts})(?=[\s,.!?;:\"'”’…]|$)",
        r"\1\2",
        text,
    )
    out = re.sub(rf"([\u0B80-\u0BFF]+)-({parts})", r"\1\2", out)
    return out


def normalize_brand_name(text: str) -> str:
    """
    Ensure the company name is NEVER decoded into Tamil script and is ALWAYS
    written and pronounced cleanly as 'MintX Solutions' or 'MintX'.
    Neutralizes all Tamil script transliterations (மிண்ட், மின்ட், மைண்ட், சொல்யூஷன்ஸ்,
    சொல்யூசன்ஸ், மின்டெக்ஸ், etc.) and Latin misspellings back to official English name.
    """
    if not text:
        return text
    t = str(text)

    # 1. Full brand name with Tamil script:
    # (மின்ட் / மிண்ட் / மைண்ட் / மைன்ட்) + (எக்ஸ் / க்ஸ்) + (சொல்யூஷன் / சொல்யூசன் / etc.)
    t = re.sub(
        r'(?:மி(?:ன்|ண்)ட்|மை(?:ன்|ண்)ட்)[\s\-_]*(?:எக்ஸ்|க்ஸ்)[\s\-_]*(?:சொல்(?:யூ|யு)(?:ஷ|ச)ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?',
        lambda m: 'MintX Solutions-ku' if 'க்கு' in m.group(0) else 'MintX Solutions',
        t,
        flags=re.IGNORECASE,
    )

    # 2. Single-word compounds: மின்டெக்ஸ் / மிண்டெக்ஸ் / மின்டக்ஸ் / மிண்டக்ஸ் / மைண்டெக்ஸ் + சொல்யூஷன்[ஸ்]*
    t = re.sub(
        r'(?:மி(?:ன்|ண்)டெக்ஸ்|மி(?:ன்|ண்)டக்ஸ்|மை(?:ன்|ண்)டெக்ஸ்|மின்தெக்ஸ்)[\s\-_]*(?:சொல்(?:யூ|யு)(?:ஷ|ச)ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?',
        lambda m: 'MintX Solutions-ku' if 'க்கு' in m.group(0) else 'MintX Solutions',
        t,
        flags=re.IGNORECASE,
    )

    # 3. Half English half Tamil: MintX + சொல்யூஷன்[ஸ்]* / சொல்யூசன்[ஸ்]*
    t = re.sub(
        r'\bMint[\s\-_]*X\b[\s\-_]*(?:சொல்(?:யூ|யு)(?:ஷ|ச)ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?',
        lambda m: 'MintX Solutions-ku' if 'க்கு' in m.group(0) else 'MintX Solutions',
        t,
        flags=re.IGNORECASE,
    )

    # 4. MintX alone in Tamil script:
    t = re.sub(
        r'(?:மி(?:ன்|ண்)ட்|மை(?:ன்|ண்)ட்)[\s\-_]*(?:எக்ஸ்|க்ஸ்)(?:(?:ு|்)?க்கு)?',
        lambda m: 'MintX-ku' if 'க்கு' in m.group(0) else 'MintX',
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r'(?:மி(?:ன்|ண்)டெக்ஸ்|மி(?:ன்|ண்)டக்ஸ்|மை(?:ன்|ண்)டெக்ஸ்|மின்தெக்ஸ்)(?:(?:ு|்)?க்கு)?',
        lambda m: 'MintX-ku' if 'க்கு' in m.group(0) else 'MintX',
        t,
        flags=re.IGNORECASE,
    )

    # 5. Latin phonetic variants for MintX Solutions:
    t = re.sub(r'\b(?:mint|mind|min)[\s\-_]*x\s+solutions?\b', 'MintX Solutions', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(?:mintech|minteq|mintec|mintex|mindx)\s+solutions?\b', 'MintX Solutions', t, flags=re.IGNORECASE)

    # 6. Latin phonetic variants for MintX alone:
    t = re.sub(r'\b(?:mint|mind|min)[\s\-_]+x\b', 'MintX', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(?:mintech|minteq|mintec|mindx|mintex|minx)\b', 'MintX', t, flags=re.IGNORECASE)
    t = re.sub(r'\bmintx\b', 'MintX', t, flags=re.IGNORECASE)

    # 7. Clean glued Tamil suffixes like MintX Solutionsகு -> MintX Solutions-ku
    t = re.sub(r'\bMintX Solutions[\s\-_]*(?:கு|க்கு)(?=[\s,.!?;:\"\'”’…]|$)', 'MintX Solutions-ku', t)
    t = re.sub(r'\bMintX[\s\-_]*(?:கு|க்கு)(?=[\s,.!?;:\"\'”’…]|$)', 'MintX-ku', t)

    return t


def tanglish_to_tamil_display(text: str) -> str:
    """Convert Tanglish to Tamil display script; strictly preserve MintX Solutions in English Latin script intact."""
    if not text or not str(text).strip():
        return text

    # First normalize any prior Tamil transliteration of MintX back to official English name
    clean_in = normalize_brand_name(str(text))

    # Only Tamil / Tanglish is converted; English and other Indian languages are never scrambled into Tamil
    try:
        from language_detector import detect_language
        if detect_language(clean_in) not in ("ta", "tunglish"):
            return clean_in.strip()
    except Exception:
        pass

    # Strictly protect brand tokens and hyphenated forms from being decoded into Tamil
    out = clean_in
    out = re.sub(r"\bMintX Solutions-ku\b", "___MINTX_SOLUTIONS_KU___", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMintX Solutionsku\b", "___MINTX_SOLUTIONS_KU___", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMintX-ku\b", "___MINTX_KU___", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMintXku\b", "___MINTX_KU___", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMintX Solutions\b", "___MINTX_SOLUTIONS___", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMintX\b", "___MINTX___", out, flags=re.IGNORECASE)

    for src, dst in _PHRASE_MAP:
        out = _replace_word_ci(out, src, dst)

    for pat, dst in _HYPHEN_PARTICLE_MAP:
        out = re.sub(pat, dst, out, flags=re.IGNORECASE)

    for pat, dst in _PARTICLE_MAP:
        out = re.sub(pat, dst, out, flags=re.IGNORECASE)

    out = _glue_tamil_particles(out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()

    # Restore protected brand names intact
    out = out.replace("___MINTX_SOLUTIONS_KU___", "MintX Solutions-ku")
    out = out.replace("___MINTX_KU___", "MintX-ku")
    out = out.replace("___MINTX_SOLUTIONS___", "MintX Solutions")
    out = out.replace("___MINTX___", "MintX")

    # Final guarantee: never show Tamil-decoded MintX
    out = normalize_brand_name(out)
    return out


def tanglish_for_tts(text: str) -> str:
    """
    Format text specifically for TTS engines (ElevenLabs / Neural TTS).
    1. If English: keep English script, format brand as 'Mint X Solutions' so TTS speaks 'Mint X Solutions'.
    2. If Tamil / Tanglish: convert Tanglish words to Tamil script (preventing English mid-sentence speed breaks
       and Telugu mispronunciations), while strictly formatting brand as 'Mint X Solutions-ku' or 'Mint X Solutions'.
    """
    if not text or not str(text).strip():
        return ""
    clean_in = normalize_brand_name(str(text).strip())

    try:
        from language_detector import detect_language
        if detect_language(clean_in) not in ("ta", "tunglish"):
            # English or another Indian language: keep its script, speak 'Mint X Solutions' cleanly
            t = re.sub(r'\bMintX Solutions\b', 'Mint X Solutions', clean_in, flags=re.IGNORECASE)
            t = re.sub(r'\bMintX\b', 'Mint X', t, flags=re.IGNORECASE)
            return t
    except Exception:
        pass

    # Tanglish or Tamil text: convert Roman Tanglish words into Tamil script
    tamil_script = tanglish_to_tamil_display(clean_in)

    # Format brand name with clean spacing so ElevenLabs pronounces "Mint" + "X" + "Solutions"
    t = re.sub(r'\bMintX Solutions[\s\-_]*(?:கு|க்கு|ku)\b', 'Mint X Solutions-ku', tamil_script, flags=re.IGNORECASE)
    t = re.sub(r'\bMintX[\s\-_]*(?:கு|க்கு|ku)\b', 'Mint X-ku', t, flags=re.IGNORECASE)
    t = re.sub(r'\bMintX Solutions\b', 'Mint X Solutions', t, flags=re.IGNORECASE)
    t = re.sub(r'\bMintX\b', 'Mint X', t, flags=re.IGNORECASE)

    return t


def maybe_tamil_display(text: str, enabled: bool = True) -> str:
    if not enabled:
        return normalize_brand_name(text)
    try:
        return tanglish_to_tamil_display(text)
    except Exception:
        return normalize_brand_name(text)


