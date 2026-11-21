"""
language_detector.py — Language Detection & Mirroring for MintX AI Assistant

Accurately classifies input speech transcript into:
- 'en'       : Pure English
- 'ta'       : Tamil in Tamil Script (Unicode \\u0B80-\\u0BFF)
- 'tunglish' : Tunglish / Tanglish (Tamil spoken words transcribed into Latin script)
- 'hi', 'te', 'kn', 'ml', 'bn', 'gu', 'pa', 'od' : other Indian languages, detected by native script
  (every language Sarvam AI Bulbul v3 can speak)
"""

import re
from typing import Literal

LanguageType = Literal["en", "ta", "tunglish", "hi", "te", "kn", "ml", "bn", "gu", "pa", "od"]

# Native Unicode script ranges for Bulbul v3 languages. Devanagari is mapped to Hindi
# (Marathi shares the script; Bulbul's hi-IN voice reads Marathi text acceptably).
INDIC_SCRIPTS = {
    "ta": r"[\u0B80-\u0BFF]",
    "hi": r"[\u0900-\u097F]",
    "bn": r"[\u0980-\u09FF]",
    "pa": r"[\u0A00-\u0A7F]",
    "gu": r"[\u0A80-\u0AFF]",
    "od": r"[\u0B00-\u0B7F]",
    "te": r"[\u0C00-\u0C7F]",
    "kn": r"[\u0C80-\u0CFF]",
    "ml": r"[\u0D00-\u0D7F]",
}

SARVAM_LANGUAGE_CODES = {
    "en": "en-IN",
    "ta": "ta-IN",
    "tunglish": "ta-IN",
    "hi": "hi-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "od": "od-IN",
}

# Common phonetic markers and vocabulary characteristic of Tunglish / Tanglish
TUNGLISH_MARKERS = {
    # Pronouns & Possessives
    "naan", "naanga", "neenga", "neengal", "unga", "ungaloda", "ungalukku", "ungaluku", "ungala", "ungalai",
    "en", "ennoda", "enakku", "enaku", "namma", "nammoda", "enga", "engaloda", "avanga", "avaru", "avargal",
    "idhu", "ithu", "idha", "idhellam", "athu", "adhu", "adha", "athellam", "edhu", "edha", "edhellam",
    "yar", "yaaru", "enna", "ennadhu", "edhavadhu", "edhavathu",
    # Verbs & Auxiliary
    "venum", "vendum", "venuma", "venumaa", "venumna", "venam", "vendaam",
    "sollunga", "solren", "solreenga", "solringa", "solla", "sollanum", "kelunga",
    "pannunga", "panren", "panreenga", "panringa", "pannreenga", "pannringa", "panniten",
    "pannuvanga", "pannuvom", "panrom", "pannalam", "pannanum", "pannikaren", "pannikkaren", "pannikalam",
    "irukku", "irukkeenga", "irukeenga", "irukkinga", "irukken", "irukkom", "irundha", "irundhaa", "irundhu", "irukkum", "irukaadhu",
    "purinjiduchu", "purinjuchu", "puriyuthu", "puriyudhu", "puriyala",
    "theriyala", "theriyum", "theriyuma", "theriyadhu", "therinjikka", "therinjikkalaam",
    "kudukren", "kudunga", "anuppunga", "anupunga",
    "mudiyum", "mudiyuma", "mudiyumaa", "mudiyaadhu", "mudiyadhu",
    "paakalaam", "paarkalaam", "paakanum", "paarkanum",
    "pesunga", "pesuren", "pesureenga", "pesuringa", "pesanum", "pesalaam",
    "kekkala", "kekala", "kekudhu", "kekkudhu",
    # Common Particles & Fillers
    "vanakkam", "seri", "sari", "seringa", "seri-nga", "saringa", "aama", "aamaa", "konjam",
    "romba", "nalla", "illa", "illai", "illama", "problem illa",
    "evlo", "evvalavu", "evalo", "eppadi", "epdi", "engae", "enge", "inga", "inge", "anga", "ange",
    "madhiri", "kandippa", "apdi", "ippo", "innum", "innu",
    "pathi", "patthi", "mattum", "podhum", "vera", "dhaan", "thaan", "kooda", "koodaadhu",
    "paathutu", "apram", "appuram", "aduthu", "marubadi", "marubadiyum", "thirumba", "nandri",
    "aachu", "aayiduchu", "paravala", "paravayilla", "kollunga"
}

# Suffixes common in Tanglish loanwords with explicit hyphens or distinctive morphology
# Prevents false positive matching of ordinary English words like menu, tesla, vanilla, formula, etc.
TUNGLISH_SUFFIXES = [
    r"-ku\b",
    r"-la\b",
    r"-nga\b",
    r"-oda\b",
    r"-aa\b",
    r"-nu\b",
    r"-lerundhu\b",
    r"\b[a-z]+-ku\b",
    r"\b[a-z]+-la\b",
    r"\b[a-z]+-nga\b",
    r"\b[a-z]+-oda\b",
    r"\b[a-z]+-aa\b",
    r"\b[a-z]+-nu\b",
]

# Common English stopwords to ensure unambiguous English sentences are never misclassified
COMMON_ENGLISH_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "want", "need", "like", "looking", "please", "tell", "help", "give", "know",
    "call", "service", "services", "price", "pricing", "cost", "quote", "quotation",
    "charges", "website", "app", "mobile", "ecommerce", "development", "business",
    "company", "shop", "store", "details", "information", "requirement", "requirements",
    "contact", "hello", "hi", "good", "morning", "afternoon", "evening", "thanks", "thank"
}


def detect_language(text: str) -> LanguageType:
    """
    Detect whether the given text is English ('en'), Tamil script ('ta'),
    or Tunglish ('tunglish').
    """
    if not text or not str(text).strip():
        return "en"

    cleaned = str(text).strip()

    # 1. Check for native Indian scripts (Tamil, Hindi, Telugu, ...): dominant script wins
    script_counts = {lang: len(re.findall(pattern, cleaned)) for lang, pattern in INDIC_SCRIPTS.items()}
    best_lang = max(script_counts, key=script_counts.get)
    indic_chars = script_counts[best_lang]
    total_letters = len(re.findall(r"[A-Za-z]", cleaned)) + sum(script_counts.values())
    if total_letters > 0 and (indic_chars / total_letters) > 0.20:
        return best_lang

    # 2. Check for Tunglish vocabulary & phonetic markers in Latin script
    words = [w.lower().strip(".,!?;:\"'()[]{}") for w in cleaned.split()]
    tunglish_hits = sum(1 for w in words if w in TUNGLISH_MARKERS)

    # Check for distinctive hyphenated suffix patterns (e.g. website-ku, app-aa, price-la)
    suffix_hits = sum(1 for pattern in TUNGLISH_SUFFIXES if re.search(pattern, cleaned.lower()))

    # If explicit Tunglish markers or suffixes are present
    if (tunglish_hits + suffix_hits) > 0:
        return "tunglish"

    # 3. Default to English
    return "en"


def get_language_display_name(lang: LanguageType) -> str:
    """Return user-friendly language name."""
    mapping = {
        "en": "English",
        "ta": "Tamil",
        "tunglish": "Tamil (Tanglish)",
        "hi": "Hindi",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "bn": "Bengali",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "od": "Odia",
    }
    return mapping.get(lang, "English")


def get_sarvam_language_code(lang: str) -> str:
    """Map a detected language to the Sarvam AI Bulbul v3 language_code (BCP-47)."""
    return SARVAM_LANGUAGE_CODES.get(lang, "en-IN")
