"""
ai_assistant.py — AI Logic

Handles all AI interactions using OpenAI.
Manages conversation history, language detection, and intent recognition.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from instructions import get_active_system_prompt, DEFAULT_MINTX_PROMPT
from rag import search_mintx_knowledge
from language_detector import detect_language
from tanglish_script import normalize_brand_name

SYSTEM_PROMPT = get_active_system_prompt() or DEFAULT_MINTX_PROMPT

# ── AI Assistant Class ──────────────────────────────────────────────



class AIAssistant:
    """
    Manages a single phone call conversation with OpenAI.

    Usage:
        assistant = AIAssistant(call_id="abc123")
        response = assistant.chat("Hello, I need help.")
        print(response)
    """

    def __init__(self, call_id: str = "default", system_prompt: str = None, voice_id: str = None):
        """
        Initialize a new AI assistant for a phone call.

        Args:
            call_id: Unique identifier for this call/conversation.
            system_prompt: Optional custom system prompt / persona instructions.
            voice_id: Optional custom voice ID (defaults to ElevenLabs voice ID).
        """
        self.call_id = call_id
        model = os.getenv("OPENAI_REASONING_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
        if "realtime" in model.lower():
            model = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")
        self.model = model
        self.tts_provider = os.getenv("TTS_PROVIDER", "sarvam").lower()
        if self.tts_provider in ("sarvam", "sarvamai"):
            self.voice_id = voice_id or os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
            self.voice_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
        elif self.tts_provider == "elevenlabs" or voice_id == "elevenlabs":
            self.voice_id = (voice_id if voice_id and voice_id != "elevenlabs" else os.getenv("ELEVENLABS_VOICE_ID", "mGboHvCVOXWYeFL8KTR0"))
            self.voice_model = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
        elif self.tts_provider in ("openai", "gpt", "gpt-speech", "tts"):
            self.voice_id = voice_id or os.getenv("OPENAI_VOICE", "nova")
            self.voice_model = "tts-1"
        else:
            self.voice_id = voice_id or os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
            self.voice_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = system_prompt or get_active_system_prompt() or SYSTEM_PROMPT

        # Customer lead data matching schema
        self.lead_data = {
            "name": "",
            "business_name": "",
            "service": "",
            "phone": "",
            "requirement": "",
            "language": "en",
            "status": "new"
        }
        try:
            from database import get_lead
            existing_lead = get_lead(self.call_id)
            if existing_lead:
                for k in ["name", "business_name", "service", "phone", "requirement", "language", "status"]:
                    if existing_lead.get(k):
                        self.lead_data[k] = existing_lead[k]
        except Exception:
            pass

        # Current detected language
        self.current_lang = self.lead_data.get("language") or "tunglish"

        # Conversation history starts with the active system prompt
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def set_system_prompt(self, new_prompt: str):
        """Update system prompt and replace system message in history."""
        self.system_prompt = new_prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": new_prompt})

    def _update_lead_context(self, user_query: str = ""):
        """Give the model the saved fields, RAG knowledge context, and the single next question in the caller's language."""
        lang = self.lead_data.get("language", "en")

        # RAG Knowledge Retrieval
        rag_context = ""
        if user_query and user_query.strip():
            try:
                rag_context = search_mintx_knowledge(user_query, top_k=2)
            except Exception as e:
                print(f"[!] RAG lookup error: {e}")

        # Determine question/response based on language (English, Tunglish, Tamil)
        has_min_details = bool(self.lead_data.get("name") and self.lead_data.get("service"))
        is_confirmed = self.lead_data.get("status") == "confirmed"

        if lang == "ta":
            # Natural spoken Tamil script
            missing_questions = [
                ("name", "உங்கள் பெயர் என்ன?"),
                ("business_name", "உங்கள் நிறுவனம் அல்லது தொழில் பெயர் என்ன?"),
                ("service", "உங்களுக்கு என்ன சேவை தேவை?"),
                ("phone", "உங்கள் தொலைபேசி எண்ணை கூற முடியுமா?"),
                ("requirement", "உங்கள் தேவை என்னவென்று கூற முடியுமா?"),
            ]
            confirm_q = f"உங்கள் விவரங்களை உறுதிப்படுத்துகிறேன்: பெயர் {self.lead_data.get('name')}, தொழில் {self.lead_data.get('business_name') or 'குறிப்பிடவில்லை'}, சேவை {self.lead_data.get('service')}. இது சரியா?"
            completion_q = "நன்றி! உங்கள் விவரங்களை உறுதிசெய்து சேமித்துவிட்டேன். MintX குழு விரைவில் உங்களை தொடர்புகொள்வார்கள்."
            lang_prompt = "Caller is speaking Tamil. Respond ONLY in natural spoken Tamil. Never speak English or Tunglish."
        elif lang == "tunglish":
            # Natural Roman Tunglish
            missing_questions = [
                ("name", "Unga name enna?"),
                ("business_name", "Unga business enna type?"),
                ("service", "Ungalukku enna service venum?"),
                ("phone", "Unga phone number sollunga."),
                ("requirement", "Unga requirement enna?"),
            ]
            confirm_q = f"Unga details once confirm pannikkaren: Unga name {self.lead_data.get('name')}, business {self.lead_data.get('business_name') or 'solla villai'}, service {self.lead_data.get('service')}. Idhu correct-aa?"
            completion_q = "Thank you! Unga details confirm panni save panniten. MintX team follow-up pannuvanga."
            lang_prompt = "Caller is speaking Tunglish. Respond ONLY in natural Roman Tunglish. Never convert to formal Tamil script or switch to English."
        else:
            # English
            missing_questions = [
                ("name", "May I know your name?"),
                ("business_name", "What is your business name?"),
                ("service", "What service are you looking for?"),
                ("phone", "Could you share your phone number?"),
                ("requirement", "What is your specific requirement?"),
            ]
            confirm_q = f"Let me confirm your details: Your name is {self.lead_data.get('name')}, business is {self.lead_data.get('business_name') or 'not specified'}, and service is {self.lead_data.get('service')}. Is that correct?"
            completion_q = "Thank you! Your details have been confirmed and saved. Our MintX team will follow up with you shortly."
            lang_prompt = "Caller is speaking English. Respond ONLY in 100% pure English. Absolutely NEVER mix Tamil or Tanglish words."

        if has_min_details and not is_confirmed and not self.lead_data.get("asked_confirmation"):
            next_q = confirm_q
        else:
            next_q = next(
                (question for field, question in missing_questions if not self.lead_data.get(field)),
                completion_q
            )

        saved = {field: value for field, value in self.lead_data.items() if value and field not in ("asked_confirmation",)}
        rag_section = f"\nVERIFIED MINTX RAG KNOWLEDGE BASE:\n{rag_context}\n" if rag_context else ""

        context = (
            f"STRICT LANGUAGE DIRECTIVE: {lang_prompt}\n"
            f"Never mix languages. Response length: 1 to 3 short spoken sentences.\n"
            f"{rag_section}"
            f"CURRENT CUSTOMER DETAILS: {saved or 'none'}.\n"
            f"Next step: If customer asked a question, answer it concisely using RAG knowledge. Then ask this next question: {next_q}\n"
            "Keep the response short, conversational, and natural for a phone call."
        )
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = f"{self.system_prompt}\n\n{context}"

    def extract_and_save_lead_fields(self, user_message: str):
        """
        Extracts customer lead fields (name, business_name, service, phone, requirement)
        and checks for confirmation before marking as confirmed in the database.
        """
        last_ai = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                last_ai = msg.get("content", "").lower()
                break

        text = user_message.strip()
        updated = {}

        # Tokenize words cleanly preserving Unicode Indic scripts (Tamil, etc.)
        clean_words = {re.sub(r"^[^\w\u0B80-\u0BFF]+|[^\w\u0B80-\u0BFF]+$", "", w) for w in text.lower().split() if w.strip()}
        confirm_words = {"correct", "yes", "yeah", "yup", "aama", "aamaa", "seri", "sari", "right", "sure", "ok", "okay", "சரி", "ஆமாம்", "ஆம்", "perfect", "done"}
        is_confirmation_response = (
            bool(clean_words & confirm_words)
            or any(phrase in text.lower() for phrase in [
                "that is correct", "it is correct", "correct-aa", "correct aa", "details correct",
                "விவரங்கள் சரி", "சரி", "ஆமாம்", "ஆம்"
            ])
        )
        is_short_confirmation = is_confirmation_response and len(clean_words) <= 6

        # Check if caller input is a question rather than a statement/detail
        is_question = text.endswith("?") or any(text.lower().startswith(q) for q in ["how", "what", "where", "can", "is", "do", "could", "would", "evlo", "enna", "eppadi", "எவ்வளவு", "என்ன", "எங்கு", "எப்போது"])

        # Detect language using language_detector
        detected_lang = detect_language(text)
        self.lead_data["language"] = detected_lang
        self.current_lang = detected_lang
        updated["language"] = detected_lang

        # Check for confirmation response (Rule 7, 8)
        if is_confirmation_response and any(k in last_ai for k in ["confirm", "correct", "சரியா", "உறுதி", "is that correct", "details"]):
            if self.lead_data.get("status") != "confirmed":
                self.lead_data["status"] = "confirmed"
                updated["status"] = "confirmed"
                print(f"[AIAssistant] Lead confirmed by customer! Status updated to confirmed.")

        # 1. Phone number validation & extraction (6-9 followed by 9 digits)
        phone_digits = re.sub(r"\D", "", text)
        phone_match = re.search(r"(?:\+?91)?[6-9]\d{9}", re.sub(r"[\s\-\(\)]", "", text))
        if phone_match and not is_short_confirmation:
            p_val = phone_match.group(0)
            if p_val.startswith("+91"):
                p_val = p_val[3:]
            elif p_val.startswith("91") and len(p_val) == 12:
                p_val = p_val[2:]
            if len(p_val) == 10 and p_val != self.lead_data.get("phone"):
                self.lead_data["phone"] = p_val
                updated["phone"] = p_val
        elif not is_short_confirmation and ("phone" in last_ai or "number" in last_ai) and not self.lead_data.get("phone"):
            if len(phone_digits) >= 10 and phone_digits[-10] in "6789":
                p_val = phone_digits[-10:]
                self.lead_data["phone"] = p_val
                updated["phone"] = p_val

        # 2. Service extraction
        services = [
            ("e-commerce website", "E-commerce website"),
            ("ecommerce website", "E-commerce website"),
            ("e-commerce", "E-commerce"),
            ("ecommerce", "E-commerce"),
            ("ஈகாமர்ஸ் வலைத்தளம்", "E-commerce website"),
            ("இணையவழி விற்பனை", "E-commerce website"),
            ("ஈகாமர்ஸ்", "E-commerce website"),
            ("mobile app", "Mobile App"),
            ("கைப்பேசி செயலி", "Mobile App"),
            ("மொபைல் ஆப்", "Mobile App"),
            ("செயலி", "Mobile App"),
            ("digital marketing", "Digital Marketing"),
            ("டிஜிட்டல் மார்க்கெட்டிங்", "Digital Marketing"),
            ("இணையவழி விளம்பரம்", "Digital Marketing"),
            ("mintx qr", "MintX QR"),
            ("maintenance", "Maintenance"),
            ("பராமரிப்பு", "Maintenance"),
            ("website", "Website"),
            ("வலைத்தளம்", "Website"),
            ("இணையதளம்", "Website"),
            ("வெப்சைட்", "Website"),
            ("app", "Mobile App"),
            ("seo", "Digital Marketing"),
            ("3d printing", "3D Printing"),
            ("3டி பிரிண்டிங்", "3D Printing"),
        ]
        matched_svc = None
        if not is_short_confirmation:
            for pattern, canonical in services:
                if pattern in text.lower():
                    matched_svc = canonical
                    break
            if matched_svc and not self.lead_data.get("service"):
                self.lead_data["service"] = matched_svc
                updated["service"] = matched_svc
            elif not is_question and ("service" in last_ai or "enna service" in last_ai or "சேவை" in last_ai) and not self.lead_data.get("service"):
                clean_svc = text.strip(" .!,")
                if len(clean_words) <= 6:
                    self.lead_data["service"] = clean_svc
                    updated["service"] = clean_svc

        # 3. Name extraction
        name_patterns = [
            r"(?:my name is|i am|i\'m|this is|naam|naan|\bna\b|en name|peru|என் பெயர்|பெயர்)\s+([A-Za-z\u0B80-\u0BFF]+)",
        ]
        extracted_name = None
        if not is_short_confirmation and not is_question:
            for p in name_patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    extracted_name = m.group(1).title()
                    break
            if extracted_name and extracted_name != self.lead_data.get("name"):
                self.lead_data["name"] = extracted_name
                updated["name"] = extracted_name
            elif ("name?" in last_ai or "unga name" in last_ai or "your name" in last_ai or "பெயர்" in last_ai) and not self.lead_data.get("name"):
                words = text.split()
                if len(words) <= 3 and not any(w.lower() in [s[0] for s in services] for w in words):
                    clean_name = text.strip(" .!,").title()
                    self.lead_data["name"] = clean_name
                    updated["name"] = clean_name

        # 4. Business name extraction
        biz_patterns = [
            r"(?:en|my)?\s*\bbusiness(?:\s+name)?(?:\s+is|\s*:)?\s+([A-Za-z0-9\u0B80-\u0BFF][A-Za-z0-9\u0B80-\u0BFF &.-]*?)(?:\.|$|,|enakku|i need|service|\s+nu)",
            r"(?:from|company is|business is|shop is|நிறுவனம்|தொழில்|கடை)\s+([A-Za-z0-9\u0B80-\u0BFF\s]+?)(?:\.|$|,|enakku|i need|service)",
            r",\s*([A-Za-z0-9\u0B80-\u0BFF][A-Za-z0-9\u0B80-\u0BFF &.-]*?)(?:\s+la\b|\s+irukken\b|,|\.)",
        ]
        extracted_biz = None
        if not is_short_confirmation and not is_question:
            for p in biz_patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    extracted_biz = m.group(1).strip()
                    break
            if extracted_biz and extracted_biz != self.lead_data.get("business_name"):
                self.lead_data["business_name"] = extracted_biz
                updated["business_name"] = extracted_biz
            elif ("business name" in last_ai or "what is your business" in last_ai or "type of business" in last_ai or "business enna" in last_ai or "company name" in last_ai or "தொழில் பெயர்" in last_ai or "நிறுவனம்" in last_ai) and not self.lead_data.get("business_name"):
                if len(clean_words) <= 6:
                    clean_biz = text.strip(" .!,")
                    self.lead_data["business_name"] = clean_biz
                    updated["business_name"] = clean_biz

        # 5. Requirement extraction (keep customer's original words)
        if not is_short_confirmation and not is_question:
            req_prompt_asked = any(p in last_ai for p in ["your requirement", "specific requirement", "unga requirement", "உங்கள் தேவை என்ன", "தேவை என்ன", "தேவை என்னவென்று"])
            if (req_prompt_asked or re.search(r"\b(?:my\s+)?requirement\s+(?:is|:)", text, re.IGNORECASE)) and text and not self.lead_data.get("requirement"):
                self.lead_data["requirement"] = text
                updated["requirement"] = text

        # Update database if fields changed
        if updated:
            try:
                from database import save_or_update_lead
                save_or_update_lead(self.call_id, self.lead_data)
            except Exception as e:
                print(f"[!] DB lead update error: {e}")

    def _is_correction_intent(self, text: str) -> bool:
        """Detect whether user message is explicitly a training correction or rule update command."""
        lower = text.lower()
        triggers = [
            "correction:", "கரெக்ஷன்:", "rule update", "update rule", "change rule", "ரூல் மாத்து", "விதி மாற்று",
            "starting greeting change", "set starting greeting", "starting greeting-a மாத்து",
            "இப்படி பேசாத", "அப்படி பேசாத", "don't say like that", "never say that again",
            "fix pannu rule", "update instruction", "train assistant", "டெவலப்பர் கமாண்ட்", "developer command",
            "rule-a mathu", "rule-a update pannu"
        ]
        return any(t in lower for t in triggers)

    def _handle_live_correction(self, user_message: str) -> str:
        """Process spoken or typed correction, train instructions, and return spoken confirmation."""
        try:
            from instructions import refine_instruction_with_feedback, get_all_data
            data = get_all_data()
            active_id = data.get("active_id", "default-mintx")
            last_ai = ""
            for msg in reversed(self.messages):
                if msg.get("role") == "assistant":
                    last_ai = msg.get("content", "")
                    break

            res = refine_instruction_with_feedback(
                instruction_id=active_id,
                user_feedback=user_message,
                last_assistant_response=last_ai,
                conversation_context="",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            new_prompt = res.get("new_instructions")
            verbal_ack = res.get("verbal_ack") or "Sure sir, naan rule update panniten!"
            if new_prompt:
                self.set_system_prompt(new_prompt)

            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": verbal_ack})
            print(f"[AIAssistant Auto-Train]: Learned rule '{res.get('summary_en')}' -> '{verbal_ack}'")
            return verbal_ack
        except Exception as e:
            print(f"[!] Live correction error in AIAssistant: {e}")
            ack = "Sure sir, naan rule update panniten!"
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": ack})
            return ack

    def chat(self, user_message: str) -> str:
        """
        Send a message to the AI and get a response.
        Auto-detects corrections to self-train, else extracts lead info and answers.
        """
        # Ensure company brand name is never decoded into Tamil script
        user_message = normalize_brand_name(user_message)

        # 1. Detect live training / correction intent
        if self._is_correction_intent(user_message):
            return normalize_brand_name(self._handle_live_correction(user_message))

        # 2. Regular customer care flow: extract and update lead information
        self.extract_and_save_lead_fields(user_message)

        # Keep the model aligned with the saved fields, RAG knowledge, and one-question flow.
        self._update_lead_context(user_query=user_message)

        # Add the caller's message to history
        self.messages.append({"role": "user", "content": user_message})

        # Call OpenAI with support for reasoning models (max_completion_tokens)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_completion_tokens=600,  # Room for internal reasoning + 1-3 spoken sentences
            )
        except Exception as e:
            if "max_completion_tokens" in str(e):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    max_tokens=160,
                    temperature=0.3,
                )
            else:
                raise e

        # Extract the AI's response
        raw_message = response.choices[0].message.content.strip()

        # Enforce clean short responses (1-3 sentences max) without chopping words abruptly
        lines = [line.strip(' "\'') for line in raw_message.split("\n") if line.strip(' "\'')]
        ai_message = " ".join(lines[:3]) if lines else raw_message.strip(' "\'')
        ai_message = normalize_brand_name(ai_message)

        # Add to history
        self.messages.append({"role": "assistant", "content": ai_message})

        return ai_message

    def get_history(self) -> list:
        """
        Get the conversation history (excluding system prompt).

        Returns:
            List of message dicts with 'role' and 'message' keys.
        """
        history = []
        for msg in self.messages:
            if msg["role"] != "system":
                history.append({
                    "role": msg["role"],
                    "message": msg["content"],
                })
        return history

    def get_summary(self) -> str:
        """
        Ask the AI to summarize the conversation in one line.

        Returns:
            A short summary string.
        """
        if len(self.messages) <= 1:
            return "No conversation took place."

        summary_messages = self.messages + [
            {
                "role": "user",
                "content": (
                    "Summarize this entire phone conversation in one short "
                    "sentence. Focus on what the caller wanted. "
                    "Respond only with the summary, nothing else."
                ),
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_messages,
                max_completion_tokens=60,
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_messages,
                max_tokens=60,
                temperature=0.3,
            )

        return response.choices[0].message.content.strip()

    def generate_speech(self, text: str) -> bytes:
        """
        Synthesize speech using the main assistant's configured ElevenLabs voice ID.
        """
        from voice import text_to_speech
        return text_to_speech(text, voice=self.voice_id)

    def speak(self, text: str, play_aloud: bool = True) -> bytes:
        """
        Synthesize and optionally play speech using the main assistant's ElevenLabs voice ID.
        """
        from voice import speak
        return speak(text, voice=self.voice_id, play_aloud=play_aloud)

    def save_speech(self, text: str) -> str:
        """
        Synthesize and save speech to an MP3 file using the main assistant's ElevenLabs voice ID.
        """
        from voice import text_to_speech_save
        return text_to_speech_save(text, voice=self.voice_id)

    def detect_language(self, text: str) -> str:
        """
        Detect language: English ('en'), Tamil script ('ta'), or Tunglish ('tunglish').
        Uses the specialized MintX language detector.
        """
        return detect_language(text)

    def detect_intent(self, text: str) -> str:
        """
        Simple keyword-based intent detection.

        Args:
            text: The caller's message.

        Returns:
            'transfer' — caller wants a human agent
            'appointment' — caller wants to book an appointment
            'general' — general conversation
        """
        text_lower = text.lower()

        # Check for human transfer intent
        transfer_keywords = [
            "speak to a person", "talk to a human", "transfer",
            "real person", "human agent", "agent", "speak to someone",
            "manitha", "person kitta",  # Tamil keywords
        ]
        for keyword in transfer_keywords:
            if keyword in text_lower:
                return "transfer"

        # Check for appointment intent
        appointment_keywords = [
            "appointment", "book", "schedule", "booking",
            "naalaikku", "appointment venum", "book pannu",  # Tamil
        ]
        for keyword in appointment_keywords:
            if keyword in text_lower:
                return "appointment"

        return "general"


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("--- Starting AI Assistant Terminal Mode ---")
    assistant = AIAssistant()
    print("AI: Hello! How can I help you today? (Type 'exit' to quit)")

    while True:
        try:
            user_msg = input("\nYou: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit"):
                print("AI: Goodbye!")
                break
            ai_reply = assistant.chat(user_msg)
            print(f"AI: {ai_reply}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
