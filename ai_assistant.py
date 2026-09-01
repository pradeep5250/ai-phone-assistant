"""
ai_assistant.py — AI Logic

Handles all AI interactions using OpenAI.
Manages conversation history, language detection, and intent recognition.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a friendly AI phone assistant.

You are an AI, not a human. Always be honest about this if asked.
Speak naturally and politely.
Keep responses short (1-2 sentences) because you are talking over a phone call.
Understand English and Tamil.
Respond in the same language the caller uses.
If the caller speaks Tamil, respond in Tamil (use Tamil script or romanized Tamil).
If information is missing, ask a clarifying question.
Never invent information.

You can help with:
- Booking appointments (ask for date and time)
- Answering general questions
- Transferring to a human agent (if the caller requests)

If the caller wants to speak to a human, respond with:
"Sure, I'll transfer you to our team now."

If the caller wants an appointment, collect:
- Preferred date
- Preferred time
- Purpose (optional)
Then confirm the details.
"""

# ── AI Assistant Class ──────────────────────────────────────────────


class AIAssistant:
    """
    Manages a single phone call conversation with OpenAI.

    Usage:
        assistant = AIAssistant(call_id="abc123")
        response = assistant.chat("Hello, I need help.")
        print(response)
    """

    def __init__(self, call_id: str = "default"):
        """
        Initialize a new AI assistant for a phone call.

        Args:
            call_id: Unique identifier for this call/conversation.
        """
        self.call_id = call_id
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Conversation history starts with the system prompt
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str) -> str:
        """
        Send a message to the AI and get a response.

        Args:
            user_message: What the caller said (transcribed text).

        Returns:
            The AI's response text.
        """
        # Add the caller's message to history
        self.messages.append({"role": "user", "content": user_message})

        # Call OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=150,  # Keep responses short for phone calls
            temperature=0.7,  # Natural but not too creative
        )

        # Extract the AI's response
        ai_message = response.choices[0].message.content.strip()

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=summary_messages,
            max_tokens=60,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def detect_language(self, text: str) -> str:
        """
        Simple language detection: Tamil or English.

        Checks if the text contains Tamil Unicode characters.
        For romanized Tamil, defaults to English (AI handles this via prompt).

        Args:
            text: The text to check.

        Returns:
            'ta' for Tamil, 'en' for English.
        """
        for char in text:
            # Tamil Unicode block: U+0B80 to U+0BFF
            if "\u0B80" <= char <= "\u0BFF":
                return "ta"
        return "en"

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
