"""
chat.py — Interactive AI Terminal Conversation

Run this script to talk to the AI assistant directly in your terminal
without needing any phone numbers, Exotel, or external services.

Usage:
    python chat.py
"""

import sys
import os
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from ai_assistant import AIAssistant


def main():
    print("=" * 60)
    print("       AI Phone Assistant — Terminal Conversation Mode")
    print("=" * 60)
    print("Commands:")
    print("  exit / quit   - End the conversation")
    print("  /reset        - Clear history and start a new conversation")
    print("  /speak        - Hear the assistant speak its last response aloud")
    print("  /summary      - Summarize current conversation")
    print("  /history      - View full conversation history")
    print("=" * 60)
    print("Supported Languages: English & Tamil")

    assistant = AIAssistant(call_id="terminal_session")
    print(f"🔊 Main Agent Voice ID: {assistant.voice_id} ({assistant.voice_model})")
    print(f"⚡ TTS Provider:        {assistant.tts_provider}\n")
    print("AI Assistant: Welcome to MintX Solutions. How can I help?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q", "/exit"):
                print("\n[AI Assistant]: Thank you! Unga details save panniten. Team contact pannuvanga.")
                break

            if user_input.lower() == "/reset":
                assistant = AIAssistant(call_id="terminal_session")
                print("\n[System]: Conversation reset. Starting fresh.")
                print("AI Assistant: Welcome to MintX Solutions. How can I help?")
                continue

            if user_input.lower() == "/summary":
                summary = assistant.get_summary()
                print(f"\n[Summary]: {summary}")
                continue

            if user_input.lower() == "/history":
                history = assistant.get_history()
                print("\n--- Conversation History ---")
                for msg in history:
                    role = "You" if msg["role"] == "user" else "AI"
                    print(f"[{role}]: {msg['message']}")
                print("----------------------------")
                continue

            # Detect language and intent
            lang = assistant.detect_language(user_input)
            intent = assistant.detect_intent(user_input)

            # Get AI response
            response = assistant.chat(user_input)

            lang_label = "Tamil" if lang == "ta" else "English"
            print(f"AI Assistant ({lang_label} | Intent: {intent}): {response}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
