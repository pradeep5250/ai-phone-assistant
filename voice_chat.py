"""
voice_chat.py — Full Voice-to-Voice AI Conversation in Terminal

Use your computer microphone to speak and speakers to listen to the AI assistant!
No phone numbers, Exotel, or external services needed.

Usage:
    python voice_chat.py
"""

import sys
import os
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from ai_assistant import AIAssistant
from voice import (
    record_audio_press_enter, speech_to_text, speak, text_to_speech,
    DEFAULT_VOICE, SUPPORTED_VOICES, REALTIME_SESSION_CONFIG
)
from instructions import get_active_starting_greeting


def main():
    sarvam_speaker = os.getenv("SARVAM_TTS_SPEAKER", "kavitha")
    selected_voice = sarvam_speaker
    mode = os.getenv("OPENAI_MODE", "realtime")

    print("=" * 65)
    print(f"  🎙️  AI Assistant — Voice Conversation Mode ({mode.upper()}) 🎙️")
    print(f"  🔊 Main Agent Voice: {selected_voice} (Sarvam AI bulbul:v3)")
    print("=" * 65)
    print("Instructions:")
    print("  1. Press [ENTER] to start recording your voice.")
    print("  2. Speak clearly into your microphone in English or Tamil.")
    print("  3. Press [ENTER] again when you finish speaking.")
    print("  4. The AI will transcribe your speech, answer, and speak back!")
    print("Commands (type instead of pressing Enter):")
    print("  exit / quit     - End the voice conversation")
    print("  /tamil          - Switch to native Tamil voice (Pallavi)")
    print("  /english        - Switch to English/Realtime voice (Marin)")
    print("  /voice <name>   - Change voice (pallavi, valluvar, marin, alloy, nova, etc.)")
    print("  /mode           - View current mode and Realtime session config")
    print("  /mute           - Toggle AI audio voice reply ON/OFF")
    print("  /summary        - Summarize conversation so far")
    print("  /reset          - Start a new conversation")
    print("=" * 65)

    assistant = AIAssistant(call_id="voice_terminal_session")
    voice_output_enabled = True

    initial_greeting = get_active_starting_greeting()
    print(f"\n[AI Assistant]: {initial_greeting}")
    try:
        speak(initial_greeting, voice=selected_voice, play_aloud=voice_output_enabled)
    except Exception as e:
        print(f"[Note: Audio playback not available: {e}]")

    while True:
        try:
            print("\n-----------------------------------------------------------")
            prompt_input = input("👉 Press [ENTER] to speak (or type command/text): ").strip()

            if prompt_input.lower() in ("exit", "quit", "q", "/exit"):
                farewell = "நன்றி! வணக்கம்." if "ta" in selected_voice or selected_voice in ("pallavi", "valluvar") else "Thank you for chatting! Goodbye."
                print(f"\n[AI Assistant]: {farewell}")
                if voice_output_enabled:
                    speak(farewell, voice=selected_voice)
                break

            if prompt_input.lower() == "/tamil":
                selected_voice = "pallavi"
                print(f"\n[System]: Switched to Native Tamil Voice: 'pallavi' (ta-IN-PallaviNeural).")
                continue

            if prompt_input.lower() == "/english":
                selected_voice = "marin"
                print(f"\n[System]: Switched to English/Realtime Voice: 'marin'.")
                continue

            if prompt_input.lower().startswith("/voice") or prompt_input.lower() == "/voices":
                parts = prompt_input.split()
                if len(parts) > 1:
                    new_voice = parts[1].lower()
                    selected_voice = new_voice
                    print(f"\n[System]: Voice changed to '{selected_voice}'.")
                else:
                    print(f"\n[System]: Current voice is '{selected_voice}'.")
                    print("  Tamil Voices:   pallavi (Female), valluvar (Male), saranya, kumar")
                    print(f"  Standard Voices: {', '.join(SUPPORTED_VOICES)}")
                continue

            if prompt_input.lower() == "/mode":
                import json
                print(f"\n[Mode]: {mode}")
                print("[Realtime Session Config]:")
                print(json.dumps(REALTIME_SESSION_CONFIG, indent=2))
                continue

            if prompt_input.lower() == "/reset":
                assistant = AIAssistant(call_id="voice_terminal_session")
                print("\n[System]: Conversation reset.")
                reset_msg = "Hello again! How can I assist you?"
                print(f"[AI Assistant]: {reset_msg}")
                if voice_output_enabled:
                    speak(reset_msg, voice=selected_voice)
                continue

            if prompt_input.lower() == "/summary":
                summary = assistant.get_summary()
                print(f"\n[Summary]: {summary}")
                continue

            if prompt_input.lower() == "/mute":
                voice_output_enabled = not voice_output_enabled
                status = "ENABLED (AI will speak)" if voice_output_enabled else "MUTED (Text only)"
                print(f"\n[System]: Voice audio reply is now {status}.")
                continue

            user_text = ""

            # If user typed something directly (text input fallback)
            if prompt_input:
                user_text = prompt_input
                print(f"\n[You (Typed)]: {user_text}")
            else:
                # Record from microphone
                audio_bytes = record_audio_press_enter()
                if not audio_bytes or len(audio_bytes) < 1000:
                    print("⚠️  No audio detected. Please try again.")
                    continue

                print("⏳ Transcribing audio with OpenAI Whisper...")
                user_text = speech_to_text(audio_bytes)

                if not user_text.strip():
                    print("⚠️  Could not recognize speech. Please try speaking again.")
                    continue

                print(f"\n🗣️  [You (Voice)]: \"{user_text}\"")

            # Detect language and intent
            lang = assistant.detect_language(user_text)
            intent = assistant.detect_intent(user_text)
            lang_label = "Tamil" if lang == "ta" else "English"

            print(f"   (Detected Language: {lang_label} | Intent: {intent})")
            print("⏳ AI is thinking...")

            # Get AI response
            response = assistant.chat(user_text)
            print(f"\n🤖 [AI Assistant]: {response}")

            # Speak response aloud using ElevenLabs voice ID
            if voice_output_enabled:
                print(f"🔊 Playing AI voice response ({selected_voice})...")
                speak(response, voice=selected_voice, play_aloud=True)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
