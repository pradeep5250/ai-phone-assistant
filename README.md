# 🤖 MintX Solutions — AI Customer Care Assistant ("Yazhini")

> **A professional, friendly, attentive female AI customer-care executive for MintX Solutions, Salem, Tamil Nadu. Built with OpenAI GPT Realtime / GPT-5.6 Reasoning, Vector RAG retrieval (text-embedding-3-small), Sarvam.ai Indian TTS (bulbul:v3), Exotel telephony, and MongoDB.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT%20Realtime-black?logo=openai)
![Sarvam](https://img.shields.io/badge/TTS-Sarvam.ai%20bulbul:v3-orange)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?logo=mongodb)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Architecture Flow

```text
                    📞 CUSTOMER
                         │
                         ▼
              ┌─────────────────────┐
              │ STT                 │
              │ gpt-live-transcribe │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ LANGUAGE DETECTION  │
              │ Tamil / Tunglish /  │
              │ English             │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   GPT-5.6 /         │
              │   GPT Realtime      │
              │   Reasoning         │
              └──────────┬──────────┘
                         │
                 User question
                         │
                         ▼
              ┌─────────────────────┐
              │       RAG           │
              │                     │
              │ Query Embedding     │
              │        ↓            │
              │ Vector Database     │
              │        ↓            │
              │ Relevant MintX data │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   LLM RESPONSE      │
              │                     │
              │ MintX Knowledge +   │
              │ Conversation        │
              │ Context + Rules     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ TTS                 │
              │ Sarvam.ai bulbul:v3 │
              │ Yazhini (Kavitha)   │
              └──────────┬──────────┘
                         │
                         ▼
                    🔊 CUSTOMER
```

---

## 🌟 Key Features

1. **Strict Language Rule & Mirroring**:
   - **English**: Responds in 100% pure English.
   - **Tunglish**: Responds in natural Roman Tunglish (e.g. *Website-ku price features based panni vary aagum. Ungalukku normal business website venuma illa e-commerce website venuma?*).
   - **Tamil**: Responds in natural spoken Tamil script.
   - Zero language mixing within turns.

2. **RAG Knowledge Base (`rag.py`)**:
   - 12 comprehensive MintX Solutions knowledge domains (Salem location, pricing, web, e-commerce, mobile apps, software, SEO, QR, 3D printing, timelines, support).
   - Semantic vector retrieval using OpenAI `text-embedding-3-small` with local caching + lexical keyword matching.
   - Never hallucinates pricing or services; falls back gracefully to human handoff.

3. **Requirement Collection & Confirmation Flow**:
   - Gathers Name, Business Name, Service, Phone Number, and Requirement one question at a time.
   - Repeats collected details for explicit customer confirmation before marking as `confirmed` in MongoDB.

4. **Real-Time Voice Studio & Exotel Telephony**:
   - Web Audio Studio at `/realtime` with sub-second latency and live audio visualization.
   - Full Exotel call handling at `/exotel/webhook` and `/exotel/gather`.

---

## 📂 Project Structure

```
ai-phone-assistant/
├── main.py              # FastAPI server + entry point (realtime, health, test)
├── realtime_ui.py       # Standalone 1-click launcher for Realtime Voice Studio UI
├── realtime_server.py   # FastAPI WebSocket proxy & server for OpenAI Realtime API
├── realtime_chat.py     # OpenAI Realtime WebSocket streaming (supports UI & CLI)
├── ai_assistant.py      # OpenAI AI logic (Mintx Solutions persona)
├── voice.py             # Speech-to-text, TTS (Tamil & OpenAI), Realtime config
├── voice_chat.py        # Terminal voice-to-voice conversation mode
├── chat.py              # Text terminal conversation mode
├── phone.py             # Exotel telephony integration
├── database.py          # MongoDB operations
├── templates/           # Modern glassmorphism Realtime Studio Web UI
│   ├── realtime.html
│   └── agent.html       # Dedicated ElevenLabs Conversational AI Agent Studio UI
├── static/              # Dark mode glassmorphic styling & micro-animations
│   ├── css/realtime.css
│   └── js/realtime.js   # 24kHz Web Audio PCM engine & reactive canvas visualizer
├── .env.example         # Template for environment configuration
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
└── README.md            # System documentation
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+
- MongoDB (running locally or MongoDB Atlas)
- OpenAI API Key
- Sarvam.ai API Key (for Indian voice synthesis)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/pradeep5250/ai-phone-assistant.git
cd ai-phone-assistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` using `.env.example`:
```ini
OPENAI_API_KEY=your_openai_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=mintx_voice_ai
EXOTEL_SID=your_exotel_sid_here
EXOTEL_TOKEN=your_exotel_token_here
```

### 3. Launch the Application

#### Option A: Web Audio Studio (Browser)
```bash
python realtime_ui.py
```
Open **`http://localhost:8000/realtime`** in your browser.

#### Option B: Terminal Voice Chat
```bash
python voice_chat.py
```

#### Option C: Terminal Text Chat
```bash
python chat.py
```

---

## 🔒 Security Best Practices

- API keys and telephony credentials are stored in `.env` (excluded by `.gitignore`).
- Webhook endpoints validate caller signatures in production.
- Database runs with strict schema validations on MongoDB.

---

## 👤 Author

Developed by **[Pradeep S](https://github.com/pradeep5250)**
