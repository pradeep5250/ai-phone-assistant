# 🤖 MintX Solutions — AI Customer Care Assistant ("Yazhini")

A professional, friendly, attentive female AI customer-care executive for **MintX Solutions**, Salem, Tamil Nadu. Built with **OpenAI GPT Realtime / GPT-5.6 Reasoning**, **Vector RAG retrieval (text-embedding-3-small)**, **Sarvam.ai Indian TTS (bulbul:v3)**, **Exotel telephony**, and **MongoDB**.

---

## 📋 Architecture Flow

```
                    📞 CUSTOMER
                         │
                         ▼
              ┌─────────────────────┐
              │ STT                 │
              │ gpt-live-transcribe│
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

## 📁 Project Structure

```
ai-phone-assistant/
│
├── main.py              # FastAPI server + entry point (/realtime, /health, /test)
├── realtime_ui.py       # Standalone 1-click launcher for Realtime Voice Studio UI
├── realtime_server.py   # FastAPI WebSocket proxy & server for OpenAI Realtime API
├── realtime_chat.py     # OpenAI Realtime WebSocket streaming (supports --ui & CLI)
├── ai_assistant.py      # OpenAI AI logic (Mintx Solutions persona)
├── voice.py             # Speech-to-text, TTS (Tamil & OpenAI), Realtime config
├── voice_chat.py        # Terminal voice-to-voice conversation mode
├── chat.py              # Text terminal conversation mode
├── phone.py             # Exotel telephony integration
├── database.py          # MongoDB operations
├── templates/
│   ├── realtime.html    # Modern glassmorphism Realtime Voice Studio Web UI
│   └── agent.html       # Dedicated ElevenLabs Conversational AI Agent Studio UI
├── static/
│   ├── css/realtime.css # Dark mode glassmorphic styling & micro-animations
│   └── js/realtime.js   # 24kHz Web Audio PCM engine & reactive canvas visualizer
├── .env                 # API keys & voice configuration
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🛠️ Prerequisites

- **Python 3.11+**
- **MongoDB** (local or Atlas)
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **Exotel account** (for real phone calls — [exotel.com](https://exotel.com))

---

## 🚀 Setup

### 1. Clone / Download the project

```bash
cd ai-phone-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit the `.env` file and add your API keys:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here

SARVAM_API_KEY=your_sarvam_api_key_here
TTS_PROVIDER=sarvam
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SPEAKER=kavitha
SARVAM_TTS_LANGUAGE_CODE=ta-IN
SARVAM_TTS_PACE=1.0

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=ai_phone_assistant

EXOTEL_SID=your_exotel_sid
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_PHONE_NUMBER=your_exotel_number

PUBLIC_URL=http://your-public-url.ngrok.io
HUMAN_AGENT_NUMBER=+91XXXXXXXXXX
```

### 5. Start MongoDB

```bash
# If using local MongoDB
mongod

# Or use MongoDB Atlas (cloud) — update MONGODB_URI in .env
```

### 6. Start the server

```bash
python main.py
```

The server will start at `http://localhost:8000`.

---

## 🎙️ Realtime Voice Studio UI (OpenAI Realtime API)

A browser-based studio for voice-to-voice streaming with OpenAI Realtime API.

### 🚀 Quick Launch

You can launch the Realtime UI with any of these methods:

```bash
# Method 1: Dedicated 1-Click Launcher (opens browser automatically)
python realtime_ui.py

# Method 2: Via realtime_chat.py CLI argument
python realtime_chat.py --ui

# Method 3: Via FastAPI Server
python main.py
# Then open: http://localhost:8000/realtime
```

### ✨ Key Features:
- 🌌 **Reactive Glowing Voice Orb & Canvas Visualizer**: Organic multi-layer fluid wave ripples synchronized to mic input and assistant voice output.
- ⚡ **Sub-Second Latency 24kHz PCM Streaming**: Gapless bi-directional audio capture and playback directly in Chrome/Edge/Firefox via Web Audio API.
- 📝 **Live Streaming Transcripts**: Token-by-token typewriter effect for the AI assistant and real-time speech transcription for the user.
- 🗣️ **English & Native Tamil (தமிழ்) Support**: Fluent bilingual interactions tailored for Mintx Solutions customer discovery.
- 🎛️ **Full Session Controls**: Start/End Call, Mute Mic, Interrupt / Stop AI, Audio Mode toggle (Browser Web Audio vs Host machine hardware).
- 🎙️ **Voice Selector**: Switch between 12+ OpenAI Realtime voices (`marin`, `alloy`, `echo`, `coral`, `nova`, `shimmer`, etc.) or Tamil specialists (`pallavi`, `valluvar`).
- 💬 **Hybrid Text + Voice Input**: Type messages or click one-touch suggestion chips while still hearing spoken voice replies.
- 📜 **Conversation History Drawer**: Review timestamps, copy full transcripts to clipboard, and clear sessions.

---

## 🤖 ElevenLabs Conversational AI Agent Integration

Direct connection to ElevenLabs Conversational AI Agent (`agent_7101m22rjpzjffpab208595x1jnc`):

- **Dedicated Agent Web Page**: Visit `http://localhost:8000/agent` to talk directly with the ElevenLabs agent.
- **Embedded Studio Widget**: The official ElevenLabs ConvAI widget is embedded directly in the `/realtime` Studio UI.
- **Config API Endpoint**: `GET /api/agent/config` returns the current active agent metadata.
- **Environment Setting**: Set `ELEVENLABS_AGENT_ID=agent_7101m22rjpzjffpab208595x1jnc` in `.env` / `.env.local`.

---

## 🧪 Testing

### Step 1: Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "service": "AI Phone Call Assistant", "version": "1.0.0"}
```

### Step 2: Test AI chat (no phone needed)

```bash
curl -X POST http://localhost:8000/test/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Hello\", \"caller_number\": \"+919999999999\"}"
```

Expected response:
```json
{
  "response": "Hi! I'm an AI assistant. How can I help you today?",
  "language": "en",
  "intent": "general",
  "history": [...]
}
```

### Step 3: Test Tamil

```bash
curl -X POST http://localhost:8000/test/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Naalaikku appointment venum\", \"caller_number\": \"+919999999999\"}"
```

### Step 4: Test appointment booking

```bash
curl -X POST http://localhost:8000/test/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"I want to book an appointment for tomorrow at 10 AM\", \"caller_number\": \"+919999999999\"}"
```

### Step 5: Test human transfer

```bash
curl -X POST http://localhost:8000/test/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"I want to speak to a person\", \"caller_number\": \"+919999999999\"}"
```

### Step 6: Reset conversation

```bash
curl -X POST http://localhost:8000/test/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"caller_number\": \"+919999999999\"}"
```

### Step 7: Check API docs

Open `http://localhost:8000/docs` in your browser for interactive Swagger docs.

---

## 📞 Connecting Exotel (Real Phone Calls)

### 1. Get a public URL

For development, use [ngrok](https://ngrok.com):

```bash
ngrok http 8000
```

Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`) and update `.env`:

```env
PUBLIC_URL=https://abc123.ngrok.io
```

### 2. Configure Exotel

1. Log in to your [Exotel dashboard](https://my.exotel.com)
2. Go to **App Bazaar** → **Create New App**
3. Set the **Webhook URL** to: `https://your-ngrok-url/exotel/webhook`
4. Set the **Status Callback URL** to: `https://your-ngrok-url/exotel/status`
5. Assign your +91 virtual number to this app

### 3. Test with a real call

Call your Exotel +91 number and talk to the AI!

---


## 🗄️ MongoDB Collections

| Collection      | Purpose                        |
|-----------------|--------------------------------|
| `calls`         | Call records (SID, duration)   |
| `conversations` | Full chat history per call     |
| `customers`     | Caller information             |
| `appointments`  | Booked appointments            |

---

## 🌐 API Endpoints

| Method | Endpoint           | Description                          |
|--------|---------------------|--------------------------------------|
| GET    | `/health`           | Health check                         |
| POST   | `/test/chat`        | Text-based AI test (no phone)        |
| POST   | `/test/reset`       | Reset a test conversation            |
| POST   | `/exotel/webhook`   | Incoming call webhook (Exotel)       |
| POST   | `/exotel/gather`    | Process caller speech (Exotel)       |
| POST   | `/exotel/status`    | Call status callback (Exotel)        |
| GET    | `/audio/{filename}` | Serve TTS audio files                |
| GET    | `/docs`             | Interactive API documentation        |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| MongoDB connection error | Make sure MongoDB is running: `mongod` |
| OpenAI error | Check your API key in `.env` |
| Exotel not receiving calls | Check ngrok is running and URLs are correct |
| No speech detected | Ensure the caller's audio is clear |
| Tamil not working | Whisper auto-detects language; try speaking clearly |

---

## 🔒 Security Notes

- API keys are stored in `.env` (never committed to Git)
- The `.gitignore` file excludes `.env`
- Validate Exotel webhook signatures in production
- Do not expose MongoDB publicly
- The AI always identifies itself as an AI, not a human

---

## 📝 License

This project is for educational purposes (college project / prototype).
#   a i - p h o n e - 
 
 
