# 🤖 AI Phone Call Assistant

A simple Python AI phone assistant that receives real Indian phone calls via **Exotel**, understands the caller using **OpenAI**, responds with a natural AI voice, and saves everything to **MongoDB**.

Built as a college project / prototype.

---

## 📋 Architecture

```
                 📞 CUSTOMER
                     │
                     ▼
             🇮🇳 +91 EXOTEL NUMBER
                     │
                     ▼
              phone.py (webhooks)
                     │
                     ▼
              voice.py (STT/TTS)
                     │
                     ▼
           ai_assistant.py (OpenAI)
                     │
                     ▼
             Natural AI Voice
                     │
                     ▼
                 📞 CUSTOMER

                     │
                     ▼
            database.py (MongoDB)
```

---

## 📁 Project Structure

```
ai-phone-assistant/
│
├── main.py              # FastAPI server + entry point
├── ai_assistant.py      # OpenAI AI logic
├── voice.py             # Speech-to-text, TTS (Tamil & OpenAI), Realtime config
├── voice_chat.py        # Terminal voice-to-voice conversation mode
├── realtime_chat.py     # OpenAI Realtime WebSocket streaming (marin voice, 24kHz PCM)
├── chat.py              # Text terminal conversation mode
├── phone.py             # Exotel telephony integration
├── database.py          # MongoDB operations
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
