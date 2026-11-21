"""
database.py — MongoDB Operations

Handles all database interactions for the AI Phone Call Assistant.
Uses MongoDB to store calls, conversations, customers, and appointments.
"""

import os
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB Connection ──────────────────────────────────────────────

# Global database reference (initialized by connect_db)
db = None


def connect_db():
    """
    Connect to MongoDB and return the database instance.
    Call this once at application startup.
    """
    global db

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "ai_phone_assistant")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        db = client[database_name]

        # Test connection
        client.admin.command("ping")
        print(f"[OK] Connected to MongoDB: {database_name}")
        return db
    except Exception as e:
        print(f"[!] MongoDB connection error: {e}")
        return None


def get_db():
    """Get active MongoDB instance, auto-connecting if needed."""
    global db
    if db is None:
        return connect_db()
    return db


# ── Call Operations ─────────────────────────────────────────────────

def save_call(call_data: dict) -> str:
    """
    Save a call record to the 'calls' collection.

    Example call_data:
    {
        "call_sid": "exotel_call_id",
        "caller_number": "+91XXXXXXXXXX",
        "status": "completed",
        "duration": 120,
        "language": "en",
        "summary": "Customer asked about an appointment."
    }
    """
    call_data["created_at"] = datetime.now(timezone.utc)
    result = db.calls.insert_one(call_data)
    print(f"[CALL] Call saved: {result.inserted_id}")
    return str(result.inserted_id)


def update_call(call_sid: str, update_data: dict):
    """Update an existing call record by Exotel call SID."""
    update_data["updated_at"] = datetime.now(timezone.utc)
    db.calls.update_one(
        {"call_sid": call_sid},
        {"$set": update_data}
    )
    print(f"[CALL] Call updated: {call_sid}")


def get_call(call_sid: str) -> dict:
    """Get a call record by Exotel call SID."""
    return db.calls.find_one({"call_sid": call_sid})


# ── Conversation Operations ────────────────────────────────────────

def save_conversation(conversation_data: dict) -> str:
    """
    Save a conversation to the 'conversations' collection.

    Example conversation_data:
    {
        "call_sid": "exotel_call_id",
        "caller_number": "+91XXXXXXXXXX",
        "messages": [
            {"role": "user", "message": "I need an appointment."},
            {"role": "assistant", "message": "Sure! What day would you prefer?"}
        ]
    }
    """
    conversation_data["created_at"] = datetime.now(timezone.utc)
    result = db.conversations.insert_one(conversation_data)
    print(f"[CHAT] Conversation saved: {result.inserted_id}")
    return str(result.inserted_id)


def update_conversation(call_sid: str, messages: list):
    """Update conversation messages for a call."""
    db.conversations.update_one(
        {"call_sid": call_sid},
        {
            "$set": {
                "messages": messages,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


# ── Customer Operations ────────────────────────────────────────────

def save_customer(customer_data: dict) -> str:
    """
    Save or update a customer in the 'customers' collection.
    Uses phone number as a unique identifier (upsert).

    Example customer_data:
    {
        "phone_number": "+91XXXXXXXXXX",
        "name": "Raj",
        "language": "ta",
        "total_calls": 1
    }
    """
    phone = customer_data.get("phone_number")
    result = db.customers.update_one(
        {"phone_number": phone},
        {
            "$set": customer_data,
            "$setOnInsert": {"first_call_at": datetime.now(timezone.utc)},
            "$inc": {"total_calls": 1},
            "$currentDate": {"last_call_at": True},
        },
        upsert=True,
    )
    print(f"[USER] Customer saved/updated: {phone}")
    return str(result.upserted_id or phone)


def get_customer(phone_number: str) -> dict:
    """Get a customer record by phone number."""
    return db.customers.find_one({"phone_number": phone_number})


# ── Appointment Operations ──────────────────────────────────────────

def save_appointment(appointment_data: dict) -> str:
    """
    Save an appointment to the 'appointments' collection.

    Example appointment_data:
    {
        "caller_number": "+91XXXXXXXXXX",
        "date": "2024-01-15",
        "time": "10:00 AM",
        "purpose": "General consultation",
        "status": "booked"
    }
    """
    appointment_data["created_at"] = datetime.now(timezone.utc)
    appointment_data["status"] = appointment_data.get("status", "booked")
    database = get_db()
    if database is None:
        print("[!] DB offline: appointment not saved")
        return "offline"
    result = database.appointments.insert_one(appointment_data)
    print(f"[APPT] Appointment saved: {result.inserted_id}")
    return str(result.inserted_id)


# ── Customer Lead Operations (Immediate Real-time Per-Answer Updates) ─────────

def save_or_update_lead(call_id: str, fields: dict) -> dict:
    """
    Immediately creates or updates a customer lead record in MongoDB after every answer.
    Matches schema:
    {
      "name": "",
      "business_name": "",
      "service": "",
      "phone": "",
      "requirement": "",
      "language": "",
      "created_at": "",
      "status": "new"
    }
    """
    database = get_db()
    clean_fields = {}
    valid_keys = {"name", "business_name", "service", "phone", "requirement", "language", "status"}
    for k, v in fields.items():
        if k in valid_keys and v is not None and str(v).strip():
            clean_fields[k] = str(v).strip()

    if not clean_fields:
        return {}

    now_iso = datetime.now(timezone.utc).isoformat()
    clean_fields["updated_at"] = now_iso

    if database is None:
        print(f"💾 [LEAD MEMORY ONLY] Call {call_id}: Updated {list(clean_fields.keys())}")
        return clean_fields

    try:
        if call_id:
            query = {"call_id": call_id}
        elif "phone" in clean_fields and clean_fields["phone"]:
            query = {"phone": clean_fields["phone"]}
        else:
            return clean_fields

        default_schema = {
            "call_id": call_id,
            "name": "",
            "business_name": "",
            "service": "",
            "requirement": "",
            "language": "ta",
            "status": "new",
            "created_at": now_iso,
        }
        # Only include phone in insert if actually provided to avoid unique constraint collisions on empty strings
        if "phone" in clean_fields and clean_fields["phone"]:
            default_schema["phone"] = clean_fields["phone"]

        set_on_insert = {k: v for k, v in default_schema.items() if k not in clean_fields}

        update_payload = {
            "$set": clean_fields,
            "$setOnInsert": set_on_insert
        }

        database.leads.update_one(query, update_payload, upsert=True)
        try:
            cust_query = {"phone": clean_fields["phone"]} if clean_fields.get("phone") else query
            database.customers.update_one(cust_query, update_payload, upsert=True)
        except Exception:
            pass

        for k, v in clean_fields.items():
            if k != "updated_at":
                print(f"[DB SAVED] {k} = {v}")

        return clean_fields
    except Exception as e:
        print(f"[!] Error updating lead in MongoDB: {e}")
        return clean_fields


def get_lead(call_id: str) -> dict:
    """Retrieve the customer lead record by call_id or phone."""
    database = get_db()
    if database is None:
        return {}
    lead = database.leads.find_one({"call_id": call_id}) or database.customers.find_one({"call_id": call_id})
    return lead or {}
