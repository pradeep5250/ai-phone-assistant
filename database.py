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

    client = MongoClient(uri)
    db = client[database_name]

    # Test connection
    client.admin.command("ping")
    print(f"[OK] Connected to MongoDB: {database_name}")

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
    result = db.appointments.insert_one(appointment_data)
    print(f"[APPT] Appointment saved: {result.inserted_id}")
    return str(result.inserted_id)
