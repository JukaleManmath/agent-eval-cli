import os

from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Demo Customer Support Agent")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Session history: session_id -> list of messages
sessions: dict[str, list] = {}

SYSTEM_PROMPT = """
You are a helpful customer support agent for a restaurant booking service
called TableEase.

You can help customers with:
- Making table reservations (ask for name, party size, date and time)
- Cancellation policy (free up to 24 hours before, 50 percent fee same-day)
- Refunds (processed in 3-5 business days)
- Escalating complaints to a human manager

Rules:
- Never compare yourself to or mention competitor services
- Keep responses concise (2-3 sentences max)
- Always confirm bookings with a reference number like BK-XXXX
- If a customer is angry or insists on a human, escalate with reference ESC-XXXX
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    history = sessions.setdefault(req.session_id, [])

    history.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.3,
        max_tokens=200,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    return {"response": reply}
