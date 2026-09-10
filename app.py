import os
import time
import uuid
from datetime import timedelta

from flask import Flask, jsonify, request, send_from_directory
from livekit import api

app = Flask(__name__)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
AGENT_NAME = os.getenv("LIVEKIT_LAPTOP_AGENT_NAME", "aarna-sania-laptop-test")
TOKEN_TTL_MINUTES = int(os.getenv("LIVEKIT_LAPTOP_TOKEN_TTL_MINUTES", "30"))

@app.get("/")
def index():
    return send_from_directory(".", "livekit_laptop_client.html")

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/api/session")
async def create_session():
    data = request.get_json(silent=True) or {}
    room_name = f"aarna-web-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    identity = f"web-tester-{uuid.uuid4().hex[:8]}"

    metadata = {
        "partner_name": str(data.get("partner_name", "")).strip(),
        "contact_name": str(data.get("contact_name", "")).strip(),
        "category": str(data.get("category", "")).strip(),
        "company_synopsis": str(data.get("company_synopsis", "")).strip(),
        "digitisation": str(data.get("digitisation", "semi")).strip() or "semi",
    }

    lkapi = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=__import__("json").dumps(metadata),
            )
        )
    finally:
        await lkapi.aclose()

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .with_ttl(timedelta(minutes=TOKEN_TTL_MINUTES))
        .to_jwt()
    )

    return jsonify(serverUrl=LIVEKIT_URL, token=token, room=room_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
