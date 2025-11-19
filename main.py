import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests

from database import db, create_document, get_documents
from schemas import FitnessProfile, Plan, SavedPlan

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "AI Fitness Coach Backend Running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "Unknown"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response

@app.post("/api/generate-plan")
def generate_plan(profile: FitnessProfile):
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    result = _call_llm_for_plan(llm_provider, profile)
    return result

@app.post("/api/tts")
def tts(req: TTSRequest):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")
    voice_id = req.voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {"text": req.text, "model_id": "eleven_multilingual_v2"}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    import base64
    audio_b64 = base64.b64encode(r.content).decode("utf-8")
    return {"audio": audio_b64}

@app.post("/api/image")
def image(req: ImageRequest):
    provider = os.getenv("IMAGE_PROVIDER", "replicate").lower()
    if provider == "replicate":
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            # Fallback placeholder image if no token
            return {
                "status": "succeeded",
                "urls": {"get": None},
                "output": [f"https://source.unsplash.com/featured/?{requests.utils.quote(req.prompt)}"]
            }
        resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
            json={
                "version": "09b49f80d15e3a58f3f2f7c2b0d5af1b3c8916a49874b3d59a6f1d8b7df2d5d7",
                "input": {"prompt": req.prompt}
            }
        )
        if resp.status_code >= 300:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        return data
    else:
        # Generic fallback
        return {
            "status": "succeeded",
            "output": [f"https://source.unsplash.com/featured/?{requests.utils.quote(req.prompt)}"]
        }

@app.post("/api/save-plan")
def save_plan(plan: SavedPlan):
    try:
        plan_id = create_document("savedplan", plan)
        return {"id": plan_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plans")
def list_plans(limit: int = 20):
    try:
        docs = get_documents("savedplan", limit=limit)
        from bson import ObjectId
        from datetime import datetime
        def serialize(doc):
            out = {}
            for k, v in doc.items():
                if isinstance(v, ObjectId):
                    out[k] = str(v)
                elif isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = v
            return out
        return [serialize(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- LLM helper -----------------

def _call_llm_for_plan(provider: str, profile: FitnessProfile) -> Dict[str, Any]:
    system = (
        "You are an expert fitness and nutrition coach. Create a 7-day workout and diet plan. "
        "Be specific with sets, reps, rest times, and meal portions. Output structured JSON with keys: "
        "workout (array per day), diet (array per day), tips (array of strings)."
    )
    user = (
        f"Name: {profile.name}\nAge: {profile.age}\nGender: {profile.gender}\n"
        f"Height(cm): {profile.height_cm}\nWeight(kg): {profile.weight_kg}\n"
        f"Goal: {profile.goal}\nLevel: {profile.level}\nLocation: {profile.location}\n"
        f"Diet: {profile.diet}\nMedical: {profile.medical_history or 'None'}\n"
        f"Stress: {profile.stress_level or 'Unknown'}\n"
        "Return only JSON."
    )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Dev fallback
            return _dev_mock(profile)
        import json
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "response_format": {"type": "json_object"}
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code >= 300:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        content = r.json()["choices"][0]["message"]["content"]
        import json as _json
        try:
            return {"profile": profile.model_dump(), **(_json.loads(content))}
        except Exception:
            return {"profile": profile.model_dump(), "workout": [], "diet": [], "tips": [content]}
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return _dev_mock(profile)
        import json
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "max_tokens": 1500,
            "system": system,
            "messages": [{"role": "user", "content": user}]
        }
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        if r.status_code >= 300:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        text = r.json()["content"][0]["text"]
        import json as _json
        try:
            return {"profile": profile.model_dump(), **(_json.loads(text))}
        except Exception:
            return {"profile": profile.model_dump(), "workout": [], "diet": [], "tips": [text]}
    else:
        return _dev_mock(profile)


def _dev_mock(profile: FitnessProfile) -> Dict[str, Any]:
    return {
        "profile": profile.model_dump(),
        "workout": [
            {"day": "Day 1", "exercises": [
                {"name": "Bodyweight Squats", "sets": 3, "reps": 12, "rest": "60s"},
                {"name": "Push-ups", "sets": 3, "reps": 10, "rest": "60s"},
                {"name": "Plank", "duration": "45s", "sets": 3, "rest": "45s"}
            ]},
            {"day": "Day 2", "exercises": [
                {"name": "Walking Lunges", "sets": 3, "reps": 10, "rest": "60s"},
                {"name": "Dumbbell Rows", "sets": 3, "reps": 12, "rest": "60s"}
            ]}
        ],
        "diet": [
            {"day": "Day 1", "meals": [
                {"name": "Oats with berries", "calories": 350},
                {"name": "Grilled chicken salad", "calories": 500},
                {"name": "Greek yogurt + nuts", "calories": 200}
            ]},
            {"day": "Day 2", "meals": [
                {"name": "Egg white omelette", "calories": 300},
                {"name": "Quinoa bowl", "calories": 550}
            ]}
        ],
        "tips": ["Stay hydrated", "Walk 8k steps", "Sleep 7-8 hours"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
