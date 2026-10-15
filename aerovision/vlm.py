"""Optional generative VLM narrative over an OpenAI-compatible vision endpoint (GPT-4o, Qwen2-VL via vLLM, LLaVA via
Ollama). The model is asked for strict JSON matching `VLMFinding`; anything unparsable or below 0.6 confidence is
treated as unverified and the deterministic detector decides (SOP rule). The mock provider needs no key."""
from __future__ import annotations
import base64, io, json, os
from pydantic import BaseModel, Field, ValidationError
from PIL import Image

PROMPT = ("You are an aerial inspection analyst. Return ONLY JSON: {\"summary\": str, \"anomaly_count\": int, "
          "\"max_delta_c_estimate\": number, \"confidence\": number in [0,1]}. Be conservative.")


class VLMFinding(BaseModel):
    summary: str
    anomaly_count: int = Field(ge=0)
    max_delta_c_estimate: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def describe(img: Image.Image, detector_summary: dict, provider: str | None = None, model: str | None = None) -> VLMFinding:
    provider = provider or os.environ.get("AEROVISION_PROVIDER", "mock")
    if provider == "mock":
        n = detector_summary.get("n_anomalies", 0); d = detector_summary.get("max_delta_c", 0.0)
        return VLMFinding(summary=f"Deterministic detector found {n} anomalies (max Δ {d:.1f} °C).", anomaly_count=n,
                          max_delta_c_estimate=d, confidence=0.95 if n == 0 or d >= 10 else 0.5)
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
    msg = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
        {"type": "text", "text": f"Detector context: {json.dumps(detector_summary)}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}}]}]
    raw = client.chat.completions.create(model=model or os.environ.get("AEROVISION_MODEL", "gpt-4o-mini"), messages=msg,
                                         temperature=0, response_format={"type": "json_object"}).choices[0].message.content
    try:
        return VLMFinding.model_validate_json(raw)
    except ValidationError:
        return VLMFinding(summary="VLM output failed schema validation", anomaly_count=0, max_delta_c_estimate=0, confidence=0.0)
