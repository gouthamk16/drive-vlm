from PIL import Image

SYSTEM_PROMPT = """You are a driving scene analyst. Given a dashcam image, respond in strict JSON with this exact structure:

{
  "perception": [
    {"object": "<what>", "location": "<where in frame: left/center/right/far-left/far-right + near/mid/far>", "state": "<what it is doing>"}
  ],
  "prediction": [
    {"subject": "<object from perception>", "action": "<what it will likely do next>", "confidence": "<high|medium|low>"}
  ],
  "planning": {
    "action": "<what the ego vehicle should do: e.g. maintain speed, brake, change lane left>",
    "reason": "<one sentence explanation>",
    "causal_factors": ["<specific perceived or predicted element that drove this decision>"]
  }
}

Rules:
- Output valid JSON only. No markdown, no explanation outside the JSON.
- perception must list every traffic-relevant object visible.
- causal_factors must reference objects or predictions already listed above.
- If the scene is clear, perception may be an empty list and planning.action should be "maintain speed"."""


def build_messages(image_path: str) -> list:
    img = Image.open(image_path).convert("RGB")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Analyse this driving scene."},
            ],
        },
    ]
