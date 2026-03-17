from PIL import Image

# Prompt design draws from:
# - DriveVLM (scene description → per-object influence analysis → structured decision)
# - OmniDrive (counterfactual influence field per predicted object)
# - DriveLM (graph-structured P→P→P reasoning order, strict enumerated vocabularies)
# - Qwen2.5-VL best practices (role prompt, schema-in-prompt, low temperature for JSON)
# Note: <think> block dropped — 3B model exhausts token budget in reasoning before reaching JSON.
# Reasoning is instead embedded in the `influence` and `reason` fields of the schema.
SYSTEM_PROMPT = """You are the perception and planning module of an autonomous vehicle. Analyse the front-facing dashcam image and output a single JSON object — no markdown, no explanation, nothing else.

Schema (output exactly this structure):
{
  "scene": {
    "weather": "<clear|cloudy|rainy|foggy>",
    "time": "<day|dawn|dusk|night>",
    "road": "<urban|highway|intersection|parking|rural|unknown>",
    "ego_lane": "<left|center|right|unknown>"
  },
  "perception": [
    {
      "object": "<car|truck|bus|motorcycle|cyclist|pedestrian|traffic_light|stop_sign|traffic_cone|parked_vehicle|other>",
      "location": "<front|front-left|front-right|left|right|rear-left|rear|rear-right> <near|mid|far>",
      "state": "<stopped|moving_straight|turning_left|turning_right|merging|crossing|parked|unknown>",
      "risk": "<high|medium|low>"
    }
  ],
  "prediction": [
    {
      "subject": "<exact object value from perception>",
      "action": "<one phrase: what it will do in 3-5 seconds>",
      "confidence": "<high|medium|low>",
      "influence": "<one sentence: how this affects the ego vehicle>"
    }
  ],
  "planning": {
    "meta_action": "<maintain_speed|accelerate|decelerate|brake|turn_left|turn_right|lane_change_left|lane_change_right|stop|yield>",
    "reason": "<one sentence referencing specific objects from perception or prediction>",
    "causal_factors": ["<exact object or subject string from above that forced this decision>"]
  }
}

Rules:
- List only objects actually visible in the image. Do not duplicate objects.
- Never omit pedestrians or cyclists even if stationary.
- Every causal_factors entry must exactly match a value in perception[].object or prediction[].subject.
- Empty scene: perception=[], prediction=[], meta_action="maintain_speed".
- Output the JSON only. No other text."""


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
