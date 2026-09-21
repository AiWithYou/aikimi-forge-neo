"""Isolated typesafe-sdk 0.7.0 worker. stdout is one JSON response only."""

from __future__ import annotations

import importlib.metadata
import json
import sys


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Input limit")
        payload = json.loads(raw)
        if importlib.metadata.version("typesafe-sdk") != "0.7.0":
            raise ValueError("Unexpected SDK version")
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

        state, allowed = payload["state"], payload["allowed"]
        if not isinstance(state, dict) or not isinstance(allowed, dict) or not 1 <= len(allowed) <= 256:
            raise ValueError("Invalid request")
        timeout = float(payload["timeout"])
        if not 0.5 <= timeout <= 20:
            raise ValueError("Invalid timeout")
        questions = {
            layer: Choice(
                instructions=f"Choose attention KEEP percentage for layer {layer} to trade compute cost against estimated visual importance. Follow state.constraints and compare the supplied measurements with peer layers. Pixel-identical reproduction is NOT the objective; visually good, prompt-consistent generation with less compute is. Use lower keep for relatively weak or stable contributions, middle keep for typical or ambiguous contributions, and maximum keep for unusually strong or rapidly changing contributions. Do not default to maximum solely because these are proxy measurements. Do not invent image content, measured quality scores, or force a quota or variation. Return only a supplied choice.",
                criteria={
                    str(float(keep)): f"Keep {keep}% of eligible key blocks; larger values are more conservative"
                    for keep in choices
                },
            )
            for layer, choices in allowed.items()
        }
        with TypeSafeClient(
            model="jev-1.13.0", retry=RetryPolicy(max_retries=0), timeout=timeout, base_url="https://api.typesafe.ai"
        ) as client:
            response = client.system_one(state=state, questions=questions)
        answer = {
            "decisions": {
                str(layer): {"choice": choice.choice, "confidence": choice.confidence}
                for layer, choice in response.choices.items()
            }
        }
        sys.stdout.write(json.dumps(answer, allow_nan=False))
        return 0
    except Exception:
        # HTTP errors may contain prompts or credentials. Never print the exception.
        sys.stdout.write('{"error":"sdk_failure"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
