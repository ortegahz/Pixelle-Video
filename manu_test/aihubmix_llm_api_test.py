import argparse
import json
import os
from typing import Any, Optional

import httpx
from openai import OpenAI


def _extract_text_from_message(message: Any) -> Optional[str]:
    """
    Robustly extract text from OpenAI-compatible response message.
    Handles:
      - message.content is str
      - message.content is list of parts
      - some gateways return message.content as empty string but put text elsewhere
    """
    if message is None:
        return None

    content = getattr(message, "content", None)

    # 1) Standard: string
    if isinstance(content, str):
        return content

    # 2) Parts array
    if isinstance(content, list):
        parts_text = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if "text" in p and isinstance(p["text"], str):
                parts_text.append(p["text"])
            elif "content" in p and isinstance(p["content"], str):
                parts_text.append(p["content"])
        if parts_text:
            return "".join(parts_text)

    # 3) Fallback fields
    for k in ("output_text", "text", "response"):
        v = getattr(message, k, None)
        if isinstance(v, str) and v.strip():
            return v

    return None


def _build_messages(prompt: str, use_gemini_parts: bool):
    if use_gemini_parts:
        # Gemini-style parts (works for some OpenAI-compatible gateways)
        return [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
    return [{"role": "user", "content": prompt}]


def main():
    parser = argparse.ArgumentParser(
        description="Test aihubmix OpenAI-compatible LLM (gemini-friendly parts supported)"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.inferera.com/v1",
        help="OpenAI-compatible base_url, e.g. https://.../v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AIHUBMIX_API_KEY", ""),
        help="API key (or set env AIHUBMIX_API_KEY). Do not hardcode secrets.",
    )
    parser.add_argument(
        "--model",
        # default=os.getenv("AIHUBMIX_MODEL", "gemini-3.1-pro-preview"),
        # default=os.getenv("AIHUBMIX_MODEL", "gpt-5.4-nano"),
        default=os.getenv("AIHUBMIX_MODEL", "glm-4.7"),
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("AIHUBMIX_PROMPT", "你好！"),
        help="Prompt text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("AIHUBMIX_TEMPERATURE", "0.7")),
        help="Temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("AIHUBMIX_MAX_TOKENS", "512")),
        help="max_tokens",
    )
    parser.add_argument(
        "--use-gemini-parts",
        action="store_true",
        help="Send Gemini-style messages[].content parts",
    )
    parser.add_argument(
        "--auto-gemini-parts",
        action="store_true",
        help="Automatically enable parts when model name contains 'gemini' (recommended)",
    )
    parser.add_argument(
        "--debug-request",
        action="store_true",
        help="Print request body (no secrets)",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print raw response summary when content extraction is empty",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing api-key. Set --api-key or env AIHUBMIX_API_KEY")
    if not args.base_url:
        raise RuntimeError("Missing base-url.")

    # Avoid socks:// proxy scheme issues in some envs
    http_client = httpx.Client(trust_env=False, timeout=60)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=http_client,
    )

    use_parts = args.use_gemini_parts or (args.auto_gemini_parts and "gemini" in (args.model or "").lower())

    messages = _build_messages(args.prompt, use_gemini_parts=use_parts)

    req = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.debug_request:
        # prompt may be shown; key is not
        print("Request:")
        print(json.dumps(req, ensure_ascii=False, indent=2))

    resp = client.chat.completions.create(**req)

    choice0 = resp.choices[0]
    message = choice0.message
    finish_reason = getattr(choice0, "finish_reason", None)

    extracted = _extract_text_from_message(message)

    print("\n=== LLM Test Result ===")
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"use_gemini_parts: {use_parts}")
    print(f"finish_reason: {finish_reason!r}")
    print("content (extracted):")
    print(extracted if extracted is not None else "")

    if args.debug_raw or (not extracted):
        print("\n=== Raw Response Debug ===")
        print(f"resp.model: {getattr(resp, 'model', None)!r}")
        print(f"choices len: {len(resp.choices)}")
        print(f"message.role: {getattr(message, 'role', None)!r}")
        print(f"message.content repr: {repr(getattr(message, 'content', None))}")
        if hasattr(choice0, "__dict__"):
            # Show finish_reason already printed; keep minimal
            pass


if __name__ == "__main__":
    main()
