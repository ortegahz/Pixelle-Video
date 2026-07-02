import argparse
import json
import os
from typing import Any, Optional

import httpx
from openai import OpenAI


def _extract_text_from_message(message: Any) -> Optional[str]:
    """
    Try multiple shapes because some OpenAI-compatible gateways return different fields.
    """
    if message is None:
        return None

    content = getattr(message, "content", None)

    # 1) Standard: message.content is a string
    if isinstance(content, str):
        return content

    # 2) Some gateways: message.content is a list of parts
    if isinstance(content, list):
        parts_text = []
        for p in content:
            if not isinstance(p, dict):
                continue
            # common keys
            if "text" in p and isinstance(p["text"], str):
                parts_text.append(p["text"])
            elif "content" in p and isinstance(p["content"], str):
                parts_text.append(p["content"])
        if parts_text:
            return "".join(parts_text)

    # 3) Other potential fields (best-effort)
    for k in ("output_text", "text", "response"):
        v = getattr(message, k, None)
        if isinstance(v, str) and v.strip():
            return v

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test aihubmix OpenAI-compatible LLM via chat.completions (robust output extraction)"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.inferera.com/v1",
        help="OpenAI-compatible base_url (should be API root, e.g. https://.../v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AIHUBMIX_API_KEY", ""),
        help="API key (or set env AIHUBMIX_API_KEY). Avoid hardcoding secrets.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AIHUBMIX_MODEL", "gpt-5.4-nano"),
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("AIHUBMIX_PROMPT", "你好"),
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
        "--use-parts",
        action="store_true",
        help="Send Gemini-style parts: messages[].content = [{type:'text', text:...}]",
    )
    parser.add_argument(
        "--debug-request",
        action="store_true",
        help="Print request body (no secrets)",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print raw response summary",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing api-key. Set --api-key or env AIHUBMIX_API_KEY")

    # IMPORTANT: avoid socks proxy parsing issues (previously you hit socks:// scheme error)
    http_client = httpx.Client(trust_env=False, timeout=60)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=http_client,
    )

    if args.use_parts:
        # Some gateways require Gemini-style parts for certain models
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        }]
    else:
        messages = [{"role": "user", "content": args.prompt}]

    req = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.debug_request:
        safe_req = dict(req)
        if isinstance(safe_req.get("messages"), list):
            # prompt itself is not a secret; keep it
            pass
        print("Request:")
        print(json.dumps(safe_req, ensure_ascii=False, indent=2))

    resp = client.chat.completions.create(**req)

    choice0 = resp.choices[0]
    message = choice0.message
    finish_reason = getattr(choice0, "finish_reason", None)

    extracted = _extract_text_from_message(message)

    print("\n=== LLM Test Result ===")
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"finish_reason: {finish_reason!r}")
    print("content (extracted):")
    print(extracted if extracted is not None else "")

    # If extracted is empty, print more detail (without dumping huge objects)
    if args.debug_raw or (not extracted):
        print("\n=== Raw Response Debug ===")
        print(f"resp.model: {getattr(resp, 'model', None)!r}")
        print(f"choices len: {len(resp.choices) if getattr(resp, 'choices', None) is not None else None}")
        print(f"message.type: {type(message)}")
        # show message.content directly (may be '', list, etc.)
        print(f"message.content repr: {repr(getattr(message, 'content', None))}")
        print(f"message.role: {getattr(message, 'role', None)!r}")
        print(f"message keys: {getattr(message, '__dict__', message)}")


if __name__ == "__main__":
    main()
