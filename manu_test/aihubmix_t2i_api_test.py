import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI


def _extract_img_fields(img_obj: Any) -> tuple[Optional[str], Optional[str]]:
    """Returns (b64_json, url). Either may be None depending on provider."""
    b64 = getattr(img_obj, "b64_json", None)
    url = getattr(img_obj, "url", None)
    if isinstance(b64, str) and not b64.strip():
        b64 = None
    if isinstance(url, str) and not url.strip():
        url = None
    return b64, url


def _save_b64_to_file(b64: str, file_path: str) -> None:
    raw = base64.b64decode(b64)
    with open(file_path, "wb") as f:
        f.write(raw)


def _download_to_file(
        http_client: httpx.Client,
        url: str,
        save_path: str,
        *,
        api_key: str,
        download_auth: bool,
        timeout_s: float = 120,
) -> bool:
    headers = {
        "User-Agent": "Pixelle-Video-T2I-Tester/1.0",
    }
    if download_auth:
        # 解决“url 需要鉴权才能访问”导致 404/403 的常见情况
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with http_client.stream("GET", url, headers=headers, timeout=timeout_s) as r:
            if r.status_code != 200:
                # 不中断，直接返回失败，由上层决定 fallback
                print(f"[download failed] status={r.status_code} url={url}")
                return False
            with open(save_path, "wb") as f:
                for chunk in r.iter_bytes():
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"[download failed] url={url} error={e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test aihubmix/OpenAI-compatible T2I (images.generate)")
    parser.add_argument("--base-url", default="https://api.inferera.com/v1",
                        help="OpenAI-compatible base_url, e.g. https://.../v1")
    parser.add_argument("--api-key", default=os.getenv("AIHUBMIX_API_KEY", ""),
                        help="API key (avoid hardcoding secrets)")
    parser.add_argument("--model", default=os.getenv("AIHUBMIX_IMAGE_MODEL", "glm-image"), help="Image model name")
    parser.add_argument("--prompt", default=os.getenv("AIHUBMIX_PROMPT",
                                                      "A cute orange cat lying on a sunny windowsill, watercolor style"),
                        help="Prompt text")

    parser.add_argument("--size", default=os.getenv("AIHUBMIX_IMAGE_SIZE", "1024x1024"),
                        help='Image size like "1024x1024"')
    parser.add_argument("--quality", default=os.getenv("AIHUBMIX_IMAGE_QUALITY", ""),
                        help='Quality value; if empty it is NOT sent (avoid 400)')
    parser.add_argument("--n", type=int, default=int(os.getenv("AIHUBMIX_IMAGE_N", "1")), help="Number of images")

    # 关键：优先 b64，避免下载 url
    parser.add_argument(
        "--response-format",
        default=os.getenv("AIHUBMIX_IMAGE_RESPONSE_FORMAT", "b64_json"),
        help='OpenAI-style response_format: usually "b64_json" or "url". Empty = do not send.',
    )

    parser.add_argument("--save-dir", default=os.getenv("AIHUBMIX_IMAGE_SAVE_DIR", "image_test_out"),
                        help="Where to save images; empty disables saving")
    parser.add_argument("--download-auth", action="store_true",
                        help="When saving from url, send Authorization header to avoid 404/403")
    parser.add_argument("--debug-request", action="store_true", help="Print request (no secrets)")
    parser.add_argument("--debug-raw", action="store_true", help="Print extra debug")

    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing api-key. Set --api-key or env AIHUBMIX_API_KEY")
    if not args.base_url:
        raise RuntimeError("Missing base-url.")

    http_client = httpx.Client(trust_env=False, timeout=60)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, http_client=http_client)

    save_dir = (args.save_dir or "").strip()

    req: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "n": args.n,
    }
    if (args.quality or "").strip():
        req["quality"] = args.quality.strip()

    # 可选 response_format；空表示不传
    if (args.response_format or "").strip():
        req["response_format"] = args.response_format.strip()

    if args.debug_request:
        print("Request:")
        print(json.dumps(req, ensure_ascii=False, indent=2))

    resp = client.images.generate(**req)

    data = getattr(resp, "data", None) or []
    if not data:
        raise RuntimeError("images.generate returned empty data")

    out_paths: list[str] = []

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(data):
        b64, url = _extract_img_fields(img)

        if args.debug_raw:
            print(f"[img {i}] has_b64={bool(b64)}, has_url={bool(url)}")

        if not save_dir:
            if b64:
                print(f"[img {i}] base64_len={len(b64)}")
            elif url:
                print(f"[img {i}] url={url}")
            else:
                print(f"[img {i}] no b64_json and no url")
            continue

        file_path = os.path.join(save_dir, f"img_{i}.png")

        if b64:
            _save_b64_to_file(b64, file_path)
            out_paths.append(file_path)
        elif url:
            ok = _download_to_file(
                http_client,
                url,
                file_path,
                api_key=args.api_key,
                download_auth=args.download_auth,
            )
            if ok:
                out_paths.append(file_path)
        else:
            print(f"[img {i}] neither b64_json nor url returned; cannot save")

    print("\n=== T2I Test Result ===")
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"size: {args.size}, quality: {(args.quality or '').strip() or '(not sent)'}, n: {args.n}")
    print(f"response_format: {(args.response_format or '').strip() or '(not sent)'}")

    if save_dir:
        print("saved files:")
        for p in out_paths:
            print(p)
        if not out_paths:
            raise RuntimeError("Request succeeded but saving failed (no b64, and url download failed).")
    else:
        print("save-dir is empty; not saving files.")


if __name__ == "__main__":
    main()
