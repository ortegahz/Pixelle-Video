import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI

# --------- 与 api/app.py 一致：修正 CWD + sys.path ----------
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
os.chdir(_project_root)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def disable_proxy_env():
    for k in [
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
        "NO_PROXY", "no_proxy",
    ]:
        os.environ.pop(k, None)


def guess_mime_from_ext(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext in ("png",):
        return "png"
    if ext in ("webp",):
        return "webp"
    return "png"


def image_file_to_data_url(path: str) -> str:
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = guess_mime_from_ext(str(p))
    return f"data:image/{mime};base64,{b64}"


def extract_b64_or_url(img_obj) -> tuple[Optional[str], Optional[str]]:
    b64 = getattr(img_obj, "b64_json", None)
    url = getattr(img_obj, "url", None)
    if isinstance(b64, str) and not b64.strip():
        b64 = None
    if isinstance(url, str) and not url.strip():
        url = None
    return b64, url


def save_b64_png(b64: str, out_path: str) -> None:
    raw = base64.b64decode(b64)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw)


def download_to_file(
        http_client: httpx.Client,
        url: str,
        out_path: str,
        *,
        api_key: str,
        timeout_s: float = 120,
        download_auth: bool = False,
) -> None:
    headers = {"User-Agent": "Pixelle-Video-OpenAI-RefEdit/1.0"}
    if download_auth and api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with http_client.stream("GET", url, headers=headers, timeout=timeout_s) as r:
        r.raise_for_status()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)


async def main():
    disable_proxy_env()

    parser = argparse.ArgumentParser(
        description="OpenAI-compatible reference image + prompt (wan2.7-image) without dashscope"
    )
    parser.add_argument("--base-url", default="https://api.inferera.com/v1", help="OpenAI-compatible base_url, e.g. https://.../v1")
    parser.add_argument("--api-key", default="sk-jskrQVNcLpB738UQEa0649153f9e49D4BcB878D203A82784", help="API key")

    parser.add_argument("--model", default="glm-image", help="Model name (default: wan2.7-image)")
    parser.add_argument("--reference-image", default="/home/manu/tmp/img_1.png", help="Local actor/reference photo path")
    parser.add_argument("--prompt", default="参考图的人物在公园散步。保持卡通风格", help="Text prompt for generation/edit")

    parser.add_argument("--size", default="1920x1088", help='Like "1024x1024" or "1920x1088"')
    parser.add_argument("--n", type=int, default=1, help="Number of images")
    parser.add_argument("--quality", default="", help="Optional quality; empty => not sent")
    parser.add_argument("--response-format", default="url", help='Usually "b64_json" or "url"')

    parser.add_argument(
        "--identity-keep",
        action="store_true",
        help="Add strong constraint: keep same person identity as reference image.",
    )

    parser.add_argument("--save-dir", default="image_wan_openai_ref_out",
                        help="Where to save png (empty disables saving)")
    parser.add_argument("--download-auth", action="store_true", help="If downloading url, send Authorization header")
    parser.add_argument("--debug-request", action="store_true", help="Print request (no secrets)")

    args = parser.parse_args()

    base_url = args.base_url.strip()
    api_key = args.api_key.strip()
    if not base_url:
        raise RuntimeError("Missing base-url. Use --base-url.")
    if not api_key:
        raise RuntimeError("Missing api-key. Use --api-key.")

    # reference -> data URL
    ref_data_url = image_file_to_data_url(args.reference_image)

    http_client = httpx.Client(trust_env=False, timeout=120)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    prompt = args.prompt.strip()
    if args.identity_keep:
        prompt = (
                "Use the reference image as the exact actor identity. "
                "Keep the same face, identity, and key appearance features as the reference photo. "
                "Do not change the person to another actor. "
                "Only apply the requested changes from the text prompt.\n"
                + prompt
        )

    req = {
        "model": args.model,
        "prompt": prompt,
        "size": args.size,
        "n": args.n,
        "response_format": args.response_format,
    }
    if (args.quality or "").strip():
        req["quality"] = args.quality.strip()

    # 关键：通过 extra_body 传参考图
    extra_body = {"image_url": [ref_data_url]}

    if args.debug_request:
        red = dict(req)
        print("Request (redacted):")
        print(json.dumps(red, ensure_ascii=False, indent=2))
        print("extra_body keys:", list(extra_body.keys()))

    # 调 OpenAI-compatible
    resp = client.images.generate(**req, extra_body=extra_body)
    data = getattr(resp, "data", None) or []
    if not data:
        raise RuntimeError("images.generate returned empty data")

    save_dir = (args.save_dir or "").strip()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(data):
        b64, url = extract_b64_or_url(img)

        if not save_dir:
            print(f"[img {i}] has_b64={bool(b64)} has_url={bool(url)}")
            continue

        out_path = os.path.join(save_dir, f"img_{i}.png")
        if b64:
            save_b64_png(b64, out_path)
            print(f"[img {i}] saved(b64): {out_path}")
        elif url:
            download_to_file(
                http_client,
                url,
                out_path,
                api_key=api_key,
                download_auth=args.download_auth,
            )
            print(f"[img {i}] saved(url): {out_path}")
        else:
            raise RuntimeError(f"[img {i}] neither b64_json nor url returned; response_format={args.response_format}")

    print("\n=== OpenAI Ref Result ===")
    print(f"model: {args.model}")
    print(f"size: {args.size}, n: {args.n}")
    print(f"response_format: {args.response_format}")
    print(f"save_dir: {save_dir if save_dir else '(disabled)'}")


if __name__ == "__main__":
    asyncio.run(main())
