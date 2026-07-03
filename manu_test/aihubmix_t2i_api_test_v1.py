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
    # 避免 socks:// 代理 scheme 被 httpx 解析失败（你之前的致命错误）
    for k in [
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
        "NO_PROXY", "no_proxy",
    ]:
        os.environ.pop(k, None)


def try_load_openai_base_from_config() -> tuple[str, str]:
    # 尽可能从 config.yaml 获取；失败就留空让命令行补齐
    try:
        from pixelle_video.config import config_manager
        cfg = config_manager.config

        # 你贴的 config.yaml 里 llm / api_providers 都可能有 base_url
        base_url = ""
        api_key = ""

        # 先取 llm
        base_url = getattr(cfg.llm, "base_url", "") or ""
        api_key = getattr(cfg.llm, "api_key", "") or ""

        return api_key, base_url
    except Exception:
        return "", ""


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
    headers = {"User-Agent": "Pixelle-Video-GLM-Test/2.0"}
    if download_auth and api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with http_client.stream("GET", url, headers=headers, timeout=timeout_s) as r:
        r.raise_for_status()
        out_path_parent = os.path.dirname(out_path)
        if out_path_parent:
            os.makedirs(out_path_parent, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)


async def main():
    disable_proxy_env()

    parser = argparse.ArgumentParser(description="aihubmix/OpenAI-compatible images.generate test for model=glm-image")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base_url, e.g. https://.../v1")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--model", default="glm-image", help="Model name (default: glm-image)")
    parser.add_argument("--prompt", default="A cute orange cat lying on a sunny windowsill, watercolor style")
    parser.add_argument("--size", default="1024x1024", help='OpenAI images size like "1024x1024"')
    parser.add_argument("--n", type=int, default=1, help="Number of images")

    parser.add_argument(
        "--response-format",
        default="url",
        help='OpenAI-style response_format: usually "b64_json" or "url"',
    )

    parser.add_argument("--quality", default="standard", help="Optional quality; empty => not sent")
    parser.add_argument("--save-dir", default="image_glm_test_out_v2", help="Where to save png (empty disables saving)")
    parser.add_argument("--download-auth", action="store_true", help="If downloading url, send Authorization header")
    parser.add_argument("--debug-request", action="store_true", help="Print request (no secrets)")

    args = parser.parse_args()

    cfg_key, cfg_base_url = try_load_openai_base_from_config()

    api_key = args.api_key.strip() or cfg_key
    base_url = args.base_url.strip() or cfg_base_url

    if not base_url:
        raise RuntimeError("Missing base-url. Use --base-url or set it in config.yaml (llm.base_url).")
    if not api_key:
        raise RuntimeError("Missing api-key. Use --api-key or set it in config.yaml (llm.api_key).")

    save_dir = (args.save_dir or "").strip()

    # trust_env=False：避免再次读到系统 socks/代理导致初始化失败
    http_client = httpx.Client(trust_env=False, timeout=120)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    req = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "n": args.n,
        "response_format": args.response_format,
    }
    if (args.quality or "").strip():
        req["quality"] = args.quality.strip()

    if args.debug_request:
        red = dict(req)
        # 不输出 api_key
        print("Request (redacted):")
        print(json.dumps(red, ensure_ascii=False, indent=2))

    resp = client.images.generate(**req)
    data = getattr(resp, "data", None) or []
    if not data:
        raise RuntimeError("images.generate returned empty data")

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

    print("\n=== GLM Image Test Result ===")
    print(f"base_url: {base_url}")
    print(f"model: {args.model}")
    print(f"size: {args.size}, n: {args.n}")
    print(f"response_format: {args.response_format}")
    if save_dir:
        print(f"save_dir: {save_dir}")


if __name__ == "__main__":
    asyncio.run(main())
