import argparse
import asyncio
import os
import sys
from pathlib import Path

# 先计算项目根目录
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent

# 关键：确保 config.yaml 相对路径正确
os.chdir(_project_root)

# 再加 sys.path（与 api/app.py 一致做法）
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 关键：如果你环境里有 socks 代理导致 httpx/OpenAI 初始化失败，就先移除代理环境变量
for k in [
    "ALL_PROXY", "all_proxy",
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
    "NO_PROXY", "no_proxy",
]:
    os.environ.pop(k, None)

from pixelle_video.service import PixelleVideoCore  # 放在这里，避免过早触发 config_manager 初始化


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="你好！")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2000)
    args = parser.parse_args()

    core = PixelleVideoCore()
    await core.initialize()

    try:
        # 与 api/routers/llm.py 实际调用方式一致（只传 prompt/temperature/max_tokens）
        response = await core.llm(
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(response)
    finally:
        await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
