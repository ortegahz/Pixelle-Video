import argparse
import json
import random
import subprocess
from fractions import Fraction


def ffprobe_json(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def parse_fps(fr: str) -> float:
    if fr is None:
        return 30.0
    try:
        return float(Fraction(fr))
    except Exception:
        return float(fr)


def get_video_meta(path: str):
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    vstream = None
    for s in streams:
        if s.get("codec_type") == "video":
            vstream = s
            break
    if not vstream:
        raise RuntimeError("No video stream found.")

    width = int(vstream["width"])
    height = int(vstream["height"])

    r_frame_rate = vstream.get("r_frame_rate")
    avg_frame_rate = vstream.get("avg_frame_rate")
    fps = parse_fps(avg_frame_rate or r_frame_rate or "30/1")

    fmt = data.get("format", {})
    duration = float(fmt["duration"]) if fmt and fmt.get("duration") else None
    if duration is None and vstream.get("duration"):
        duration = float(vstream["duration"])
    if duration is None and vstream.get("duration"):
        duration = float(vstream["duration"])
    if duration is None:
        raise RuntimeError("Cannot determine duration.")

    # d=1 => output frames should be ~ input frames, close enough for slow movement
    total_frames = max(2, int(round(duration * fps)))
    return width, height, fps, duration, total_frames


def build_zoompan_filter(w, h, fps, total_frames, zoom_max, pan_mode, seed=None,
                         pan_amp_range=(0.02, 0.07), zoom_amp_range=(0.01, 0.04)):
    random.seed(seed)

    denom = max(1, total_frames - 1)
    y_center = f"({h}-oh)/2"

    # ----- random slow zoom mode -----
    # zoom_base = 1.0, zoom moves very slowly within [1.0, zoom_max]
    zoom_ceiling = max(1.0, float(zoom_max))
    zoom_amp_cap = max(0.0, zoom_ceiling - 1.0)
    if zoom_amp_cap <= 0:
        zoom_amp = 0.0
    else:
        zoom_amp = random.uniform(zoom_amp_range[0], zoom_amp_range[1])
        zoom_amp = min(zoom_amp, zoom_amp_cap)

    zoom_mode = random.choice(["in", "out", "inout"])  # slow zoom in / out / in-out

    if zoom_amp == 0:
        z_expr = "1.0"
    else:
        # keep within bounds explicitly
        if zoom_mode == "in":
            # 1.0 -> 1.0+zoom_amp
            z_expr = f"min(1+{zoom_amp:.6f}*on/{denom}, {zoom_ceiling:.6f})"
        elif zoom_mode == "out":
            # 1.0+zoom_amp -> 1.0
            z_expr = f"min(1+{zoom_amp:.6f}*(1-on/{denom}), {zoom_ceiling:.6f})"
        else:
            # in-out: starts ~1.0, peaks mid, returns ~1.0
            # use (1-cos(2πt))/2 which starts at 0 and ends at 0
            z_expr = (
                f"min(1+{zoom_amp:.6f}*(1-cos(2*PI*on/{denom}))/2, {zoom_ceiling:.6f})"
            )

    # ----- random slow pan (slight up/down/left/right) -----
    # We'll apply small sub-pixel motion in the available crop window.
    pan_x_amp = 0.0
    pan_y_amp = 0.0

    def rand_amp():
        return random.uniform(pan_amp_range[0], pan_amp_range[1])

    if pan_mode == "none":
        x_expr = f"(iw-ow)/2"
        y_expr = y_center
    elif pan_mode == "lr":
        pan_x_amp = rand_amp() * random.choice([-1.0, 1.0])
        x_expr = f"(iw-ow)/2 + (iw-ow)*{pan_x_amp:.6f}*(2*on/{denom}-1)"
        y_expr = y_center
    elif pan_mode == "rl":
        pan_x_amp = rand_amp() * random.choice([-1.0, 1.0])
        # reverse bias (small)
        x_expr = f"(iw-ow)/2 + (iw-ow)*{pan_x_amp:.6f}*(1-2*on/{denom})"
        y_expr = y_center
    elif pan_mode == "ud":
        pan_y_amp = rand_amp() * random.choice([-1.0, 1.0])
        x_expr = f"(iw-ow)/2"
        y_expr = f"({h}-oh)/2 + (ih-oh)*{pan_y_amp:.6f}*(2*on/{denom}-1)"
    elif pan_mode == "du":
        pan_y_amp = rand_amp() * random.choice([-1.0, 1.0])
        x_expr = f"(iw-ow)/2"
        y_expr = f"({h}-oh)/2 + (ih-oh)*{pan_y_amp:.6f}*(1-2*on/{denom})"
    else:
        # random: allow both x and y slight movement
        pan_x_amp = rand_amp() * random.choice([-1.0, 1.0])
        pan_y_amp = rand_amp() * random.choice([-1.0, 1.0])

        x_expr = f"(iw-ow)/2 + (iw-ow)*{pan_x_amp:.6f}*(2*on/{denom}-1)"
        y_expr = f"({h}-oh)/2 + (ih-oh)*{pan_y_amp:.6f}*(2*on/{denom}-1)"

    # d=1 => output frames ~ input frames (avoid frame explosion / ultra-slow issue)
    s_expr = f"{w}x{h}"
    vf = (
        "zoompan="
        f"z='{z_expr}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d=1:"
        f"s={s_expr}:"
        f"fps={fps}"
    )
    return vf, zoom_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="/run/media/manu/windows/workspace/Pixelle-Video/output/20260703_153409_8942/frames/01_segment.mp4")
    ap.add_argument("--output", default="/home/manu/tmp/move.mp4")

    ap.add_argument("--pan", default="random",
                    choices=["random", "lr", "rl", "ud", "du", "none"],
                    help="Random pan or fixed direction (slight/slow motion).")

    ap.add_argument("--zoom-max", type=float, default=1.12,
                    help="Max zoom multiplier ceiling (e.g. 1.08~1.18).")

    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", type=str, default="veryfast",
                    help="x264 preset: ultrafast/veryfast/fast/medium/...")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional random seed. If not set, motion varies each run.")
    args = ap.parse_args()

    w, h, fps, duration, total_frames = get_video_meta(args.input)

    vf, zoom_mode = build_zoompan_filter(
        w=w, h=h, fps=fps, total_frames=total_frames,
        zoom_max=args.zoom_max,
        pan_mode=args.pan,
        seed=args.seed
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", args.input,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        args.output
    ]

    print("Video meta:",
          f"{w}x{h}, fps={fps:.3f}, duration={duration:.3f}s, total_frames~{total_frames}")
    print("zoom_mode:", zoom_mode, "pan:", args.pan, "zoom_max:", args.zoom_max)
    print("Running ffmpeg with filter:\n", vf)

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
