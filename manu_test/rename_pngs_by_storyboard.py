import argparse
import difflib
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class StoryRow:
    index: int
    prompt: str
    key: str  # normalized AI Drawing Prompt


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def extract_prompt_from_filename(base: str) -> str:
    # jimeng-YYYY-MM-DD-ID-<prompt...>
    m = re.search(r"(?i)jimeng-\d{4}-\d{2}-\d{2}-\d{1,10}-(.*)$", base)
    if m:
        prompt = m.group(1)
    else:
        m2 = re.search(r"\d{4}-\d{2}-\d{2}-\d{1,10}-(.*)$", base)
        prompt = m2.group(1) if m2 else base

    # remove trailing ellipsis dots like "f....", "room..."
    prompt = re.sub(r"\.{2,}$", "", prompt)
    prompt = re.sub(r"\.{2,}", "", prompt)
    return prompt


def truncated_story_key(story_key: str, n_tokens: int) -> str:
    st_tokens = story_key.split()
    if n_tokens <= 0:
        return ""
    return " ".join(st_tokens[:n_tokens])


def score_match(filename_key: str, story_key: str) -> float:
    """
    Match ONLY with storyboard prefix having the same word-count as filename_key.
    """
    if not filename_key or not story_key:
        return 0.0

    fn_tokens = filename_key.split()
    if not fn_tokens:
        return 0.0
    n = len(fn_tokens)

    story_prefix_key = truncated_story_key(story_key, n)
    if not story_prefix_key:
        return 0.0

    # primary: similarity with same-length truncated storyboard prefix
    s_prefix = similarity_ratio(filename_key, story_prefix_key)

    # secondary: token overlap ratio (position-agnostic)
    fn_set = set(fn_tokens)
    st_set = set(story_prefix_key.split())
    overlap = len(fn_set & st_set) / max(1, len(fn_set))  # 0..1
    s_overlap = overlap * 100.0

    return 0.75 * s_prefix + 0.25 * s_overlap


def parse_storyboard_table(storyboard_path: str) -> List[StoryRow]:
    with open(storyboard_path, "r", encoding="utf-8") as f:
        text = f.read()

    rows: List[StoryRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "|" not in line[1:]:
            continue

        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue

        if not re.fullmatch(r"\d{1,3}", parts[0]):
            continue

        idx = int(parts[0])
        prompt = parts[1]
        key = normalize(prompt)
        rows.append(StoryRow(index=idx, prompt=prompt, key=key))

    seen = set()
    out: List[StoryRow] = []
    for r in rows:
        if r.index in seen:
            continue
        seen.add(r.index)
        out.append(r)

    if not out:
        raise SystemExit(
            "Failed to parse storyboard. Expected markdown table rows like: | 1 | prompt | voice |"
        )
    return out


def pick_unique_indices(
    candidates: List[Tuple[str, str, float]], used: set
) -> List[Tuple[str, str, float]]:
    candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)
    chosen: List[Tuple[str, str, float]] = []
    for img_path, idx_str, score in candidates_sorted:
        idx = int(idx_str)
        if idx in used:
            continue
        used.add(idx)
        chosen.append((img_path, idx_str, score))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rename PNG files into storyboard order by matching filename prompt to storyboard AI Drawing Prompt."
    )
    # 不改你的 default 参数
    ap.add_argument(
        "--input_dir",
        default="/home/manu/tmp/jimeng_dl",
        help="Folder containing unordered PNG images",
    )
    ap.add_argument(
        "--output_dir", default="/home/manu/tmp/jimeng_rn", help="New folder to write renamed PNGs"
    )
    ap.add_argument(
        "--storyboard",
        default="/home/manu/tmp/jimeng-2026-07-08-7767-Minimalist_Money_Rules_storyboard.md",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Match threshold (0-100). Lower means more aggressive matching.",
    )
    ap.add_argument(
        "--ext", default=".png", help="Only rename images with this extension (default: .png)"
    )

    # 新增调试开关（默认不打印）
    ap.add_argument("--debug", default=True, help="Print debug matching information")
    ap.add_argument(
        "--debug_limit", type=int, default=128, help="How many PNG files to debug print"
    )

    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    storyboard_path = os.path.abspath(args.storyboard)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"input_dir not found: {input_dir}")

    # ====== 关键修改：匹配前删除输出文件夹 ======
    if os.path.exists(output_dir):
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        else:
            os.remove(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # ===============================================

    story_rows = parse_storyboard_table(storyboard_path)
    min_idx = min(r.index for r in story_rows)
    max_idx = max(r.index for r in story_rows)

    story_rows_sorted = sorted(story_rows, key=lambda r: r.index)

    png_files = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(args.ext.lower()) and os.path.isfile(os.path.join(input_dir, name))
    ]
    png_files.sort(key=lambda p: os.path.basename(p).lower())
    if not png_files:
        raise SystemExit(f"No PNG files found in {input_dir} with extension {args.ext}")

    candidates: List[Tuple[str, str, float]] = []

    debug_shown = 0
    for img_i, img_path in enumerate(png_files):
        base = os.path.splitext(os.path.basename(img_path))[0]
        filename_prompt_raw = extract_prompt_from_filename(base)
        norm_name = normalize(filename_prompt_raw)

        if not norm_name:
            if args.debug and debug_shown < args.debug_limit:
                print("\n========== DEBUG (empty norm) ==========")
                print("IMG:", base)
                print("raw extracted prompt:", repr(filename_prompt_raw))
                print("normalized:", repr(norm_name))
                debug_shown += 1
            continue

        fn_tokens_list = norm_name.split()
        fn_set = set(fn_tokens_list)
        n_tokens = len(fn_tokens_list)

        # filter + compute best
        best_idx: Optional[int] = None
        best_score = -1.0
        best_story_prompt_original: str = ""
        best_story_key_prefix: str = ""

        # For debug: keep top 5 scores
        top_matches: List[Tuple[int, float, str]] = []  # (idx, score, storyboard_prefix_key_used)

        evaluated = 0
        for row in story_rows_sorted:
            st_set = set(row.key.split())
            if fn_set.isdisjoint(st_set):
                continue

            evaluated += 1
            s = score_match(norm_name, row.key)

            if s > best_score:
                best_score = s
                best_idx = row.index
                best_story_prompt_original = row.prompt
                best_story_key_prefix = truncated_story_key(row.key, n_tokens)

            if args.debug and debug_shown < args.debug_limit:
                prefix_key = truncated_story_key(row.key, n_tokens)
                top_matches.append((row.index, float(s), prefix_key))

        if best_idx is None:
            if args.debug and debug_shown < args.debug_limit:
                print("\n========== DEBUG (no best idx) ==========")
                print("IMG:", base)
                print("raw extracted prompt:", repr(filename_prompt_raw))
                print("normalized filename_key:", repr(norm_name))
                print("n_tokens(filename)=", n_tokens)
                print("evaluated_story_candidates(with token overlap)=", evaluated)
                debug_shown += 1
            continue

        if args.debug and debug_shown < args.debug_limit:
            top_matches_sorted = sorted(top_matches, key=lambda x: x[1], reverse=True)[:5]
            print("\n========== DEBUG ==========")
            print(f"IMG[{img_i}]: {base}")
            print("raw extracted prompt:", repr(filename_prompt_raw))
            print("normalized filename_key:", repr(norm_name))
            print("n_tokens(filename)=", n_tokens)
            print("threshold=", args.threshold)
            print("evaluated_story_candidates(with token overlap)=", evaluated)
            print("BEST MATCH:")
            print("  best_idx=", best_idx, "best_score=", round(best_score, 4))
            print(
                "  storyboard prefix_key (normalized, truncated to n_tokens):",
                repr(best_story_key_prefix),
            )
            print("  storyboard prompt (original):", repr(best_story_prompt_original))
            print("top5 (idx, score, storyboard_prefix_key_used):")
            for idx, sc, prefix_key in top_matches_sorted:
                print("  ", idx, round(sc, 4), "| prefix_key:", prefix_key)
            debug_shown += 1

        if best_score >= args.threshold and min_idx <= best_idx <= max_idx:
            candidates.append((img_path, str(best_idx), float(best_score)))

    used = set()
    chosen = pick_unique_indices(candidates, used)
    chosen_map = {img_path: (idx_str, score) for img_path, idx_str, score in chosen}

    renamed_count = 0
    report = []

    for img_path in png_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if img_path not in chosen_map:
            report.append(
                {
                    "src": img_path,
                    "matched": False,
                    "reason": "no_match_or_below_threshold",
                    "filename_base": base,
                }
            )
            continue

        idx_str, score = chosen_map[img_path]
        dst = os.path.join(output_dir, f"{idx_str}.png")

        if os.path.exists(dst):
            report.append(
                {
                    "src": img_path,
                    "dst": dst,
                    "matched": True,
                    "index": int(idx_str),
                    "score": score,
                    "reason": "dst_exists_skipped",
                    "filename_base": base,
                }
            )
            continue

        shutil.copy2(img_path, dst)
        renamed_count += 1
        report.append(
            {
                "src": img_path,
                "dst": dst,
                "matched": True,
                "index": int(idx_str),
                "score": score,
                "filename_base": base,
            }
        )

    report_path = os.path.join(output_dir, "mapping_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "storyboard": storyboard_path,
                "threshold": args.threshold,
                "story_index_range": [min_idx, max_idx],
                "total_png": len(png_files),
                "renamed": renamed_count,
                "items": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Renamed {renamed_count}/{len(png_files)} images to {output_dir} using threshold={args.threshold}. "
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
