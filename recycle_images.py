#!/usr/bin/env python3
"""
Recycles images from dataset v3 to v9 by matching spatial relation examples.
Matches on: relation, shape1, color1, shape2, color2

Also handles swapped-object matches: if v9 has (relation, A, colA, B, colB)
and v3 has (relation, B, colB, A, colA), the images are the same but pos↔neg
are swapped.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

MATCH_COLS = ["relation", "shape1", "color1", "shape2", "color2"]


def load_v3(v3_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate v3 train+val and test CSVs."""
    train = pd.concat(
        [
            pd.read_csv(v3_dir / "v3_train.csv"),
            pd.read_csv(v3_dir / "v3_val.csv"),
        ],
        ignore_index=True,
    )
    test = pd.read_csv(v3_dir / "v3_test.csv")

    # Add a 'used' flag to avoid reusing the same row twice
    train["_used"] = False
    test["_used"] = False

    return train, test


def find_match(v3_df: pd.DataFrame, row: pd.Series) -> tuple[int | None, bool]:
    """
    Return (index, swapped) of the first unused matching row in v3_df.
    swapped=False → direct match (same object order)
    swapped=True  → objects are interchanged (shape1↔shape2, color1↔color2)
    Returns (None, False) if no match found.
    """
    available = ~v3_df["_used"]

    # 1. Try direct match
    mask = available.copy()
    for col in MATCH_COLS:
        mask &= v3_df[col] == row[col]
    candidates = v3_df.index[mask]
    if len(candidates) > 0:
        return candidates[0], False

    # 2. Try swapped match (same relation, but shape1↔shape2 and color1↔color2)
    mask = available.copy()
    mask &= v3_df["relation"] == row["relation"]
    mask &= v3_df["shape1"]   == row["shape2"]
    mask &= v3_df["color1"]   == row["color2"]
    mask &= v3_df["shape2"]   == row["shape1"]
    mask &= v3_df["color2"]   == row["color1"]
    candidates = v3_df.index[mask]
    if len(candidates) > 0:
        return candidates[0], True

    return None, False


def copy_image_pair(
    src_dir: Path,
    dst_dir: Path,
    src_id: str,
    dst_id: str,
    swap: bool = False,
) -> None:
    """
    Copy pos/neg image pair from src to dst.
    If swap=True, pos_src → neg_dst and neg_src → pos_dst.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    mapping = [("pos", "neg"), ("neg", "pos")] if swap else [("pos", "pos"), ("neg", "neg")]
    for src_prefix, dst_prefix in mapping:
        src_file = src_dir / f"{src_prefix}_{src_id}"
        dst_file = dst_dir / f"{dst_prefix}_{dst_id}"
        if not src_file.exists():
            raise FileNotFoundError(f"Source image not found: {src_file}")
        shutil.copy2(src_file, dst_file)


def process(
    v9_dir: Path,
    v3_dir: Path,
    dry_run: bool = False,
) -> None:
    # ── Load v3 data ──────────────────────────────────────────────────────────
    v3_train, v3_test = load_v3(v3_dir)

    v3_train_img_dir = v3_dir / "train_images"
    v3_test_img_dir  = v3_dir / "test_images"

    # ── Load v9 data ──────────────────────────────────────────────────────────
    v9_train = pd.read_csv(v9_dir / "train.csv")
    v9_test  = pd.read_csv(v9_dir / "v9_test.csv")

    v9_train_img_dir = v9_dir / "train_images"
    v9_test_img_dir  = v9_dir / "test_images"

    stats = {"found": 0, "missing": 0, "cross": 0, "swapped": 0}

    # Search order for each v9 split: primary source first, fallback second
    search_plan = {
        "train": [
            (v3_train, v3_train_img_dir, "v3 train+val"),
            (v3_test,  v3_test_img_dir,  "v3 test"),
        ],
        "test": [
            (v3_test,  v3_test_img_dir,  "v3 test"),
            (v3_train, v3_train_img_dir, "v3 train+val"),
        ],
    }

    def process_split(
        v9_df: pd.DataFrame,
        v9_img_dir: Path,
        split_name: str,
    ) -> None:
        missing_rows = []
        for _, v9_row in v9_df.iterrows():
            dst_id = v9_row["image"]
            matched = False

            for v3_df, v3_img_dir, source_label in search_plan[split_name]:
                idx, swapped = find_match(v3_df, v9_row)
                if idx is None:
                    continue

                v3_row = v3_df.loc[idx]
                src_id = v3_row["image"]
                is_cross = source_label != search_plan[split_name][0][2]

                notes = []
                if is_cross:
                    notes.append(f"CROSS from {source_label}")
                if swapped:
                    notes.append("pos↔neg swapped")
                note_str = f" [{', '.join(notes)}]" if notes else ""

                if dry_run:
                    print(f"[DRY-RUN] {split_name}: would copy {src_id} → {dst_id}{note_str}")
                else:
                    copy_image_pair(v3_img_dir, v9_img_dir, src_id, dst_id, swap=swapped)
                    if notes:
                        print(f"  ✎ {split_name}: {src_id} → {dst_id}{note_str}")

                v3_df.at[idx, "_used"] = True
                stats["found"] += 1
                if is_cross:
                    stats["cross"] += 1
                if swapped:
                    stats["swapped"] += 1
                matched = True
                break

            if not matched:
                missing_rows.append(v9_row[MATCH_COLS].to_dict() | {"image": dst_id})
                stats["missing"] += 1

        if missing_rows:
            print(f"\n⚠️  [{split_name}] {len(missing_rows)} examples had NO match in v3:")
            for r in missing_rows:
                print(f"   {r}")

    # ── Process splits ────────────────────────────────────────────────────────
    print(f"Processing v9 train  ({len(v9_train)} rows) against v3 train+val → fallback test…")
    process_split(v9_train, v9_train_img_dir, "train")

    print(f"Processing v9 test   ({len(v9_test)} rows) against v3 test → fallback train+val…")
    process_split(v9_test, v9_test_img_dir, "test")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = stats["found"] + stats["missing"]
    print(f"\n{'='*50}")
    print(f"{'DRY-RUN ' if dry_run else ''}Results: {stats['found']}/{total} examples recycled successfully.")
    if stats["cross"]:
        print(f"  ↔  {stats['cross']} matched via cross-split fallback.")
    if stats["swapped"]:
        print(f"  🔄  {stats['swapped']} matched with pos↔neg swap.")
    if stats["missing"]:
        print(f"  ⚠️  {stats['missing']} examples could not be matched — images must be generated.")
    else:
        print("  ✅ All examples matched — no new generation needed!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recycle v3 dataset images into v9 by matching on spatial relation columns."
    )
    parser.add_argument(
        "--v9-dir",
        required=True,
        type=Path,
        help="Root directory of dataset v9 (contains train.csv, v9_test.csv, train_images/, test_images/).",
    )
    parser.add_argument(
        "--v3-dir",
        required=True,
        type=Path,
        help="Root directory of dataset v3 (contains v3_train.csv, v3_val.csv, v3_test.csv, train_images/, test_images/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying any files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for label, path in [("v9-dir", args.v9_dir), ("v3-dir", args.v3_dir)]:
        if not path.is_dir():
            print(f"ERROR: --{label} does not exist or is not a directory: {path}", file=sys.stderr)
            sys.exit(1)

    process(args.v9_dir, args.v3_dir, dry_run=args.dry_run)