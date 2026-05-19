#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

RELATIONS = [
    "left_of",
    "right_of",
    "in-front_of",
    "behind",
]

OPPOSITES = [
    {"left_of", "right_of"},
    {"in-front_of", "behind"},
]

EXPECTED_RELATIONS = set(RELATIONS)

# ============================================================
# PARSING
# ============================================================

def parse_image_name(image_path):
    """
    Extrae:
        obj1, relation, obj2

    Ejemplo:
        mug_right_of_knife.png
        -> ("mug", "right_of", "knife")
    """

    name = Path(image_path).stem

    for rel in RELATIONS:

        pattern = f"_{rel}_"

        if pattern in name:

            parts = name.split(pattern)

            if len(parts) != 2:
                return None

            obj1 = parts[0]
            obj2 = parts[1]

            return obj1, rel, obj2

    return None


def get_object_key(obj1, obj2):
    """
    Clave canónica para agrupar pares de objetos.

    mug + knife
    knife + mug

    -> mismo grupo
    """

    return tuple(sorted([obj1, obj2]))


# ============================================================
# GROUPING
# ============================================================

def group_dataset(dataset):

    sets = defaultdict(list)
    pairs = defaultdict(list)

    for item in dataset:

        parsed = parse_image_name(item["image_path"])

        if parsed is None:
            continue

        obj1, rel, obj2 = parsed

        obj_key = get_object_key(obj1, obj2)

        # ---------------------------
        # SETS
        # ---------------------------

        sets[obj_key].append(item)

        # ---------------------------
        # PAIRS
        # ---------------------------

        for opposite_pair in OPPOSITES:

            if rel in opposite_pair:

                pair_key = (
                    obj_key,
                    frozenset(opposite_pair)
                )

                pairs[pair_key].append(item)

                break

    return sets, pairs


# ============================================================
# VALIDATION
# ============================================================

def validate_sets(sets):

    print("\n==============================")
    print("VALIDATING SETS")
    print("==============================")

    valid_sets = 0
    invalid_sets = 0

    for obj_key, items in sets.items():

        relations = []

        for item in items:

            parsed = parse_image_name(item["image_path"])

            if parsed is None:
                continue

            _, rel, _ = parsed

            relations.append(rel)

        rel_set = set(relations)

        missing = EXPECTED_RELATIONS - rel_set

        duplicates = len(relations) != len(rel_set)

        is_valid = (
            len(items) == 4
            and not missing
            and not duplicates
        )

        if is_valid:

            valid_sets += 1

        else:

            invalid_sets += 1

            print(f"\n[INVALID SET] {obj_key}")
            print(f" relations: {relations}")

            if missing:
                print(f" missing relations: {missing}")

            if duplicates:
                print(" duplicated relations detected")

            if len(items) != 4:
                print(f" expected 4 items, got {len(items)}")

    print("\n------------------------------")
    print(f"Valid sets:   {valid_sets}")
    print(f"Invalid sets: {invalid_sets}")


def validate_pairs(pairs):

    print("\n==============================")
    print("VALIDATING PAIRS")
    print("==============================")

    valid_pairs = 0
    invalid_pairs = 0

    for pair_key, items in pairs.items():

        obj_key, expected_pair = pair_key

        relations = []

        for item in items:

            parsed = parse_image_name(item["image_path"])

            if parsed is None:
                continue

            _, rel, _ = parsed

            relations.append(rel)

        rel_set = set(relations)

        is_valid = (
            len(items) == 2
            and rel_set == expected_pair
        )

        if is_valid:

            valid_pairs += 1

        else:

            invalid_pairs += 1

            print(f"\n[INVALID PAIR] {obj_key}")
            print(f" relations: {relations}")
            print(f" expected: {set(expected_pair)}")

            if len(items) != 2:
                print(f" expected 2 items, got {len(items)}")

    print("\n------------------------------")
    print(f"Valid pairs:   {valid_pairs}")
    print(f"Invalid pairs: {invalid_pairs}")


# ============================================================
# EXPORT
# ============================================================

def export_grouped_json(sets, pairs, output_path):

    export_data = {
        "sets": {},
        "pairs": {}
    }

    # ========================================================
    # EXPORT SETS
    # ========================================================

    for obj_key, items in sets.items():

        key = f"{obj_key[0]}__{obj_key[1]}"

        export_data["sets"][key] = []

        for item in items:

            parsed = parse_image_name(item["image_path"])

            if parsed is None:
                continue

            _, rel, _ = parsed

            export_data["sets"][key].append({
                "image_path": item["image_path"],
                "relation": rel,
                "caption_options": item.get("caption_options", []),
                "correct_option": (
                    item["correct_option"]
                    if "correct_option" in item
                    else (
                        item["caption_options"][0]
                        if "caption_options" in item
                        and len(item["caption_options"]) > 0
                        else None
                    )
                )
            })

        # ordenar relaciones
        export_data["sets"][key] = sorted(
            export_data["sets"][key],
            key=lambda x: x["relation"]
        )

    # ========================================================
    # EXPORT PAIRS
    # ========================================================

    for pair_key, items in pairs.items():

        obj_key, rel_pair = pair_key

        rel_name = "__".join(sorted(rel_pair))

        key = f"{obj_key[0]}__{obj_key[1]}__{rel_name}"

        export_data["pairs"][key] = []

        for item in items:

            parsed = parse_image_name(item["image_path"])

            if parsed is None:
                continue

            _, rel, _ = parsed

            export_data["pairs"][key].append({
                "image_path": item["image_path"],
                "relation": rel
            })

    # ========================================================
    # SAVE
    # ========================================================

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nGrouped dataset exported to:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to original dataset json"
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="grouped_dataset.json",
        help="Output grouped json"
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nLoaded dataset with {len(dataset)} samples")

    # ========================================================
    # GROUP
    # ========================================================

    sets, pairs = group_dataset(dataset)

    print(f"\nTotal set groups:  {len(sets)}")
    print(f"Total pair groups: {len(pairs)}")

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_sets(sets)
    validate_pairs(pairs)

    # ========================================================
    # EXPORT
    # ========================================================

    export_grouped_json(
        sets,
        pairs,
        args.output_json
    )


if __name__ == "__main__":
    main()