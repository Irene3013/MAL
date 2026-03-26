import json
import os
import numpy as np

# -------------------------
# Utils
# -------------------------

def get_axes(directions):
    right = np.array(directions["right"])
    forward = np.array(directions["front"])
    up = np.array(directions["above"])
    return right, up, forward


def project(vec, axis):
    return np.dot(vec, axis)


def check_relation(rel, obj1_pos, obj2_pos, axes):
    right, up, forward = axes

    v1 = np.array(obj1_pos)
    v2 = np.array(obj2_pos)

    if rel == "left":
        return project(v1, right) < project(v2, right)

    elif rel == "right":
        return project(v1, right) > project(v2, right)

    elif rel == "front":
        return project(v1, forward) > project(v2, forward)

    elif rel == "behind":
        return project(v1, forward) < project(v2, forward)

    return False


# def find_object(scene_objects, target):
#     for obj in scene_objects:
#         if (
#             obj["color"] == target["color"] and
#             obj["size"] == target["size"] and
#             obj["shape"] == target["shape"]
#         ):
#             return obj
#     return None


# -------------------------
# MAIN
# -------------------------

def main(relations_path, scenes_folder):
    with open(relations_path, "r") as f:
        data = json.load(f)

    failed_ids = []

    for relation, entries in data.items():

        print(f"Checking relation: {relation}")

        for entry in entries:
            scene_id = entry["imageID"]

            scene_path = os.path.join(
                scenes_folder,
                f"pos_{scene_id}.json"
            )

            if not os.path.exists(scene_path):
                print(f"Missing file: {scene_path}")
                failed_ids.append(scene_id)
                continue

            with open(scene_path, "r") as f:
                scene = json.load(f)

            obj1 = scene["obj1"]
            obj2 = scene["obj2"]

            # if obj1 is None or obj2 is None:
            #     print(f"Objects not found in scene {scene_id}")
            #     failed_ids.append(scene_id)
            #     continue

            axes = get_axes(entry["directions"])

            ok = check_relation(
                relation,
                obj1["3d_coords"],
                obj2["3d_coords"],
                axes
            )

            if not ok:
                failed_ids.append(scene_id)

    # Guardar resultados
    with open("failed_ids.json", "w") as f:
        json.dump(failed_ids, f, indent=2)

    print(f"\nTotal fallidos: {len(failed_ids)}")


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    version = "v2"
    split ="test"
    main(
        relations_path=f"./{version}/{version}_{split}.json",
        scenes_folder=f"./{version}/{split}_scenes"
    )