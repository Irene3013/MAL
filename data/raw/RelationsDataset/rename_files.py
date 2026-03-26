import os
import json

def swap_files(ids, folder, ext):
    for scene_id in ids:
        pos_name = f"pos_{scene_id}.{ext}"
        neg_name = f"neg_{scene_id}.{ext}"
        tmp_name = f"tmp_{scene_id}.{ext}"

        pos_path = os.path.join(folder, pos_name)
        neg_path = os.path.join(folder, neg_name)
        tmp_path = os.path.join(folder, tmp_name)

        # Paso 1: pos -> tmp
        if os.path.exists(pos_path):
            os.rename(pos_path, tmp_path)

        # Paso 2: neg -> pos
        if os.path.exists(neg_path):
            os.rename(neg_path, pos_path)

        # Paso 3: tmp -> neg
        if os.path.exists(tmp_path):
            os.rename(tmp_path, neg_path)


def main():
    version = "v2"
    split ="test"
    # Cargar ids fallidos
    with open("failed_ids.json", "r") as f:
        ids = json.load(f)

    # Cambiar JSONs
    swap_files(ids, folder=f"./{version}/{split}_scenes", ext="json")

    # Cambiar imágenes
    swap_files(ids, folder=f"./{version}/{split}_images", ext="png")

    print("Renombrado completado ✅")


if __name__ == "__main__":
    main()