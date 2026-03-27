import json
import math
import random
import argparse
import pandas as pd  # Importante para los counts
from pathlib import Path
from decimal import Decimal

# Constants
ROOT = Path("/content/drive/MyDrive/EHU/KISA/TFM/RelationsDataset")
FOLDER_DIR = ROOT / "data"
SCENES_DIR = ROOT / "original_clevr"
RELATIONS = ["left", "right", "front", "behind"]

def center_distance_xy(obj1, obj2):
    """
    Distancia euclídea entre los centros en el plano XY.
    """
    x1, y1, _ = obj1["3d_coords"]
    x2, y2, _ = obj2["3d_coords"]
    return math.hypot(x2 - x1, y2 - y1)


def distance_ok_by_size(obj1, obj2, relation,
                        factor=1.5, margin=0.1,
                        front_behind_extra=0.5):
    """
    Threshold adaptativo:
    min_dist = factor * (size1 + size2) + margin

    Si relation es 'front' o 'behind', se añade un extra al umbral.
    """
    size_map = {
        "small": 0.35,
        "large": 0.7,
    }
    dist = center_distance_xy(obj1, obj2)
    s1 = size_map[obj1["size"]]
    s2 = size_map[obj2["size"]]
    min_dist = factor * (s1 + s2) + margin
    if relation in ("front", "behind"):
        min_dist += front_behind_extra
    return dist >= min_dist
     

def decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(description="Randomly subsample large Relation Datasets using Reservoir Sampling.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path for the output .json.gz file")
    parser.add_argument("-v", "--version", type=str, required=True, help="Dataset version (v1, v2, v3, v4, v5)")
    parser.add_argument("-s", "--split", type=str, required=True, help="Dataset split (train/test)")

    args = parser.parse_args()
    
    # Construcción de rutas con Path (más robusto)
    input_path = SCENES_DIR / f"CLEVR_{args.split}_scenes.json"
    if args.split == 'test':
        csv_path = FOLDER_DIR / args.version / f"{args.version}_test.csv" 
    else: 
        csv_path = FOLDER_DIR / args.version / f"train.csv"
    output_path = Path(args.output)

    # 1. Load examples using Pandas
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # 2. Initialize dictionaries
    output = {rel: [] for rel in RELATIONS}
    scenes = {rel: [] for rel in RELATIONS}

    # 3. Process with json
    if not input_path.exists():
        print(f"Error: Input JSON not found at {input_path}")
        return

    with open(input_path) as f:
        file = json.load(f)
        
    print(f"Classifying {input_path.name}...")
    
    # Classify scenes indices
    for index, scene in enumerate(file['scenes']):
        relationships = scene["relationships"]
        for relation in RELATIONS:
            if not all(elem == [] for elem in relationships[relation]):
                scenes[relation].append(index)

    print(f"scenes classified!")

    print(f"Creating pair-scenes...")

    # Get random pairs
    for relation in RELATIONS:
        df_relation = df[df['relation'] == relation]        

        for row in df_relation.itertuples():

            # choose scene randomly
            sceneID = random.choice(scenes[relation])
            scene = file['scenes'][sceneID]
            scene_objects = scene["objects"]
            rels_filtered = [i for i, elem in enumerate(scene["relationships"][relation]) if elem != []] # [elem for elem in scene["relationships"][relation] if elem != []]

            # choose objects
            obj2ID = random.choice(rels_filtered)
            rel = scene["relationships"][relation][obj2ID] #rels_filtered[obj2ID]
            obj1ID = random.choice(rel)

            while not distance_ok_by_size(scene_objects[obj1ID], scene_objects[obj2ID], relation):
                # choose scene randomly
                sceneID = random.choice(scenes[relation])
                scene = file['scenes'][sceneID]
                scene_objects = scene["objects"]
                rels_filtered = [i for i, elem in enumerate(scene["relationships"][relation]) if elem != []]

                # choose objects
                obj2ID = random.choice(rels_filtered)
                rel = scene["relationships"][relation][obj2ID] #rels_filtered[obj2ID]
                obj1ID = random.choice(rel)

            obj1 = scene["objects"][obj1ID].copy()
            obj2 = scene["objects"][obj2ID].copy()
            
            imageID = row.image.replace(".png", "")
            obj1["shape"] = row.shape1
            obj1["color"] = row.color1
            obj2["shape"] = row.shape2
            obj2["color"] = row.color2
            output[relation].append(
                {
                    "imageID": imageID,
                    "directions": scene['directions'],
                    "obj1": obj1,
                    "obj2": obj2,
                }
            )
 
    # 4. Save as  JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving output to: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
     json.dump(output, f, ensure_ascii=False, indent=4, default=decimal_to_float)
        
    print("\n🚀 Processing complete!")

if __name__ == "__main__":
    main()