import json
import math
import random
import argparse
import pandas as pd  # Importante para los counts
from pathlib import Path
from decimal import Decimal
import numpy as np

# Constants
ROOT = Path("/content/drive/MyDrive/EHU/KISA/TFM/RelationsDataset")
FOLDER_DIR = ROOT / "data"
SCENES_DIR = ROOT / "original_clevr"
RELATIONS = ["left", "right", "front", "behind"]

# CLEVR usa coordenadas aprox en [-3, 3] por eje
MAX_CONE_ANGLE_DEG = 40.0   # half-angle del cono — ajusta con el widget
MIN_DISTANCE       = 1.2    # distancia mínima en coordenadas CLEVR (rango ~[-3,3])

    
def OK(scene, obj1ID, obj2ID, relation):
    """
    Valida que la relación espacial sea visualmente clara.
    
    Estrategia: calcula el ángulo entre el vector delta (p1-p2) y el
    vector de dirección canónico de la relación. Si ese ángulo es menor
    que MAX_CONE_ANGLE_DEG Y la distancia supera MIN_DISTANCE → válido.
    
    Convención CLEVR:
      relationships[relation][obj2ID] contiene obj1ID
      → obj1 está en la dirección 'relation' respecto a obj2
      → delta = p1 - p2 debe apuntar en esa dirección
    """
    obj1 = scene['objects'][obj1ID]
    obj2 = scene['objects'][obj2ID]
    directions = scene['directions']

    p1 = np.array(obj1['3d_coords'])
    p2 = np.array(obj2['3d_coords'])
    delta = p1 - p2

    # Distancia euclidiana (ignoramos z para relaciones horizontales)
    dist_2d = np.linalg.norm(delta[:2])
    if dist_2d < MIN_DISTANCE:
        return False

    # Vector de dirección canónico de la escena (ya viene normalizado en CLEVR)
    dir_vec = np.array(directions[relation])

    # Ángulo entre delta y la dirección principal
    cos_theta = np.dot(delta, dir_vec) / (np.linalg.norm(delta) * np.linalg.norm(dir_vec) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    return angle_deg <= MAX_CONE_ANGLE_DEG

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
    
    # Filter scenes indices
    for index, scene in enumerate(file['scenes']):
        relationships = scene["relationships"]
        for relation in RELATIONS:
            if not all(elem == [] for elem in relationships[relation]):
                scenes[relation].append(index)

    print(f"scenes filtered!")

    print(f"Creating pair-scenes...")

    # Get random pairs
    for relation in RELATIONS:
        df_relation = df[df['relation'] == relation]        

        for row in df_relation.itertuples():

            # choose scene randomly
            sceneID = random.choice(scenes[relation])
            scene = file['scenes'][sceneID]
            rels_filtered = [i for i, elem in enumerate(scene["relationships"][relation]) if elem != []] # [elem for elem in scene["relationships"][relation] if elem != []]

            # choose objects
            obj2ID = random.choice(rels_filtered)
            rel = scene["relationships"][relation][obj2ID] #rels_filtered[obj2ID]
            obj1ID = random.choice(rel)

            while not OK(scene, obj1ID, obj2ID, relation): # chek if the relation is clear, else resample
                # choose scene randomly
                sceneID = random.choice(scenes[relation])
                scene = file['scenes'][sceneID]
                rels_filtered = [i for i, elem in enumerate(scene["relationships"][relation]) if elem != []]

                # choose objects
                obj2ID = random.choice(rels_filtered)
                rel = scene["relationships"][relation][obj2ID] #rels_filtered[obj2ID]
                obj1ID = random.choice(rel)

            obj1 = scene["objects"][obj1ID].copy()
            obj2 = scene["objects"][obj2ID].copy()
            directions = scene['directions']
            imageID = row.image.replace(".png", "")
            obj1["shape"] = row.shape1
            obj1["color"] = row.color1
            obj2["shape"] = row.shape2
            obj2["color"] = row.color2
            output[relation].append(
                {
                    "imageID": imageID,
                    "directions": directions,
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