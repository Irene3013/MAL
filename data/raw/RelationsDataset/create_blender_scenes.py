import bpy
import os
import json
import mathutils
import argparse
import sys


MATERIAL_MAP = {
    "metal": "Rubber",#"MyMetal",
    "rubber": "Rubber"
}

    
    
# ── UTILS ─────────────────────────────────────────────────

def load_properties(path):
    with open(path) as f:
        props = json.load(f)
    color_map = {}
    for name, rgb in props["colors"].items():
        rgba = [c / 255.0 for c in rgb] + [1.0]
        color_map[name] = rgba
    return props["materials"], props["shapes"], props["sizes"], color_map


def load_materials(material_dir):
    for fn in os.listdir(material_dir):
        if not fn.endswith(".blend"):
            continue
        name = os.path.splitext(fn)[0]
        directory = os.path.join(material_dir, fn, "NodeTree")
        bpy.ops.wm.append(directory=directory, filename=name)
    
def add_object(object_dir, name, scale, loc, theta=0):
    count = sum(1 for obj in bpy.data.objects if obj.name.startswith(name))
    blend_path = os.path.join(object_dir, f"{name}.blend")
    directory  = os.path.join(blend_path, "Object")
    bpy.ops.wm.append(directory=directory, filename=name)

    new_name = f"{name}_{count}"
    bpy.data.objects[name].name = new_name
    obj = bpy.data.objects[new_name]
    bpy.context.view_layer.objects.active = obj

    x, y = loc
    obj.location          = (x, y, scale)
    obj.rotation_euler[2] = theta
    obj.scale             = (scale, scale, scale)


def add_material(name, rgba):
    bpy.ops.material.new()
    mat      = bpy.data.materials["Material"]
    mat.name = f"{name}_{len(bpy.data.materials)}"

    obj = bpy.context.active_object
    obj.data.materials.append(mat)
        
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == "Material Output":
            output_node = n
    group_node = mat.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = bpy.data.node_groups[name]
    group_node.inputs["Color"].default_value = rgba

    mat.node_tree.links.new(
        group_node.outputs["Shader"],
        output_node.inputs["Surface"],
    )


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"


def render_scene(output_path, width=224, height=224):
    
    scene = bpy.context.scene
    render = scene.render

    render.engine = "CYCLES"

    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"


    render.filepath = output_path
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100
    bpy.ops.render.render(write_still=True)


def clear_objects():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and not obj.name.startswith(("Plane","Ground")):
            bpy.data.objects.remove(obj, do_unlink=True)


def save_scene_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── SCENE BUILDING ───────────────────────────────────────────────────

def set_camera_from_directions(camera, directions):
    cam_behind = mathutils.Vector(directions['behind'])
    cam_front = -cam_behind

    quat = cam_front.to_track_quat('Z', 'Y')
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = quat


def build_scene(obj1, obj2, relation, properties_path, shape_path, swap_positions=False):
    """
    - Shape eta color: CSV-ko errenkadatik (jatorrizkoa)
    - Posizioa, tamaina, materiala: CLEVR JSON pair-etik
    """
    # Load propperties for scene
    _, object_map, size_map, color_map = load_properties(properties_path)
    
    # Shape y color de cada objeto
    shape1 = object_map[obj1['shape']]
    shape2 = object_map[obj2['shape']]
    
    color1 = color_map[obj1['color']]
    color2 = color_map[obj2['color']]
    
    size1 = size_map[obj1['size']]
    size2 = size_map[obj2['size']]

    # Posición desde el par CLEVR
    x1, y1, _ = obj1["3d_coords"]
    x2, y2, _ = obj2["3d_coords"]
    
    if swap_positions:
        x1, y1, x2, y2 = x2, y2, x1, y1
    
    # Material desde el par CLEVR
    material1 = MATERIAL_MAP[obj1.get("material", "rubber")]
    material2 = MATERIAL_MAP[obj2.get("material", "rubber")]

    add_object(shape_path, shape1, size1, (x1, y1))
    add_material(material1, color1)

    add_object(shape_path, shape2, size2, (x2, y2))
    add_material(material2, color2)
    
    return {
        "relation": relation,
        "obj1": {
            "shape":     shape1,
            "color":     color1,
            "material":  material1,
            "size":      size1,
            "3d_coords": [x1, y1, size1],
        },
        "obj2": {
            "shape":     shape2,
            "color":     color2,
            "material":  material2,
            "size":      size2,
            "3d_coords": [x2, y2, size2],
        },
        "clevr_source": {
            "obj1_index":     obj1.get("object_index", -1),
            "obj2_index":     obj2.get("object_index", -1),
        }
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    
    # Parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",           default="/content/drive/MyDrive/EHU/KISA/TFM/RelationsDataset")
    parser.add_argument("--version",        default="v1")
    parser.add_argument("--width",          type=int, default=224)
    parser.add_argument("--height",         type=int, default=224)
    parser.add_argument("--n_per_rel",      type=int, default=None)

    args = parser.parse_args()
    
    N_PER_REL       = args.n_per_rel
    BASE_SCENE      = os.path.join(args.root, "clevr/base_scene.blend")
    SHAPE_DIR       = os.path.join(args.root, "clevr/shapes")
    MATERIAL_DIR    = os.path.join(args.root, "clevr/materials")
    PROPERTIES_JSON = os.path.join(args.root, "clevr/properties.json")
    
    SPLITS = ['train', 'test']
    SPLITS = ['test']

    for split in SPLITS:
        
        img_dir  = os.path.join(args.root, args.version, f"{split}_images")
        json_dir = os.path.join(args.root, args.version, f"{split}_scenes")
    
        
        os.makedirs(img_dir,  exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        json_path = os.path.join(args.root, f"{args.version}/{args.version}_{split}.json")
        
        # Cargar JSON 
        with open(json_path) as f:
            rel_data = json.load(f)

        bpy.ops.wm.open_mainfile(filepath=BASE_SCENE)
        load_materials(MATERIAL_DIR)  
        enable_gpu()
        
        index = 0
        for rel_key in rel_data.keys():
            
            if N_PER_REL is not None:
                for key in rel_data.keys():
                    rel_data[key] = rel_data[key][:N_PER_REL]
            
            for scene in rel_data[rel_key]:
                
                # Set scene camera
                camera = bpy.data.objects['Camera']
                set_camera_from_directions(camera, scene["directions"])
                obj1, obj2 = scene["obj1"], scene["obj2"]
                
                # POSITIVE
                clear_objects()

                scene_meta = build_scene(obj1, obj2, rel_key, PROPERTIES_JSON, SHAPE_DIR)
                img_path  = os.path.join(img_dir,  f"pos_{index:06d}.png")
                json_path = os.path.join(json_dir, f"pos_{index:06d}.json")

                render_scene(img_path)
                save_scene_json(json_path, scene_meta)
                
                # NEGATIVE
                clear_objects()
                
                scene_meta = build_scene(obj1, obj2, rel_key, PROPERTIES_JSON, SHAPE_DIR, swap_positions=True)
                img_path  = os.path.join(img_dir,  f"neg_{index:06d}.png")
                json_path = os.path.join(json_dir, f"neg_{index:06d}.json")
                
                render_scene(img_path)
                save_scene_json(json_path, scene_meta)
                
                print(f"Rendered {index:06d} | {rel_key}")
                index +=1
            
            print(f" {split} done")


if __name__ == "__main__":
    main()
