# utils/data_helpers.py
import torch
import random
from utils.constants import NEGATE_RELATIONS

# CAPTION ----------------------------------------------------------------------------------

def invert_relation(caption, relation, inverse_relations=NEGATE_RELATIONS):
    """Replace the spatial relation with the oposite one."""
    if relation not in inverse_relations:
        raise ValueError(f"There is no negated relation defined for '{relation}'.")
    inverse = inverse_relations[relation]
    new_caption = caption.replace(relation, inverse, 1)
    return new_caption

# COLLATE ----------------------------------------------------------------------------------

def vsr_dual_encoder_collate(batch, config, model_name):
    """
    Collate function to evaluate VSR in Dual Encoder models (CLIP, SigLIP, Pecore)
    VSR item: {"caption": str, "negated": str, "image": PIL.Image, "label": int}
    """
    labels = []
    all_inputs = []
    
    # Input processors
    transform = config["transform"]
    processor = config["processor"]
    tokenizer = config["tokenizer"] # For PE-core
    params = config.get("params", {})
    
    for item in batch:
        caption = item["caption"]
        negation = item["negated"]
        img = item["image"]

        # Choose correct index
        correct_idx = 0 if item["label"] == 1 else 1
        labels.append(correct_idx)

        # Crop images (CLIP image transform for comparable results)
        if transform is not None:
            img = transform(img)
        img = img.convert("RGB") # secure 3 channels

        # Process each input depending on the model
        if model_name == "pecore":
            image_tensor = processor(img).unsqueeze(0)
            text_tensor = tokenizer([caption, negation])
            inputs = {"image": image_tensor, "captions": text_tensor}

        elif model_name in ["siglip", "siglip2", "clip"]:
            inputs = processor(
                text=[caption, negation],
                images=img,
                return_tensors="pt",
                **params
            )
        else:
            raise NotImplementedError()
        
        all_inputs.append(inputs)

    # Labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input": all_inputs,  
        "label": labels
    }



def whastup_dual_encoder_collate(batch, config, model_name):
    """
    Collate function to evaluate What's Up, COCO-spatial and GQA-spatial in Dual Encoder models (CLIP, SigLIP, Pecore)
    What's Up item: {"caption_options": list(str), "correct_option": str, "image": PIL.Image}
    """
    labels = []
    all_inputs = []
    
    # Input processors
    transform = config["transform"]
    processor = config["processor"]
    tokenizer = config["tokenizer"] # For PE-core
    params = config.get("params", {})
    
    for item in batch:
        options = item["caption_options"]     
        correct_caption = item["correct_option"]
        img = item["image"]

        # Choose correct index
        random.shuffle(options)
        correct_idx = options.index(correct_caption)
        labels.append(correct_idx)

        # Crop images (CLIP image transform for comparable results)
        if transform is not None:
            img = transform(img)
        img = img.convert("RGB") # secure 3 channels

        # Process each input depending on the model
        if model_name == "pecore":
            image_tensor = processor(img).unsqueeze(0)
            text_tensor = tokenizer(options)
            inputs = {"image": image_tensor, "captions": text_tensor}

        elif model_name in ["siglip", "siglip2", "clip"]:
            inputs = processor(
                text=options,
                images=img,
                return_tensors="pt",
                **params
            )
        else:
            raise NotImplementedError()
        
        all_inputs.append(inputs)

    # Labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input": all_inputs,  
        "label": labels
    }


def biscor_dual_encoder_collate(batch, config, model_name):
    """
    Collate function to evaluate BISCOR  in Dual Encoder models (CLIP, SigLIP, Pecore)
    BISCOR item: {"caption_pos": str, "caption_neg": str, "image_pos": PIL.Image, "image_neg": PIL.Image}
    """
    labels = []
    
    # Input processors
    transform = config["transform"]
    processor = config["processor"]
    tokenizer = config["tokenizer"] # For PE-core
    params = config.get("params", {})

    #for item in batch:
    item = batch[0] # batch-size=1
    img_pos = item["image_pos"]
    img_neg = item["image_neg"]
    cap_pos = item["caption_pos"]
    cap_neg = item["caption_neg"]

    #print(item)

    # Labels to evaluate
    label_t2i = [0, 1]
    label_i2t = [0, 1]
    labels.append([label_t2i, label_i2t])

        # Crop images (CLIP image transform for comparable results)
    if transform is not None:
        img_pos = transform(img_pos)
        img_neg = transform(img_neg)
    img_pos = img_pos.convert("RGB") # secure 3 channels
    img_neg = img_neg.convert("RGB") # secure 3 channels

    # Process each input depending on the model
    if model_name == "pecore":
        image_pos_tensor = processor(img_pos).unsqueeze(0)
        image_neg_tensor = processor(img_neg).unsqueeze(0)
        text_tensor = tokenizer([cap_pos, cap_neg])
        inputs = {"image": [image_pos_tensor, image_neg_tensor], "captions": text_tensor}

    elif model_name in ["siglip", "siglip2", "clip"]:
        inputs = processor(
            text=[cap_pos, cap_neg],
            images=[img_pos, img_neg],
            return_tensors="pt",
            **params
        )
    else:
        raise NotImplementedError()
    
     # Labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "input": inputs,   
        "label": labels
    }



# import random

# # Para cada par en tu dataset:
# indices = [0, 1]
# random.shuffle(indices) # A veces será [0, 1], a veces [1, 0]

# # Reordenamos textos e imágenes según los índices aleatorios
# shuffled_images = [ [img_pos, img_neg][i] for i in indices ]
# shuffled_texts = [ [cap_pos, cap_neg][i] for i in indices ]

# # El label ahora no es fijo, es el lugar donde quedó el original
# # Si el positivo (que estaba en 0) ahora está en la posición 'j', ese es tu label
# label_pos = indices.index(0) 
# label_neg = indices.index(1)
# current_ground_truth = [label_pos, label_neg]