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