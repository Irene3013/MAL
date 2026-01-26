# utils/model_helpers.py
import torch
import string
from utils.constants import PREPROCESS_TRANSFORM

def load_vision_model_components(model_name: str):
    """
    Load: model, el processor/tokenizer and specified model configuration.
    return (model, config).
    """
    config_output = {}
    
    if model_name == "clip":
        # https://huggingface.co/openai/clip-vit-base-patch32
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        config_output = {
            "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            "transform": None,
            "tokenizer": None,
            "params": {"padding": "max_length", "max_length": 64}
        }

    elif model_name in ["siglip", "siglip2"]:
        # https://huggingface.co/google/siglip-base-patch16-224
        # https://huggingface.co/google/siglip2-base-patch32-256
        from transformers import AutoModel, AutoProcessor

        model_id = "google/siglip-base-patch16-224" if model_name == "siglip" else "google/siglip2-base-patch16-224"
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="sdpa")
        config_output = {
            "processor": AutoProcessor.from_pretrained(model_id),
            "transform": PREPROCESS_TRANSFORM,
            "tokenizer": None,
            "params": {"padding": "max_length", "max_length": 64}
        }
    
    elif model_name == "pecore":
        # https://huggingface.co/facebook/PE-Core-B16-224
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as coreTransforms

        model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
        config_output = {
            "processor": coreTransforms.get_image_transform(model.image_size),
            "transform": PREPROCESS_TRANSFORM, 
            "tokenizer": coreTransforms.get_text_tokenizer(model.context_length),
            "params": None
      }
    
    elif model_name == "qwen2":
        # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct 
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
        config_output = {
            "processor": AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"),
            "transform": None, #PREPROCESS_TRANSFORM # crop images for comparable results
            "tokenizer": None,
            "params": None
        }

    elif model_name == "clip-flant5":
        # https://github.com/linzhiqiu/CLIP-FlanT5
        # https://huggingface.co/zhiqiulin/clip-flant5-xxl
        import t2v_metrics

        model = t2v_metrics.VQAScore(model='clip-flant5-xl')
        config_output = {
            "processor": None,
            "transform": None, #PREPROCESS_TRANSFORM # crop images for comparable results
            "tokenizer": None,
            "params": None
        }
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
        
    return model, config_output


# GENERATION MESSAGES ----------------------------------------------------------------------------------

def create_MC_qwen_message(imagepath, options):
    """
    Create multipple choice generation message for Qwen2-VL model.
    """
    n = len(options)
    letters = string.ascii_uppercase[:n]
    formatted_options = [
        f"{letters[i]}. {options[i]}" for i in range(n)
    ]
    options_text_block = "\n".join(formatted_options)
    instruction_suffix = " or ".join(letters)
    full_text_content = (
        "Which caption best describes the image?\n\n"
        f"{options_text_block}\n\n"
        f"Answer with one letter: {instruction_suffix}."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{imagepath}"},
                {"type": "text", "text": full_text_content},
            ],
        }
    ]

def create_qwen_message(imagepath, options):
    """
    Create generation message for Qwen2-VL model.
    """
    messages = []
    for caption in options:
        text = (
            "Caption:\n"
            f"\"{caption}\"\n\n"
            "Question:\n"
            "Does the caption accurately describe the image?\n\n"
            "Answer with one word: True or False"
        )
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file:///{imagepath}"},
                    {"type": "text", "text": text},
                ],
            }
        ])
    return messages


def create_qwen_messages_biscor(image_pos, image_neg, caption_pos, caption_neg):
    """
    Create generation message for Qwen2-VL model.
    """
    images = [image_pos, image_neg]
    captions = [caption_pos, caption_neg]
    
    messages = []
    
    # (Ip, Cp), (Ip, Cn), (In, Cp), (In, Cn)
    for img_path in images:
        for cap_text in captions:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file:///{img_path}"},
                    {"type": "text", "text": (
                        f"Caption: {cap_text}\n"
                        "Question: Does this caption accurately describe the image?\n"
                        "Answer with one word: True or False."
                    )},
                ],
            })
    print(messages)
    return messages 