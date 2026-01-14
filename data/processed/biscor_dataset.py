# # data/processed/biscor_dataset.py

# # -----------------------------
# # DATASET
# # -----------------------------
# class VSRDataset(Dataset):
#     """
#     Visual Spatial Relations (VSR) Dataset
#     """
#     def __init__(self, dataset_name="zeroshot", split="train", data_path="data", config=None):

#         # Validations
#         self.base_path = Path(data_path) / "raw" / "vsr" #relative path
#         assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
#         assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
#         assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        
#         # Data / Images path
#         self.dataset_name = dataset_name # [zeroshot, random]
#         self.split = split

#         self.image_path = Path(self.base_path) / "images"
#         self.data_path = Path(self.base_path) / self.dataset_name  / f"{split}.jsonl"
#         self.dataset = self._load_jsonl()

#         # Input processing
#         self.transform = config["transform"]
#         self.tokenizer = config["tokenizer"]
#         self.processor = config["processor"]
#         self.params = config.get("params", {})

#     def _load_jsonl(self):
#         with open(self.data_path, "r", encoding="utf-8") as f:
#             return [json.loads(line) for line in f]
    
#     def _load_image(self, image):
#         img_path = self.image_path / image
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image not found: {img_path}")
#         return Image.open(img_path)
    
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]

#         if(self.split != "train"):
#             return {
#                 "caption": item["caption"],
#                 "negated": invert_relation(item["caption"], item["relation"], negate),
#                 "image": self._load_image(item["image"]),
#                 "label": item["label"]
#             }

#         img = self._load_image(item["image"])
#         if item['label'] == 1:
#             text = item["caption"] + ' (True)'
#         else:
#             text = item["caption"] + ' (False)'


#         # Apply image preprocessing if specified
#         if self.transform is not None:
#             img = self.transform(img)
#         img = img.convert("RGB")
        
#         # Process inputs separately (procesor + tokenizer)
#         if self.tokenizer is not None:
#             img = self.processor(img).unsqueeze(0)
#             text = self.tokenizer(text)
#             return {
#                 "image": img,
#                 "text": text
#             }

#         # Process inputs all together (general processor)
#         else:
#             inputs = self.processor(
#                 text=text,
#                 images=img,
#                 return_tensors="pt",
#                 **self.params
#             )
#         return {
#             "image": inputs['pixel_values'].squeeze(0),
#             "text": inputs['input_ids'],
#         }

#     @staticmethod
#     def compute_accuracy(logits, labels, score):
#         probs = torch.sigmoid(logits)
#         return (probs.argmax(dim=1) == labels).float().mean()