# test_dataset.py
from project_datasets.vsr_dataset import get_vsr_loader, VSRDataset
import pytest


# VSR DATASET
def test_vsr_dataset():
    print("🔍 Loading VSR dataset...")
    loader = get_vsr_loader(split="train", dataset_name="zeroshot", batch_size=2) #TODO model_name 

    # get batch
    batch = next(iter(loader))
    print("✅ Dataset loaed")

    # show info
    print("Text:", batch["text"][0])
    print("Relation:", batch["relation"][0])
    print("Label:", batch["label"][0])

    # image information:
    if hasattr(batch["image"], "shape"):
        print("Image tensor shape:", batch["image"].shape)
    else:
        print("Images:", type(batch["image"][0]))

def test_vsr_dataset_len(tmp_path):
    # Asumiendo que tienes los jsonl en project_data/raw/vsr/zeroshot/
    dataset = VSRDataset(dataset_name="zeroshot", split="train", base_path="project_data/raw/vsr")
    assert len(dataset) > 0, "Dataset should not be empty"

def test_vsr_dataset_item_structure():
    dataset = VSRDataset(dataset_name="zeroshot", split="train", base_path="project_data/raw/vsr")
    item = dataset[0]
    assert all(k in item for k in ["image", "text", "label", "relation"]), "Missing expected keys in dataset item"

# VSR DATAMODULE


if __name__ == "__main__":
    test_vsr_dataset()
