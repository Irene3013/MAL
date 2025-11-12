# test_dataset.py
from project_datasets.vsr_dataset import get_vsr_loader, VSRDataset
import pytest
import clip


# VSR DATASET

@pytest.fixture
def image_transform():
    #return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    _, preprocess = clip.load("ViT-B/32", device="cpu")#TODO change to GPU
    return preprocess
    

@pytest.mark.dataset
def test_vsr_dataset(image_transform):
    print("🔍 Loading VSR dataset...")
    loader = get_vsr_loader(split="train", dataset_name="zeroshot", batch_size=2, transform=image_transform)

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

@pytest.mark.dataset
def test_vsr_dataset_len(image_transform):
    dataset = VSRDataset(dataset_name="zeroshot", split="train", base_path="project_data/raw/vsr", transform=image_transform)
    assert len(dataset) > 0, "Dataset should not be empty"

@pytest.mark.dataset
def test_vsr_dataset_item_structure(image_transform):
    dataset = VSRDataset(dataset_name="zeroshot", split="train", base_path="project_data/raw/vsr", transform=image_transform)
    item = dataset[0]
    expected_keys = ["image", "text", "label", "relation"]
    for k in expected_keys:
        assert k in item, f"Missing key '{k}' in dataset item"
