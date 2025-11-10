# test_dataset.py
from project_datasets.vsr_dataset import get_vsr_loader

def test_vsr_dataset():
    print("🔍 Loading VSR dataset...")
    loader = get_vsr_loader(split="train", batch_size=2) #TODO model_name 

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

if __name__ == "__main__":
    test_vsr_dataset()
