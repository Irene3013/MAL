from project_datasets.vsr_dataset import get_vsr_loader, VSRDataset
import pytest
import clip

# VSR DATAMODULE
import pytest
from project_datasets.vsr_dataset import VSRDataModule
from torchvision import transforms

@pytest.fixture
def image_transform():
    #return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    _, preprocess = clip.load("ViT-B/32", device="cpu")#TODO change to GPU
    return preprocess

@pytest.mark.datamodule
def test_vsr_datamodule_loaders(image_transform):
    dm = VSRDataModule(dataset_name="zeroshot", batch_size=2, transform=image_transform)
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert "image" in batch and "text" in batch, "Batch missing expected keys"
