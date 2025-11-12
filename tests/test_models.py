import torch
import pytest
from models.clip import ClipModel


#CLIP MODEL
@pytest.fixture
def model():
    return ClipModel(model_name="ViT-B/32", dataset="vsr")

@pytest.mark.model
def test_clip_forward(model):
    # Fake data
    images = torch.randn(2, 3, 224, 224)
    texts = torch.randint(0, 10000, (2, 77))  # fake tokenized text
    logits = model.forward(images, texts)
    assert logits.shape == (2, 2), f"Unexpected logits shape: {logits.shape}"

@pytest.mark.model
def test_clip_loss_accuracy(model):
    logits = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 2, 3])
    loss = model.compute_loss(logits, labels)
    assert loss > 0, "Loss should be positive"
    
    #acc = model.compute_accuracy(logits, labels)
    #assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
