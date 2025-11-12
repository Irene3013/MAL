import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models.clip import ClipModel
from project_datasets.vsr_dataset import VSRDataModule
from torchvision import transforms

@pytest.mark.integration
def test_full_training_pipeline():
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor()
    #])

    model = ClipModel(model_name="ViT-B/32")
    datamodule = VSRDataModule(dataset_name="zeroshot", batch_size=2, transform=model.preprocess)

    logger = TensorBoardLogger("logs", name="test_run", default_hp_metric=False)

    trainer = Trainer(max_epochs=1, fast_dev_run=True, logger=logger)
    trainer.test(model, dataloaders=datamodule.test_dataloader(), verbose=False)
