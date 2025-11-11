# project_datasets/__init__.py

from .vsr_dataset import VSRDataset
#from .whatsup_dataset import WhatsupDataset
#from .biscor_dataset import BiscorDataset

DATASETS = {
    "vsr": VSRDataset,
    #"whatsup": WhatsupDataset,
    #"biscor": BiscorDataset,
}
