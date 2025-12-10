import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.dual_encoder import DualEncoder
from project_datasets.vsr_dataset import VSRDataModule
from project_datasets.whatsup_dataset import WhatsUpDataModule
import argparse
import torch.serialization
import sys


## Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/MAL/data", help="Path to the data files."
    )
    parser.add_argument(
        "--image_path", type=str, default="/gaueko0/users/ietxarri010/MAL/data", help="Path to the image files if its different from the annotations files."
    )
    parser.add_argument(
        "--output_path", type=str, default="/gaueko0/users/ietxarri010/out/", help="Output directory for plots and models."
    )
    parser.add_argument(
        "--train", action="store_true", help="Fine-tune model."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Test model after fine-tuning."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )
    parser.add_argument(
        "--score", type=str, default="precision", choices=["precision", "pair-wise", "set-wise"], help = "Method to compute score."
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True, choices=["clip", "siglip", "siglip2", "pecore"],
        help = "Model type to be fine-tuned."
    )
    parser.add_argument(
        "--target_model", type=str, default=None, help="Model to be fine-tuned."
    )
    
    # DataLoader args
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["vsr", "whatsup", "cocospatial", "gqaspatial"], help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Select dataset variant to be trained on."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Workers used in the dataloader." 
    )

    # Trainer args
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps. (1 == do not use gradient accumulation)"
    )
    parser.add_argument(
        "--scheduler_off", action="store_true", help="Do not use any scheduler"
    )
    parser.add_argument(
        "--deepspeed", action="store_true", help="Use deepspeed stage-2 offload strategy."
    )
    parser.add_argument(
        "--val_check_interval", type=float, default=1.0, help="How often within a training epoch to check the val set. (1.0 == every epoch)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--precision", type=int, default=32, choices=[16, 32, 64], help="Precision for the GPUs."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=88000, help="Steps to be done during training."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )
    
    args = parser.parse_args()
    return args


def main_program():
    torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model
    print("Loading model...")

    if args.model in ["clip", "siglip", "siglip2"]:
        if args.ckpt is None:
            model = DualEncoder(args)
        else:
            model = DualEncoder.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=True) #antes era false
        model.float()
    else: 
        sys.exit()

    print("Model loaded!")

     # Load data
    print("Loading data...")

    if args.dataset == "vsr":
        datamodule = VSRDataModule(args, config=model.config)
       
    elif args.dataset in ['whatsup', 'cocospatial', 'gqaspatial']:
        datamodule = WhatsUpDataModule(args, config=model.config)

        # ZeroShot en WhatsUp
        args.train = False
        args.evaluate = True
    else:
        raise NotImplementedError
    
    # Setup datamodule
    datamodule.setup()   
       
    print("Data loaded!")

     # Define checkpoint filename and tensorboard run name
    if args.run_name == None:
        print('A run name has to be provided')
        sys.exit()

    tb_run_name = args.run_name
    print(f'Run name: {tb_run_name}')
    
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)

     # Use ModelCheckPoint to store best validation model
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path, 
        monitor='val_accuracy', 
        mode='max', 
        filename=tb_run_name + "-{epoch:02d}-{val_accuracy:.2f}", 
        save_weights_only=True, 
        save_top_k=1)
    
    # Prepare Trainer
    trainer = pl.Trainer(
                devices=args.gpus, 
                fast_dev_run=False, 
                logger=logger, 
                max_steps=args.max_steps, 
                accumulate_grad_batches=args.accumulate_grad_batches, 
                strategy= 'auto', 
                precision=args.precision, 
                callbacks=[checkpoint_callback]
            )
    
    # Train model
    if args.train:
        model.train()
        print("Training starts!")
        model.train()
        trainer.fit(model, datamodule)
        print("Training finished!")

    # Evaluate model
    if args.evaluate and args.train:
        print(f'Loading {checkpoint_callback.best_model_path} with val accuracy of {checkpoint_callback.best_model_score} to test')
        print('Testing starts!')
        model.eval()
        trainer.test(ckpt_path = 'best', dataloaders=datamodule.test_dataloader(), verbose=False)
        print('Testing finished!')

    elif args.evaluate and not args.train:
        print('Testing starts!')
        model.eval()
        trainer.test(model=model, dataloaders=datamodule.test_dataloader(), verbose=False)
        print('Testing finished!')
    
    return 0

if __name__ == "__main__":
    main_program()