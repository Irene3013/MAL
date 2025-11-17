import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.clip import ClipModel
from project_datasets.vsr_dataset import VSRDataModule
from project_datasets.whatsup_dataset import WhatsUpDataModule, COCO_SpatialDataModule
import argparse
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
        "--root", type=str, default="/gaueko0/users/ietxarri010/MAL/project_data", help="Path to the Coco or VinVL prediction files."
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

    # Model args
    parser.add_argument(
        "--model", type=str, required=True, choices=["clip", "todo"],
        help = "Model type to be fine-tuned."
    )
    parser.add_argument(
        "--target_model", type=str, default=None, help="Model to be fine-tuned."
    )
    parser.add_argument(
        "--clip_mode", type=str, default="bin", help="Clip approach to  use"
    )

    # DataLoader args
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["vsr", "whatsup", "cocospatial"], help="Select dataset to be trained on."
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
   
    
    # For coco: "/ikerlariak/asalaberria009/datasets/mscoco"
    
    # parser.add_argument(
    #     "--visual_root", type=str, default="/ikerlariak/asalaberria009/datasets/mscoco/images", help="Path to the Coco or VinVL prediction files."
    # )
    # parser.add_argument(
    #     "--vsr_variant", type=str, default="random", choices=["random", "zero-shot"], help="Variant of the VSR dataset."
    # )
    # parser.add_argument(
    #     "--source", type=str, default="vinvl", choices=["coco", "vinvl"], help="Source of the object annotations."
    # )
    # parser.add_argument(
    #     "--location_encoding", type=str, default="none", choices= ["none", "token", "grid", "rect", "none"], help="What kind of spatial representation to use."
    # )
    # parser.add_argument(
    #     "--distractors", type=int, default=-1, help="How many objects we should use as distractors (-1: all available)."
    # )
    # parser.add_argument(
    #     "--attributes", action="store_true", help="Use VinVL attributes for image descriptions."
    # )
    # parser.add_argument(
    #     "--spatial_val_file", type=str, default="/gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-noattr.json", help="Use an already prepared spatial validation file; if None, it will be generated on the fly."
    # )
    # parser.add_argument(
    #     "--tiny", action="store_true", help="Use tiny version of the dataset for development."
    # )
    # parser.add_argument(
    #     "--grid_size", type=int, default=32, help="The size of the grid for the location encoding."
    # )


    args = parser.parse_args()
    return args




def main_program():
    print("Parsing args...")
    args = parse_args()
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model
    print("Loading model...")

    if (args.model == "clip"):
        if args.ckpt is None:
            model = ClipModel(args)
        else:
            model = ClipModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=True) #antes era false
        model.float()
    else: 
        sys.exit()

    print("Model loaded!")

     # Load data
    print("Loading data...")

    if args.dataset == "vsr":
        datamodule = VSRDataModule(args, transform=model.preprocess)
    else:
        if args.dataset == "whatsup":
            datamodule = WhatsUpDataModule(args, transform=model.preprocess)
        elif args.dataset == "cocospatial":
            datamodule = COCO_SpatialDataModule(args, transform=model.preprocess)
        else:
            datamodule = VSRDataModule(args, transform=model.preprocess)
        
        # ZeroShot en WhatsUp
        args.train = False
        args.evaluate = False
       
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
    model.train()
    if args.train:
        print("Training starts!")
        model.train()
        trainer.fit(model, datamodule)
        print("Training finished!")

    # Evaluate model
    model.eval()
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