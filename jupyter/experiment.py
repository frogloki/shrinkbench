import os
import argparse
import torch.multiprocessing as mp
from pathlib import Path
from shrinkbench.experiment import PruningExperiment
from causalpruner import CausalWeightsTrainerConfig, SGDPrunerConfig
from shrinkbench.plot import df_from_results, plot_df
from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader
from tests.datasets import get_dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tests.trainer import (
    DataConfig,
)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

os.environ["DATAPATH"] = "./shrinkbench/datasets/data"


def get_collate_fn(mixup_alpha: float, cutmix_alpha: float, num_classes: int):
    transforms = []

    if mixup_alpha > 0:
        transforms.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        transforms.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))

    if len(transforms) == 0:
        return None

    return v2.RandomChoice(transforms)


def delete_dir_if_exists(dir_path):
    """Deletes a directory if it exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        import shutil

        print(f"Removing existing directory: {dir_path}")
        shutil.rmtree(dir_path)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for running pruning experiments."""
    parser = argparse.ArgumentParser(
        description="ShrinkBench Pruning Experiment Runner"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["MNIST", "CIFAR10", "CIFAR100"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="tmp",
        help="Base directory to save experiment results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained model weights (if available)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training/validation dataloaders",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of fine-tuning epochs after pruning",
    )

    # These are only used if strategy is 'GlobalCausalPruning'
    parser.add_argument(
        "--num_prune_iterations",
        type=int,
        default=1,
        help="Number of iterations for causal pruning",
    )
    parser.add_argument(
        "--num_prune_epochs",
        type=int,
        default=10,
        help="Number of epochs for pruning within each iteration",
    )
    parser.add_argument(
        "--causal_pruner_train_lr",
        type=float,
        default=5e-4,
        help="Prune optimizer learning rate for Causal Pruning",
    )
    parser.add_argument(
        "--reset_weights_after_pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset_params_after_pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--causal_pruner_threaded_checkpoint_writer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--start_clean", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--causal_pruner_init_lr", type=float, default=0.1)
    parser.add_argument(
        "--causal_pruner_l1_regularization_coeff", type=float, default=1e-3
    )
    parser.add_argument("--causal_pruner_max_iter", type=int, default=30)
    parser.add_argument("--causal_pruner_loss_tol", type=float, default=1e-7)
    parser.add_argument("--causal_pruner_num_iter_no_change", type=int, default=2)
    parser.add_argument(
        "--causal_pruner_batch_size",
        type=int,
        default=256,
        help="Use -1 for full dataset",
    )
    (
        parser.add_argument(
            "--num_causal_pruner_dataloader_workers", type=int, default=1
        ),
    )
    (parser.add_argument("--num_dataloader_workers", type=int, default=1),)
    (
        parser.add_argument(
            "--causal_pruner_pin_memory",
            action=argparse.BooleanOptionalAction,
            default=False,
        ),
    )
    (
        parser.add_argument(
            "--pin_memory", action=argparse.BooleanOptionalAction, default=False
        ),
    )
    parser.add_argument(
        "--delete_checkpoint_dir_after_training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--causal_pruner_backend",
        type=str,
        default="torch",
        choices=["sklearn", "torch"],
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="./data",
        help="Directory to download datasets",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        help="Model name",
    )

    parser.add_argument(
        "--device_ids",
        type=str,
        default="-1",
        help="The device id. Useful for multi device systems",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=[
            "32",
            "16-mixed",
            "bf16-mixed",
            "16-true",
            "bf16-true",
            "transformer-engine",
        ],
    )

    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=-1,
        help="Mixup alpha for MixUp augmentations",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=-1,
        help="Mixup alpha for CutMix augmentations",
    )
    parser.add_argument(
        "--batch_size_while_pruning",
        type=int,
        default=256,
        help="Batch size while pruning",
    )
    parser.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle the train and test datasets",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        print("Running with the following arguments:")
        print(args)

    identifier = f"{args.model}_{args.dataset}"
    exp_path = Path(args.path) / identifier
    exp_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment results will be saved to: {exp_path.resolve()}")

    # 2. Prepare strategy-specific keyword arguments (`strategy_kwargs`)
    strategy_kwargs = {}

    # 3. Create the Training and Dataloader arguments for the experiment
    # train_kwargs = {"epochs": args.epochs}
    # dl_kwargs = {
    #     "batch_size": args.batch_size,
    #     "num_workers": args.num_dataloader_workers,
    #     "pin_memory": args.pin_memory,
    # }

    # print("Initializing Lightning Fabric...")
    # fabric = Fabric(
    #     devices=args.device_ids,
    #     accelerator="auto",
    #     precision=args.precision,
    # )
    # fabric.launch()

    # # 4. Instantiate and run the experiment
    # for strategy in [
    #     "LayerCausalPruning",
    #     # "GlobalMagWeight",
    #     # "LayerMagWeight",
    #     # "RandomPruning",
    # ]:
    #     print(f"Starting new strategy: {strategy}")
    #     for c in [2, 4, 8, 16, 32, 64]:
    #         # for c in [2]:
    #         if strategy == "LayerCausalPruning":
    #             print("Configuring LayerCausalPruning strategy...")
    #             prune_amount = 1.0 - (1.0 / c)
    #             print(f"{prune_amount} : This is prune amount")
    #             # train_dataset, test_dataset, num_classes = get_dataset(
    #             #     args.dataset.lower(),
    #             #     args.model,
    #             #     args.dataset_root_dir,
    #             # )
    #             # self.collate_fn = get_collate_fn(
    #             #     args.mixup_alpha, args.cutmix_alpha, num_classes=num_classes
    #             # )
    #             world_size = fabric.world_size
    #             batch_size = args.batch_size // world_size
    #             batch_size_while_pruning = args.batch_size_while_pruning // world_size

    #             # data_config = DataConfig(
    #             #     train_dataset=train_dataset,
    #             #     test_dataset=test_dataset,
    #             #     batch_size=batch_size,
    #             #     batch_size_while_pruning=batch_size_while_pruning,
    #             #     num_workers=args.num_dataloader_workers,
    #             #     pin_memory=args.pin_memory,
    #             #     shuffle=args.shuffle_dataset,
    #             #     num_classes=num_classes,
    #             #     collate_fn=collate_fn,
    #             # )
    #             # prune_dataloader = DataLoader(
    #             #     data_config.train_dataset,
    #             #     batch_size=data_config.batch_size_while_pruning,
    #             #     shuffle=data_config.shuffle,
    #             #     pin_memory=data_config.pin_memory,
    #             #     num_workers=data_config.num_workers,
    #             #     persistent_workers=data_config.num_workers > 0,
    #             # )

    #             causal_weights_trainer_config = CausalWeightsTrainerConfig(
    #                 fabric=fabric,
    #                 init_lr=args.causal_pruner_init_lr,
    #                 l1_regularization_coeff=args.causal_pruner_l1_regularization_coeff,
    #                 prune_amount=prune_amount,
    #                 max_iter=args.causal_pruner_max_iter,
    #                 loss_tol=args.causal_pruner_loss_tol,
    #                 num_iter_no_change=args.causal_pruner_num_iter_no_change,
    #                 batch_size=args.causal_pruner_batch_size,
    #                 num_dataloader_workers=args.num_causal_pruner_dataloader_workers,
    #                 pin_memory=args.causal_pruner_pin_memory,
    #                 backend=args.causal_pruner_backend,
    #             )

    #             sgd_pruner_config = SGDPrunerConfig(
    #                 fabric=fabric,
    #                 model=None,  # MUST BE MODIFIED BY PRUNING EXPERIMENT
    #                 pruner="SGDPruner",
    #                 checkpoint_dir=None,  # MUST BE MODIFIED BY PRUNING EXPERIMENT
    #                 prune_dataloader=None,
    #                 prune_optimizer_lr=args.causal_pruner_train_lr,
    #                 num_prune_iterations=args.num_prune_iterations,
    #                 num_prune_epochs=args.num_prune_epochs,
    #                 threaded_checkpoint_writer=args.causal_pruner_threaded_checkpoint_writer,
    #                 delete_checkpoint_dir_after_training=args.delete_checkpoint_dir_after_training,
    #                 trainer_config=causal_weights_trainer_config,
    #                 return_masks=True,  # Ensure masks are returned
    #                 verbose=args.verbose,
    #                 start_clean=args.start_clean,
    #                 reset_weights=args.reset_weights_after_pruning,
    #                 reset_params=args.reset_params_after_pruning,
    #             )

    #             strategy_kwargs["sgd_pruner_config"] = sgd_pruner_config
    #         exp = PruningExperiment(
    #             dataset=args.dataset,
    #             model=args.model,
    #             strategy=strategy,
    #             compression=c,
    #             seed=args.seed,
    #             dl_kwargs=dl_kwargs,
    #             train_kwargs=train_kwargs,
    #             pretrained=args.pretrained,
    #             **strategy_kwargs,
    #         )

    #         exp.run()
    # print(f"\nExperiment finished. Results saved in {exp_path.resolve()}")

    df = df_from_results('results')

    plot_df(df, 'compression', 'post_acc1', markers='strategy', fig=False, colors='strategy')


if __name__ == "__main__":
    # Recommended for PyTorch multiprocessing
    main()
