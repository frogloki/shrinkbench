# autopep8: off
import os
import sys
import argparse
from copy import deepcopy


from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from ..pruning import LayerPruning, VisionPruning, GradientMixin, ActivationMixin

from causalpruner import (
    CausalWeightsTrainerConfig,
    SGDPruner,
    SGDPrunerConfig,
)

#I can use the same class as my layer wise logic is inside sgd_pruner.py
class LayerCausalPruning(VisionPruning):
    def __init__(
        self, model, inputs=None, outputs=None, compression=1, sgd_pruner_config=None
    ):
        """
        A ShrinkBench strategy that uses the refactored SGDPruner.

        Args:
            model (nn.Module): The model to be pruned.
            fraction (float): The total fraction of weights to prune.
            sgd_pruner_config (SGDPrunerConfig): A fully populated config object.
                                                 This is the key to linking everything.
        """
        super().__init__(model, inputs, outputs, compression)

        self.sgd_pruner_config = deepcopy(sgd_pruner_config)
        self.sgd_pruner_config.return_masks_only = True
        self.sgd_pruner_config.model = self.model

    def model_masks(self):
        """
        Computes the final masks by running the iterative SGDPruner process.
        """
        print(
            f"\nStarting Global Causal Pruning for {self.sgd_pruner_config.num_prune_iterations} iterations..."
        )
        pruner = SGDPruner(self.sgd_pruner_config)
        pruner.start_pruning()
        for i in range(self.sgd_pruner_config.num_prune_iterations):
            print(
                f"  Pruning Iteration {i + 1}/{self.sgd_pruner_config.num_prune_iterations}"
            )
            pruner.run_prune_iteration()
        final_masks = pruner.get_masks()
        masks = {}
        for module_name, module in pruner.modules_dict.items():
            mask = final_masks.get(module_name)
            if mask is not None:
                masks[module] = {"weight": mask.detach().clone()}
        pruner.remove_masks()
        print("Global Causal Pruning finished.")
        return masks

