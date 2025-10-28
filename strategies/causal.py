# autopep8: off
import os
import sys
import argparse
from copy import deepcopy


from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)

from causalpruner import (
    CausalWeightsTrainerConfig,
    SGDPruner,
    SGDPrunerConfig,
)


class GlobalCausalPruning(VisionPruning):
    
    def __init__(self, model, inputs=None, outputs=None, compression=1, sgd_pruner_config = None):
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
        self.sgd_pruner_config.trainer_config.prune_amount = self.fraction 
        
    def model_masks(self):
        """
        Computes the final masks by running the iterative SGDPruner process.
        """
        print(f"\nStarting Global Causal Pruning for {self.sgd_pruner_config.num_prune_iterations} iterations...")
        pruner = SGDPruner(self.sgd_pruner_config)
    
        masks = None
        for i in range(self.sgd_pruner_config.num_prune_iterations):
            print(f"  Pruning Iteration {i + 1}/{self.sgd_pruner_config.num_prune_iterations}")
            masks = pruner.run_prune_iteration()
        
        print("Global Causal Pruning finished.")
        final_masks = {f"{name}.weight": mask for name, mask in masks.items()}

        model_params = self.params().keys()
        final_masks = {name: mask for name, mask in final_masks.items() if name in model_params}

        return final_masks