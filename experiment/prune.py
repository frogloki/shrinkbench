import json
# I'm assuming SGDPrunerConfig and other necessary imports are available in the scope
# from causalpruner import SGDPrunerConfig 

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc
from pathlib import Path





class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 **strategy_kwargs): 

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq)
        self.add_params(strategy=strategy, compression=compression)
        self.strategy_kwargs = strategy_kwargs

        self.apply_pruning(strategy, compression)

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression):
        constructor = getattr(strategies, strategy)
        x, y = next(iter(self.train_dl))
        
        strategy_init_kwargs = self.strategy_kwargs.copy()

        if strategy == 'GlobalCausalPruning':
            printc(f"Preparing special configuration for {strategy}", color='BLUE')
            
            if 'sgd_pruner_config' not in strategy_init_kwargs:
                raise ValueError("For GlobalCausalPruning, you must provide an 'sgd_pruner_config' object when initializing PruningExperiment.")
            
            config = strategy_init_kwargs['sgd_pruner_config']
            config.model = self.model
            causal_ckpt_path =  Path('_results/causal_checkpoints')
            causal_ckpt_path.mkdir(exist_ok=True)
            config.checkpoint_dir = str(causal_ckpt_path)
            strategy_init_kwargs['sgd_pruner_config'] = config
        
        self.pruning = constructor(self.model, x, y, compression=compression, **strategy_init_kwargs)
        
        self.pruning.apply()
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        # printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        if self.pruning.compression > 1:
            self.run_epochs()

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        return metrics