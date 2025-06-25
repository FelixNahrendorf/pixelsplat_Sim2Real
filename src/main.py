import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
# from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'


import sys
import builtins

# For single GPU use
import sys
import builtins

# Block only specific MPI-related imports
original_import = builtins.__import__

def block_mpi_import(name, *args, **kwargs):
    # Only block specific MPI libraries
    mpi_libraries = [
        'mpi4py', 'mpi4py.MPI', 'openmpi', 'mpich', 'mvapich',
        'horovod', 'horovod.torch'
    ]
    
    if name in mpi_libraries or (name.startswith('mpi4py') and '.' in name):
        print(f"Blocking MPI import of {name}")
        import types
        dummy = types.ModuleType(name)
        
        # Create a dummy communicator class
        class DummyComm:
            @staticmethod
            def Get_size():
                return 1  # Single process
            @staticmethod
            def Get_rank():
                return 0  # Main process
            @staticmethod
            def barrier():
                pass
            @staticmethod
            def bcast(*args, **kwargs):
                return args[0] if args else None
        
        # Create a proper MPI dummy class
        class DummyMPI:
            COMM_WORLD = DummyComm()  # Create an actual dummy communicator
            @staticmethod
            def Init(*args, **kwargs):
                return None
            @staticmethod
            def Init_thread(*args, **kwargs):
                return None, 0
            @staticmethod
            def Finalize(*args, **kwargs):
                return None
            @staticmethod
            def Get_rank():
                return 0
            @staticmethod
            def Get_size():
                return 1
            @staticmethod
            def Is_initialized():
                return False
        
        # For mpi4py package
        if name == 'mpi4py':
            dummy.MPI = DummyMPI()
        # For mpi4py.MPI submodule
        elif name == 'mpi4py.MPI':
            for attr in dir(DummyMPI):
                if not attr.startswith('_'):
                    setattr(dummy, attr, getattr(DummyMPI, attr))
            # Make sure COMM_WORLD is properly set
            dummy.COMM_WORLD = DummyComm()
        
        sys.modules[name] = dummy
        return dummy
    
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_mpi_import



# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    print(f"Experiment Configurations:\n{cfg_dict}\n")
    
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # Set fixed GPU if specified in config
    if hasattr(cfg_dict, 'gpu') and cfg_dict.gpu.device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg_dict.gpu.device_id)
        print(cyan(f"Setting CUDA_VISIBLE_DEVICES to GPU {cfg_dict.gpu.device_id}"))
    
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    # Determine device configuration
    if hasattr(cfg_dict, 'gpu') and cfg_dict.gpu.device_id is not None:
        devices = 1  # Use single GPU when specific device is set
        strategy = "auto"
    else:
        devices = "auto"  # Use auto-detection when no specific GPU is set
        strategy = (
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        )

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches = 10,
        check_val_every_n_epoch=None,  # new code
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()