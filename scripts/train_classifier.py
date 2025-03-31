"""
This python module handles the training of the classifier.
"""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import wandb
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    BatchSizeFinder,
)

from src.datamodules import DataModuleClassifier
from src.neural_models import Classifier


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


@hydra.main(
    config_path='../.conf/hydra/classifier',
    config_name='train_classifier',
    version_base='1.3',
)
def main(cfg) -> None:
    """The main script loop."""

    # Convert DictConfig to a standard dictionary for logging
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # Initialize W&B Logger
    wandb.login()
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=f'{cfg.datamodule.rx_enc}_{cfg.seed}_{cfg.datamodule.dataset}',
        id=f'{cfg.datamodule.rx_enc}_{cfg.seed}_{cfg.datamodule.dataset}',
        log_model='all',
        config=wandb_config,
    )

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Initialize the datamodule
    datamodule = DataModuleClassifier(
        dataset=cfg.datamodule.dataset,
        rx_enc=cfg.datamodule.rx_enc,
    )

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # Initialize the model
    model = Classifier(
        input_dim=datamodule.input_size,
        num_classes=datamodule.num_classes,
        hidden_dim=cfg.classifier.hidden_dim,
        lr=cfg.classifier.lr,
    )

    # Callbacks definition
    callbacks = [
        LearningRateMonitor(logging_interval='step', log_momentum=True),
        EarlyStopping(monitor='valid/loss_epoch', patience=20),
        ModelCheckpoint(monitor='valid/acc_epoch', mode='max'),
        BatchSizeFinder(mode='binsearch', max_trials=8),
    ]

    trainer = Trainer(
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # Training
    trainer.fit(model, datamodule=datamodule)

    # Testing
    trainer.test(datamodule=datamodule, ckpt_path='best')

    wandb.finish()
    return None


if __name__ == '__main__':
    main()
