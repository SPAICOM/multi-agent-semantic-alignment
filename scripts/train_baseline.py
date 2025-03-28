"""
This python module computes a simulation of Baseline methods TK and FK;
"""

# Add root to the path
import sys
import pandas as pd
import typing
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

if typing.TYPE_CHECKING:
    import torch

import wandb
import hydra
import polars as pl
from tqdm.auto import tqdm
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import TensorDataset, DataLoader
from src.baseline import LinearBaseline
from types import SimpleNamespace
import hydra
from src.neural_models import Classifier
from src.utils import complex_gaussian_matrix
from src.linear_models import BaseStation, Agent
from src.download_utils import download_zip_from_gdrive
from src.datamodules import DataModule, DataModuleClassifier


def setup(
    models_path: Path,
) -> None:
    """Setup the repository:
    - downloading classifiers models.

    Args:
        models_path : Path
            The path to the models
    """
    print()
    print('Start setup procedure...')

    print()
    print('Check for the classifiers model availability...')
    # Download the classifiers if needed
    # Get from the .env file the zip file Google Drive ID
    id = dotenv_values()['CLASSIFIERS_ID']
    download_zip_from_gdrive(id=id, name='classifiers', path=str(models_path))

    print()
    print('All done.')
    print()
    return None

@hydra.main(
    config_path='../conf/train_baseline',
    config_name='train_baseline',
    version_base='1.3',
)

# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================
def main(config: DictConfig): #def main(config:dict = config_dict):
    #config = SimpleNamespace(**config)
    """The main loop."""
    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results'/'baseline'

    # Define some variables
    trainer: Trainer = Trainer(
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
    )

    # Create results directory
    RESULTS_PATH.mkdir(exist_ok=True)
    # Setup procedure
    setup(models_path=MODEL_PATH)

    # Convert DictConfig to a standard dictionary before passing to wandb
    #wandb_config = OmegaConf.to_container(
    #    config, resolve=True, throw_on_missing=True
    #)
      # Setting the seed
    seed_everything(config.seed, workers=True)
        # Channel Initialization
    channel_matrixes: dict[int, torch.Tensor] = {
        idx: complex_gaussian_matrix(
            0,
            1,
            (
                config.antennas_receiver,
                config.antennas_transmitter,
            ),
        )
        for idx, _ in enumerate(config.agents_models)
    }
    
    # Datamodules Initialization
    datamodules = {
        idx: DataModule(
            dataset=config.dataset,
            tx_enc=config.base_station_model,
            rx_enc=agent_model,
            batch_size=config.batch_size
        )
        for idx, agent_model in enumerate(config.agents_models)
    }
    
    for datamodule in datamodules.values():
        datamodule.prepare_data()
        datamodule.setup()
    
    #BaseLine class Initialization
    Base = LinearBaseline(
        tx_latents={id: datamodule.train_data.z_tx 
                    for id, datamodule in datamodules.items()},
        rx_latents={id: datamodule.train_data.z_rx 
                    for id, datamodule in datamodules.items()},
        tx_size=datamodules[0].train_data.z_tx.shape,
        strategy=config.strategy,
        device = config.device,
        snr= config.snr,
        antennas_receiver=config.antennas_receiver,
        antennas_transmitter=config.antennas_transmitter,
        channel_usage=config.channel_usage  
    )
    Base.equalization(channel_matrixes=channel_matrixes)
    mse, y_preds = Base.evaluate(channel=channel_matrixes) # mse: list of mse for each user language;
                                                           # y_preds: dictionary of {idx:y_hat} where y_hat should be used to the classification task

    
    # Classifiers Initialization
    classifiers: dict[int, Classifier] = {}
    print()
    for agent_id in tqdm(range(len(Base.rx_latents)), desc='Initialize Classifiers'):
        # Define the path towards the classifier
        clf_path: Path = (
            MODEL_PATH
            / f'classifiers/{config.dataset}/{config.agents_models[agent_id]}/seed_{config.seed}.ckpt'
        )

        # Load the classifier model
        classifiers[agent_id] = Classifier.load_from_checkpoint(clf_path)
        classifiers[agent_id].eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(
            dataset=config.dataset,
            rx_enc=config.agents_models[agent_id],
            batch_size=config.batch_size,
        )
        clf_datamodule.prepare_data()
        clf_datamodule.setup()


    # ===========================================================================
    #                 Calculating Metrics over Train Dataset
    # ===========================================================================
    losses: dict[int, float] = {}
    accuracy: dict[str, float] = {}
    dataloaders: dict[int, DataLoader] = {}
    for idx, datamodule in datamodules.items():
        # Decode the msg from the base station
        received: dict[idx:torch.Tensor] = y_preds[idx].T
        #print(f"Shape of predicted y : {y_preds[idx].shape}")
        #print(f"Shape of labels : {datamodule.train_data.labels.shape}")
        # Get accuracy
        dataloaders[idx] = DataLoader(
            TensorDataset(received, datamodule.train_data.labels),
            batch_size=config.batch_size,
        )
        accuracy[
            f'Agent-{idx} ({config.agents_models[idx]}) - Task Accuracy (Train)'
        ] = trainer.test(
            model=classifiers[idx], dataloaders=dataloaders[idx]
        )[0]['test/acc_epoch']

    # Get alignment loss
    losses[
        f'Agent-{idx} (config.agents_models[idx]) - MSE loss (Train)'
    ]: float = mse[idx]

    #for i,acc in accuracy.items():
    #    print(f"{i}:,{acc} ")

        # Save results
    pl.DataFrame(
        {
            'Dataset': config.dataset,
            'Seed': config.seed,
            'Channel Usage': config.channel_usage,
            'Antennas Transmitter': config.antennas_transmitter,
            'Antennas Receiver': config.antennas_receiver,
            'SNR': config.snr,
            'Accuracy': list(accuracy.values()),
            'Agent Model': [
                config.agents_models[int(a.split(' ')[0].split('-')[-1])]
                for a in accuracy
            ],
            'Base Station Model': config.base_station_model,
            'Case': f'{config.strategy} Baseline',
            'Latent Real Dim':Base.tx_size,
            'Latent Complex Dim':Base.tx_size/2
        }
    ).write_parquet(
        RESULTS_PATH
        / f'{config.seed}_{config.channel_usage}_{config.antennas_transmitter}_{config.antennas_receiver}_{config.snr}.parquet'
    )

    wandb.finish()
    return None

if __name__ == '__main__':
    main()