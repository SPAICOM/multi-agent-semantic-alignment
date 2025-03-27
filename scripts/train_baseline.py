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

config_dict = {
    "batch_size": 512,
    "device": "cpu",
    "snr": 20.0,#[-20.0,-10.0,10.0,20.0,30.0 ],
    "seed": 42, #[27, 42, 100, 123, 144, 200],
    "dataset": 'cifar10',
    "antennas_transmitter": 8, #[1, 2, 4, 8],
    "antennas_receiver": 8, #[1, 2, 4, 8],
    "base_station_model": "vit_tiny_patch16_224", #:torch.Size([192, 42500])
    "agents_models": [
                    "vit_small_patch16_224",
                    "vit_small_patch32_224",
                    "vit_base_patch16_224",
                    "vit_base_patch32_clip_224",
                    "rexnet_100",
                    "mobilenetv3_small_075",
                    "mobilenetv3_large_100",
                    "mobilenetv3_small_100",
                    "efficientvit_m5.r224_in1k",
                    "levit_128s.fb_dist_in1k"],
    "channel_usage": 8, #[1,2,4,6,8],
    "strategy": "FK"
}

# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================
def main(config:dict = config_dict):
    config = SimpleNamespace(**config)
    """The main loop."""
    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'baseline'

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
        snr= config.snr,
        antennas_receiver=config.antennas_receiver,
        antennas_transmitter=config.antennas_transmitter,
        channel_usage=config.channel_usage  # Use the current channel usage value.
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

    for i,acc in accuracy.items():
        print(f"{i}:,{acc} ")

        # Save results
    #pl.DataFrame(
    #    {
    #        'Dataset': config.dataset,
    #        'Seed': config.seed,
    #        'Channel Usage': config.channel_usage,
    #        'Antennas Transmitter': config.antennas_transmitter,
    #        'Antennas Receiver': config.antennas_receiver,
    #        'SNR': config.snr,
    #        'Accuracy': list(accuracy.values()),
    #        'Agent Model': [
    #            config.agents_models[int(a.split(' ')[0].split('-')[-1])]
    #            for a in accuracy
    #        ],
    #        'Base Station Model': config.base_station_model,
    #        'Case': 'Linear Semantic Precoding/Decoding',
    #    }
    #).write_parquet(
    #    RESULTS_PATH
    #    / f'{config.seed}_{config.channel_usage}_{config.antennas_transmitter}_{config.antennas_receiver}_{config.snr}.parquet'
    #)

    wandb.finish()
    return None

    return None

def main_runs():
    # Definisci i percorsi utili
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results' / 'baseline'
    RESULTS_PATH.mkdir(exist_ok=True)
    
    # Ciclo sui parametri: ogni run avrà seed, channel_usage e antenne (uguali per trasmettitore e ricevitore)
    for seed_val in config_dict["seed"]:
        for channel_usage_val in config_dict["channel_usage"]:
            #for snr_value in config_dict["snr"]:
              #for antennas_val in config_dict["antennas_transmitter"]:
                # Crea una copia della configurazione e aggiorna i valori scalari per questo run
                current_config = config_dict.copy()
                current_config.update({
                    "seed": seed_val,
                    "channel_usage": channel_usage_val,
                    #'snr': snr_value,
                    #"antennas_transmitter": antennas_val,
                    #"antennas_receiver": antennas_val  # antenne uguali
                })
                config = SimpleNamespace(**current_config)
    
                print(f"\nRunning experiment with seed={seed_val}, "
                      f"channel_usage={channel_usage_val},"
                      f"snr={config.snr}",
                      f"antennas (tx & rx)={config.antennas_transmitter}")
    
                # Inizializza il trainer e imposta i percorsi/modelli
                trainer = Trainer(
                    inference_mode=True,
                    enable_progress_bar=False,
                    logger=False,
                )
    
                setup(models_path=MODEL_PATH)
                seed_everything(config.seed, workers=True)
    
                # Inizializzazione del canale
                channel_matrixes = {
                    idx: complex_gaussian_matrix(
                        0,
                        1,
                        (config.antennas_receiver, config.antennas_transmitter)
                    )
                    for idx, _ in enumerate(config.agents_models)
                }
    
                # Inizializzazione dei datamodules
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
    
                # Inizializzazione della classe BaseLine
                Base = LinearBaseline(
                    tx_latents={idx: datamodule.train_data.z_tx for idx, datamodule in datamodules.items()},
                    rx_latents={idx: datamodule.train_data.z_rx for idx, datamodule in datamodules.items()},
                    tx_size=datamodules[0].train_data.z_tx.shape,
                    strategy=config.strategy,
                    snr = config.snr,
                    antennas_receiver=config.antennas_receiver,
                    antennas_transmitter=config.antennas_transmitter,
                    channel_usage=config.channel_usage  # Usa il valore corrente di channel_usage
                )
                Base.equalization(channel_matrixes=channel_matrixes)
                mse, y_preds = Base.evaluate(channel=channel_matrixes)
    
                # Inizializzazione dei Classifier
                classifiers = {}
                for agent_id in tqdm(range(len(Base.rx_latents)), desc='Initialize Classifiers'):
                    clf_path = MODEL_PATH / f'classifiers/{config.dataset}/{config.agents_models[agent_id]}/seed_{config.seed}.ckpt'
                    classifiers[agent_id] = Classifier.load_from_checkpoint(clf_path)
                    classifiers[agent_id].eval()
    
                    clf_datamodule = DataModuleClassifier(
                        dataset=config.dataset,
                        rx_enc=config.agents_models[agent_id],
                        batch_size=config.batch_size,
                    )
                    clf_datamodule.prepare_data()
                    clf_datamodule.setup()
    
                # =======================================================================
                #                 Calcolo delle metriche sul dataset di Train
                # =======================================================================
                accuracy = {}
                dataloaders = {}
                for idx, datamodule in datamodules.items():
                    # Decodifica il messaggio dalla base station
                    received = y_preds[idx].T
                    dataloaders[idx] = DataLoader(
                        TensorDataset(received, datamodule.train_data.labels),
                        batch_size=config.batch_size,
                    )
                    accuracy[f'Agent-{idx} ({config.agents_models[idx]}) - Task Accuracy (Train)'] = trainer.test(
                        model=classifiers[idx], dataloaders=dataloaders[idx]
                    )[0]['test/acc_epoch']
    
                # Stampa le metriche
                for key, acc in accuracy.items():
                    print(f"{key}: {acc}")
    
                # Salva i risultati in un file polars (formato identico a quello della funzione main() iniziale)
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
                        'Case': f'{config.strategy} Semantic Baseline',
                        'Latent Real Dim':Base.tx_size,
                        'Latent Complex Dim':Base.tx_size/2
                    }
                ).write_parquet(
                    RESULTS_PATH / f'{config.seed}_{config.channel_usage}_{config.antennas_transmitter}_{config.antennas_receiver}_{config.snr}.parquet'
                )
    
                wandb.finish()

if __name__ == '__main__':
    main()