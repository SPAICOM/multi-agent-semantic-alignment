"""
This python module computes a simulation of a Base Station communicating with a group of agents.
"""

# Add root to the path
import sys
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


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


@hydra.main(
    config_path='../.conf/hydra/linear',
    config_name='train_linear',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main loop."""

    # Only square channel if cfg.communication.square is set to true
    if cfg.communication.square and (
        cfg.communication.antennas_receiver
        != cfg.communication.antennas_transmitter
    ):
        return None

    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results/linear_model'

    # Create results directory
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # Define some variables
    trainer: Trainer = Trainer(
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
        accelerator=cfg.device,
    )

    # Setup procedure
    setup(models_path=MODEL_PATH)

    # Convert DictConfig to a standard dictionary before passing to wandb
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # Initialize W&B and log config
    wandb.init(
        project=cfg.wandb.project,
        name=f'{cfg.seed}_{cfg.communication.channel_usage}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.train_subset_ratio}',
        id=f'{cfg.seed}_{cfg.communication.channel_usage}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.train_subset_ratio}',
        config=wandb_config,
    )

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Channels Initialization
    channel_matrixes: dict[int : torch.Tensor] = {
        idx: complex_gaussian_matrix(
            0,
            1,
            (
                cfg.communication.antennas_receiver,
                cfg.communication.antennas_transmitter,
            ),
        )
        for idx, _ in enumerate(cfg.agents.models)
    }

    # Datamodules Initialization
    datamodules: dict[int, DataModule] = {
        idx: DataModule(
            dataset=cfg.datamodule.dataset,
            tx_enc=cfg.base_station.model,
            rx_enc=agent_model,
            train_subset_ratio=cfg.datamodule.train_subset_ratio,
            batch_size=cfg.datamodule.batch_size,
        )
        for idx, agent_model in enumerate(cfg.agents.models)
    }

    for datamodule in datamodules.values():
        datamodule.prepare_data()
        datamodule.setup()

    # Agents Initialization
    print()
    agents: dict[int, Agent] = {
        idx: Agent(
            id=idx,
            pilots=datamodule.train_data.z_rx,
            model_name=datamodule.rx_enc,
            antennas_receiver=cfg.communication.antennas_receiver,
            channel_matrix=channel_matrixes[idx],
            channel_usage=cfg.communication.channel_usage,
            snr=cfg.communication.snr,
            privacy=cfg.agents.privacy,
            device=cfg.device,
        )
        for idx, datamodule in tqdm(
            datamodules.items(), desc='Agents Initialization'
        )
    }

    # Classifiers Initialization
    classifiers: dict[int, Classifier] = {}
    print()
    for agent_id in tqdm(agents, desc='Initialize Classifiers'):
        # Define the path towards the classifier
        clf_path: Path = (
            MODEL_PATH
            / f'classifiers/{cfg.datamodule.dataset}/{agents[agent_id].model_name}/seed_{cfg.seed}.ckpt'
        )

        # Load the classifier model
        classifiers[agent_id] = Classifier.load_from_checkpoint(clf_path)
        classifiers[agent_id].eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(
            dataset=cfg.datamodule.dataset,
            rx_enc=agents[agent_id].model_name,
            batch_size=cfg.datamodule.batch_size,
        )
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

    # Base Station Initialization
    transmitter_dim: int = datamodules[0].input_size
    base_station: BaseStation = BaseStation(
        model=cfg.base_station.model,
        dim=transmitter_dim,
        antennas_transmitter=cfg.communication.antennas_transmitter,
        channel_usage=cfg.communication.channel_usage,
        rho=cfg.base_station.rho,
        px_cost=cfg.base_station.px_cost,
        device=cfg.device,
    )

    # Perform Handshaking
    print()
    for agent_id in tqdm(agents, desc='Handshaking Procedure'):
        base_station.handshake_step(
            idx=agent_id,
            pilots=datamodules[agent_id].train_data.z_tx,
            channel_matrix=channel_matrixes[agent_id]
            if cfg.base_station.channel_aware
            else None,
        )

    # Base Station - Agent alignment
    print()
    for i in tqdm(range(cfg.iterations), desc='Semantic Alignment'):
        # Base Station transmits FX or HFX (depends if Base Station is channel aware or not)
        grp_msgs: dict[int, torch.Tensor] = base_station.group_cast()

        # (i) Agents perform local G and F steps
        # (ii) Agents send msg1 and msg2 to the base station
        for idx, agent in agents.items():
            a_msg: torch.Tensor = agent.step(
                grp_msgs[idx],
                channel_awareness=base_station.is_channel_aware(idx),
            )
            base_station.received_from_agent(msg=a_msg)

        # Base Station computes global F, Z, and U steps
        base_station.step()

        # ===========================================================================
        #                          Logging Metrics
        # ===========================================================================

        # Logging the trace of F during alignment
        wandb.log({'F trace': base_station.get_trace()})

        # ===========================================================================
        #                 Calculating Metrics over Train Dataset
        # ===========================================================================
        losses: dict[int, float] = {}
        accuracy: dict[str, float] = {}
        dataloaders: dict[int, DataLoader] = {}
        for idx, datamodule in datamodules.items():
            # Send the message from the base station
            msg: torch.Tensor = base_station.transmit_to_agent(
                idx, datamodule.train_data.z_tx.T
            )

            # Decode the msg from the base station
            received: torch.Tensor = agents[idx].decode(
                msg,
                channel_awareness=base_station.is_channel_aware(idx),
            )

            if cfg.metrics.train_acc:
                # Get accuracy
                dataloaders[idx] = DataLoader(
                    TensorDataset(received, datamodule.train_data.labels),
                    batch_size=cfg.datamodule.batch_size,
                )
                accuracy[
                    f'Agent-{idx} ({agents[idx].model_name}) - Task Accuracy (Train)'
                ] = trainer.test(
                    model=classifiers[idx], dataloaders=dataloaders[idx]
                )[0]['test/acc_epoch']

            # Get alignment loss
            losses[
                f'Agent-{idx} ({agents[idx].model_name}) - MSE loss (Train)'
            ]: float = agents[idx].eval(
                received,
                datamodule.train_data.z_rx,
            )

        wandb.log(losses)
        wandb.log(
            {
                'Average Agents - MSE loss (Train)': sum(list(losses.values()))
                / len(losses)
            }
        )
        if cfg.metrics.train_acc:
            wandb.log(
                {
                    'Average Agents - Task Accuracy (Train)': sum(
                        list(accuracy.values())
                    )
                    / len(accuracy)
                }
            )

        # ===========================================================================
        #                 Calculating Metrics over Val Dataset
        # ===========================================================================
        losses: dict[int, float] = {}
        accuracy: dict[str, float] = {}
        dataloaders: dict[int, DataLoader] = {}
        for idx, datamodule in datamodules.items():
            # Send the message from the base station
            msg: torch.Tensor = base_station.transmit_to_agent(
                idx, datamodule.val_data.z_tx.T
            )

            # Decode the msg from the base station
            received: torch.Tensor = agents[idx].decode(
                msg,
                channel_awareness=base_station.is_channel_aware(idx),
            )

            if cfg.metrics.val_acc:
                # Get accuracy
                dataloaders[idx] = DataLoader(
                    TensorDataset(received, datamodule.val_data.labels),
                    batch_size=cfg.datamodule.batch_size,
                )
                accuracy[
                    f'Agent-{idx} ({agents[idx].model_name}) - Task Accuracy (Val)'
                ] = trainer.test(
                    model=classifiers[idx], dataloaders=dataloaders[idx]
                )[0]['test/acc_epoch']

            # Get alignment loss
            losses[
                f'Agent-{idx} ({agents[idx].model_name}) - MSE loss (Val)'
            ]: float = agents[idx].eval(
                received,
                datamodule.val_data.z_rx,
            )

        wandb.log(losses)
        wandb.log(accuracy)
        wandb.log(
            {
                'Average Agents - MSE loss (Val)': sum(list(losses.values()))
                / len(losses)
            }
        )
        if cfg.metrics.val_acc:
            wandb.log(
                {
                    'Average Agents - Task Accuracy (Val)': sum(
                        list(accuracy.values())
                    )
                    / len(accuracy)
                }
            )

    # ==============================================================================
    #                     Evaluate over the test set
    # ==============================================================================
    losses: dict[str, float] = {}
    accuracy: dict[str, float] = {}
    dataloaders: dict[int, DataLoader] = {}
    for idx, datamodule in datamodules.items():
        # Send the message from the base station
        msg: torch.Tensor = base_station.transmit_to_agent(
            idx, datamodule.test_data.z_tx.T
        )

        # Decode the msg from the base station
        received: torch.Tensor = agents[idx].decode(
            msg,
            channel_awareness=base_station.is_channel_aware(idx),
        )

        # Get accuracy
        dataloaders[idx] = DataLoader(
            TensorDataset(received, datamodule.test_data.labels),
            batch_size=cfg.datamodule.batch_size,
        )
        accuracy[f'Agent-{idx} ({agents[idx].model_name})'] = trainer.test(
            model=classifiers[idx], dataloaders=dataloaders[idx]
        )[0]['test/acc_epoch']

        # Get alignment loss
        loss: float = agents[idx].eval(
            received,
            datamodule.test_data.z_rx,
        )
        losses[f'Agent-{idx} ({agents[idx].model_name})'] = loss

    # Create a WandB Table
    table = wandb.Table(
        data=[
            [idx, loss, acc]
            for idx, loss, acc in zip(
                losses.keys(), losses.values(), accuracy.values()
            )
        ],
        columns=['Agent', 'MSE loss (Test)', 'Task Accuracy (Test)'],
    )

    # Log the bar chart
    wandb.log(
        {
            'Agents Test Performance - MSE loss': wandb.plot.bar(
                table,
                'Agent',
                'MSE loss (Test)',
                title='Agents Test Performance - MSE loss',
            )
        }
    )

    # Log the bar chart
    wandb.log(
        {
            'Agents Test Performance - Task Accuracy': wandb.plot.bar(
                table,
                'Agent',
                'Task Accuracy (Test)',
                title='Agents Test Performance - Task Accuracy',
            )
        }
    )

    # Save results
    pl.DataFrame(
        {
            'Dataset': cfg.datamodule.dataset,
            'Training Subset Ratio': cfg.datamodule.train_subset_ratio,
            'Seed': cfg.seed,
            'Channel Usage': cfg.communication.channel_usage,
            'Antennas Transmitter': cfg.communication.antennas_transmitter,
            'Antennas Receiver': cfg.communication.antennas_receiver,
            'SNR': cfg.communication.snr,
            'Accuracy': list(accuracy.values()),
            'Loss': list(losses.values()),
            'Agent Model': [
                agents[int(a.split(' ')[0].split('-')[-1])].model_name
                for a in accuracy
            ],
            'Base Station Model': base_station.model,
            'Case': 'Federated Semantic Alignment',
            'Latent Real Dim': base_station.dim,
            'Latent Complex Dim': (base_station.dim + 1) // 2,
            'Simulation': cfg.simulation,
        }
    ).write_parquet(
        RESULTS_PATH
        / f'{cfg.seed}_{cfg.communication.channel_usage}_{cfg.communication.antennas_transmitter}_{cfg.communication.antennas_receiver}_{cfg.communication.snr}_{cfg.datamodule.dataset}_{cfg.datamodule.train_subset_ratio}_{cfg.simulation}.parquet'
    )

    wandb.finish()
    return None


if __name__ == '__main__':
    main()
