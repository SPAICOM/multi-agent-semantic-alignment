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

import hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix
from src.linear_models import BaseStation, Agent


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================
@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    """The main loop."""
    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Channel Initialization
    channel_matrix: torch.Tensor = complex_gaussian_matrix(
        0, 1, (cfg.antennas_receiver, cfg.antennas_transmitter)
    )

    # Datamodules Initialization
    datamodules: dict[int, DataModule] = {
        idx: DataModule(
            dataset=cfg.dataset,
            tx_enc=cfg.base_station_model,
            rx_enc=agent_model,
        )
        for idx, agent_model in enumerate(cfg.agents_models)
    }

    for datamodule in datamodules.values():
        datamodule.prepare_data()
        datamodule.setup()

    # Agents Initialization
    agents: dict[int, Agent] = {
        idx: Agent(
            id=idx,
            pilots=datamodule.train_data.z_rx,
            antennas_receiver=cfg.antennas_receiver,
            channel_matrix=channel_matrix,
            snr=cfg.snr,
            device=cfg.device,
        )
        for idx, datamodule in datamodules.items()
    }

    # Base Station Initialization
    transmitter_dim: int = datamodules[0].input_size
    base_station: BaseStation = BaseStation(
        dim=transmitter_dim,
        antennas_transmitter=cfg.antennas_transmitter,
        channel_matrix=channel_matrix,
        rho=cfg.rho,
        px_cost=cfg.px_cost,
        device=cfg.device,
    )

    # Perform Handshaking
    for agent_id in agents:
        base_station.handshake_step(
            idx=agent_id, pilots=datamodules[agent_id].train_data.z_tx
        )

    # Base Station - Agent alignment
    for i in tqdm(range(cfg.iterations)):
        # Base Station transmits FX or HFX (depends if Base Station is channel aware or not)
        grp_msgs: dict[int, torch.Tensor] = base_station.group_cast()

        # (i) Agents perform local G and F steps
        # (ii) Agents send msg1 and msg2 to the base station
        for idx, agent in agents.items():
            a_msg = agent.step(
                grp_msgs[idx], channel_awareness=base_station.channel_awareness
            )
            base_station.received_from_agent(msg=a_msg)

        # Base Station computes global F, Z, and U steps
        base_station.step()

    return None


if __name__ == '__main__':
    main()
