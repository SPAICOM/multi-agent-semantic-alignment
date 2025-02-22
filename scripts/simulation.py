"""
This python module computes a simulation of a Base Station communicating with a group of agents.
"""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from tqdm.auto import tqdm
from pytorch_lightning import seed_everything

from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix
from src.linear_models import BaseStation, Agent


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================
def main() -> None:
    """The main loop."""
    # Variables Definition
    seed: int = 42
    iterations: int = 20
    dataset: str = 'cifar10'
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    base_station_model: str = 'vit_tiny_patch16_224'
    agents_models: list[str] = [
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
    ]

    # Setting the seed
    seed_everything(seed, workers=True)

    # Channel Initialization
    channel_matrix = complex_gaussian_matrix(
        0, 1, (antennas_receiver, antennas_transmitter)
    )

    # Datamodules Initialization
    datamodules = {
        idx: DataModule(
            dataset=dataset, tx_enc=base_station_model, rx_enc=agent_model
        )
        for idx, agent_model in enumerate(agents_models)
    }

    for datamodule in datamodules.values():
        datamodule.prepare_data()
        datamodule.setup()

    # Agents Initialization
    agents = {
        idx: Agent(
            id=idx,
            pilots=datamodule.train_data.z_rx,
            antennas_receiver=antennas_receiver,
            channel_matrix=channel_matrix,
        )
        for idx, datamodule in datamodules.items()
    }

    # Base Station Initialization
    transmitter_dim = datamodules[0].input_size
    base_station = BaseStation(
        dim=transmitter_dim,
        antennas_transmitter=antennas_transmitter,
        channel_matrix=channel_matrix,
    )

    # Perform Handshaking
    for agent_id in agents:
        base_station.handshake_step(
            idx=agent_id, pilots=datamodules[agent_id].train_data.z_tx
        )

    # Base Station - Agent alignment
    for i in tqdm(range(iterations)):
        # Base Station transmits FX or HFX (depends if Base Station is channel aware or not)
        grp_msgs = base_station.group_cast()

        # (i) Agents performs local G and F steps
        # (ii) Agents send msg1 and msg2 to the base station
        for idx, agent in agents.items():
            a_msg = agent.step(
                grp_msgs[idx], channel_awareness=base_station.channel_awareness
            )
            base_station.received_from_agent(msg=a_msg)

        # Base Station computes global F, Z and U steps
        base_station.step()

    return None


if __name__ == '__main__':
    main()
