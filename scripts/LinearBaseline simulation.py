# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))
from pytorch_lightning import seed_everything
from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix
from src.linear_models import BaseStation, Agent


def main() -> None:
    """The main loop."""
    # Variables Definition
    seed: int = 42
    iterations: int = 10
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
    for agent_id in agents:
        base_station.disjoint_alignment(agents[agent_id].pilots, agent_id)

    for _ in range(iterations):
        for agent_id in agents:
            base_station.G_step_bs(agent_id)

        for agent_id in agents:
            base_station.F_step_bs(agent_id)

        # Update F, Z, U
        base_station.step()

    return None


if __name__ == '__main__':
    main()
