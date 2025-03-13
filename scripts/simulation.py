"""
This python module computes a simulation of a Base Station communicating with a group of agents.
"""

# Add root to the path
import sys
import math
from pathlib import Path
import hydra
from scipy.linalg import solve_sylvester
from hydra.core import config_store
sys.path.append(str(Path(sys.path[0]).parent))
import torch
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix
from src.linear_models import BaseStation, Agent
from src.utils import sigma_given_snr

# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================

def main() -> None:
    """The main loop."""
    # Variables Definition
    seed: int = 42
    snr = 20.0
    iterations: int = 20
    dataset: str = 'cifar10'
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    base_station_model: str =  "efficientvit_m5.r224_in1k" #'vit_tiny_patch16_224'
    agents_models: list[str] = [
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        "mobilenetv3_small_075"
        #'vit_base_patch16_224',
    ]

    # Setting the seed
    seed_everything(seed, workers=True)

    # Channel Initialization
    channel_matrixes: dict[int : torch.Tensor] = {
    idx: complex_gaussian_matrix(
        0,
        1,
        (
            antennas_receiver,
            antennas_transmitter,
        ),
    )
    for idx, _ in enumerate(agents_models)
}
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
            model_name=datamodule.rx_enc,
            antennas_receiver=antennas_receiver,
            channel_matrix=channel_matrixes[idx],
            channel_usage = 1,
            snr=snr,
            privacy= True,
            device= "cpu"
        )
        for idx, datamodule in datamodules.items()
    }
    
    # Base Station Initialization
    transmitter_dim = datamodules[0].input_size
    base_station = BaseStation(
        dim=transmitter_dim,
        antennas_transmitter=antennas_transmitter,
        channel_usage= 1,
        rho = 100,
        px_cost = 1.0,
        device = "cpu"
    )
    
    print()
    # Perform Handshaking
    for agent_id in agents:
        base_station.handshake_step(
            idx=agent_id, pilots=datamodules[agent_id].train_data.z_tx,
            channel_matrix=channel_matrixes[agent_id]
        )
    print()
    # Base Station - Agent alignment
    for i in tqdm(range(iterations)):
        # Base Station transmits FX or HFX (depends if Base Station is channel aware or not)
        grp_msgs = base_station.group_cast()

        # (i) Agents performs local G and F steps
        # (ii) Agents send msg1 and msg2 to the base station
        for idx, agent in agents.items():
            a_msg = agent.step(grp_msgs[idx],  channel_awareness=base_station.is_channel_aware(idx))
            base_station.received_from_agent(msg=a_msg)

        # Base Station computes global F, Z and U steps
        base_station.step()

        losses = {}
        total_loss = 0
        for idx, datamodule in datamodules.items():
            msg = base_station.transmit_to_agent(
                idx, datamodule.val_data.z_tx.T
            )

            loss = agents[idx].eval(
                msg.T,
                datamodule.val_data.z_rx,
                channel_awareness=base_station.is_channel_aware(idx),
            )
            losses[
                f'Agent-{idx} ({agents[idx].model_name}) - MSE loss (Val)'
            ] = loss
            total_loss += loss
        
        #print( f"Total Loss : {total_loss / len(agents)}")

    
    eval_losses = {}
    for idx, datamodule in datamodules.items():
        msg = base_station.transmit_to_agent(idx, datamodule.test_data.z_tx.T)

        loss = agents[idx].eval(
            msg.T,
            datamodule.test_data.z_rx,
            channel_awareness=base_station.is_channel_aware(idx),
        )
        eval_losses[f'Agent-{idx} ({agents[idx].model_name})'] = loss


    return None


if __name__ == '__main__':
    main()