import math
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from scipy.linalg import  lstsq
from dataclasses import dataclass,field
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

#To be removed after trials 
if __name__ == "__main__":
    from datamodules import DataModule
else:
    from src.datamodules import DataModule
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    from utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
        complex_gaussian_matrix,
        awgn, decompress_complex_tensor
    )
else:
    from src.utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
        complex_gaussian_matrix,
        awgn, 
        decompress_complex_tensor
    )

# ============================================================
#
#                    CLASSES DEFINITION
#
# ============================================================

class BaseStation:
    """A class simulating a Base Station.

    Args:
        dim: int
            The dimentionality of the base station encoding space.
        antennas_transmitter : int
            The number of antennas at transmitter side.
        rho : float
            The rho coeficient for the admm method. Default 1e2.
        px_cost : int
            The transmitter power constraint. Default 1.
        channel_matrix : torch.Tensor
            The channel matrix. Set to None for the channel unaware case. Default None.
        device : str
            The device on which we run the simulation. Default "cpu".

    Attributes:
        self.<param_name> :
            The self. version of the passed parameter.
        self.agents_id : set[int]
            The set of agents IDs that are connected to this base station.
        self.agents_pilot : dict[int, torch.Tensor]
            The semantic pilots for the specific agent, already complex compressed and prewhitened.
        self.msgs : dict[str, int]
            The total messages.
        self.channel_awareness : bool
            The channel awareness of the base station.
        self.F : torch.Tensor
            The global F transformation.
        self.Fk : dict[int, torch.Tensor]
            The local F transformation of each agent.
        self.Z : torch.Tensor
            The Z parameter required by ADMM.
        self.U : torch.Tensor
            The U parameter required by Scaled ADMM.
    """

    def __init__(
        self,
        dim: int,
        antennas_transmitter: int,
        rho: float = 1e2,
        px_cost: int = 1,
        channel_matrix: torch.Tensor = None,
        device: str = 'cpu',
        snr : float = 20.0
    ) -> None:
        self.dim: int = dim
        self.antennas_transmitter: int = antennas_transmitter
        self.rho: float = rho
        self.px_cost: int = px_cost
        self.channel_matrix: torch.Tensor = channel_matrix
        self.device: str = device
        self.snr = snr
        
        # Attributes Initialization
        self.L = {}
        self.mean = {}
        self.agents_id = set()
        self.agents_pilots = {}
        self.msgs = defaultdict(int)
        self.channel_awareness = self.channel_matrix is not None

        # Set the channel matrix to the device
        if self.channel_awareness:
            self.channel_matrix = self.channel_matrix.to(device)
        #Initialize G variable for local baseline process
        self.G : dict[int:torch.Tensor] = {}
        # Initialize Global F at random and locals
        self.F = torch.randn(
            (self.antennas_transmitter, (self.dim + 1) // 2),
            dtype=torch.complex64,
        ).to(self.device)
        self.Fk = {}

        # ADMM variables
        self.Z = torch.zeros(
            self.antennas_transmitter,
            (self.dim + 1) // 2,
            dtype=torch.complex64,
        ).to(self.device)
        self.U = torch.zeros(
            self.antennas_transmitter,
            (self.dim + 1) // 2,
            dtype=torch.complex64,
        ).to(self.device)
        self.W = {idx:None for idx in range(len(self.agents_pilots))}
        return None
    
    def __str__(self) -> str:
        """A usefull string description of the base station status.

        Returns:
            description : str
                The description of the Base Station status in string format.
        """
        description = f"""Base Station Infos:
        Channel Awareness: {self.channel_awareness}.
        
        {len(self.agents_id)} agents connected:
            {self.agents_id}

        {np.sum(list(self.msgs.values()))} messages:
            {dict(self.msgs)}
        """

        return description
    
    def disjoint_alignment(self,
                           rx_pilot: torch.Tensor,
                           idx:int):
        """ Disjoint alignment of the agents to the base station by MSE;
            Baseline implementation.
            -Args: rx_pilots: RX pilots of i^th agent.
                   idx:       index of i^th agent
            -Return:None
        """
        self.W[idx] = torch.linalg.lstsq(self.agents_pilots[idx].H.unsqueeze(0), 
                                         rx_pilot.H.unsqueeze(0)).solution.squeeze(0).T
        return None 
    
    def G_step_bs(self,
                  idx:int):
        """The Baseline G-step of the BS at the idx^th agent.
           Baseline case G has dimension (d/2 x N_TX)
        
        """
        _,n = self.agents_pilots[idx].shape
        sigma= 0
        sigma = sigma_given_snr(self.snr, torch.ones(1)/math.sqrt(self.antennas_transmitter))
        X = self.agents_pilots[idx]
        HFX = self.channel_matrix @ self.F @ self.agents_pilots[idx]
        self.G[idx] = (X @ HFX.H @ torch.linalg.inv(HFX @ HFX.H + n * sigma * torch.view_as_complex(torch.stack((torch.eye(HFX.shape[0]), torch.eye(HFX.shape[0])), dim=-1)).to(self.device))).to(self.device)
        return None
    
    def F_step_bs(self, 
                idx:int):
        """ The Baseline F-step of the BS at the idx^th agent.
        """
        X = self.agents_pilots[idx]
        _, n = X.shape
        rho = self.rho * n
        O = self.G[idx] @ self.channel_matrix
        A = O.H @ O
        B = rho * torch.linalg.inv(X @ X.H)
        C = (rho * (self.Z - self.U) + O.H @ X @ X.H) @ (B/rho)

        self.Fk[idx] = torch.tensor(solve_sylvester(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()), device=self.device, dtype=self.channel_matrix.dtype)
        return None
        
    def handshake_step(
        self,
        idx: int,
        pilots: torch.Tensor,
        c: int = 1
    ) -> None:
        """Handshaking step simulation.

        Args:
            idx : int
                The id of the agent.
            pilots : torch.Tensor
                The semantic pilots for the agent.
            c : int
                The handshake cost in terms of messages. Default 1.

        Returns:
            None
        """
        pilots = pilots.T

        assert idx not in self.agents_id, (
            f'Agent of id {idx} already connected to the base station.'
        )
        assert self.dim == pilots.shape[0], (
            "The dimention of the semantic pilots doesn't match the dimetion of the base station encodings."
        )

        # Connect the agent to the base station
        self.agents_id.add(idx)

        # Compress the pilots
        compressed_pilots = complex_compressed_tensor(
            pilots, device=self.device
        )
        # Learn L and the mean for the Prewhitening
        self.agents_pilots[idx], self.L[idx], self.mean[idx] = prewhiten(
            compressed_pilots, device=self.device
        )

        # Update the number of messages used for handshaking
        self.msgs['handshaking'] += c

        return None

    def transmit_to_agent(
        self,
        idx: int,
    ) -> list[torch.Tensor]:
        """Transmit to agent i its respective FX or HFX.

        Args:
            idx : int
                The idx of the specific agent.

        Returns:
            msg : torch.Tensor
                The message for the specific agent.
        """
        assert idx in self.agents_id, (
            'The passed idx is not in the connected agents.'
        )

        # Create a message for an agent
        if self.channel_awareness:
            msg = self.channel_matrix @ self.F @ self.agents_pilots[idx]
        else:
            msg = self.F @ self.agents_pilots[idx]

        # Updating the number of trasmitting messages
        self.msgs['transmitting'] += 1

        return msg

    def group_cast(self) -> dict[int, torch.Tensor]:
        """Send a message to the whole group.

        Returns:
            grp_msgs : dict[int, torch.Tensor]
                A collection of messages to the whole agent group.
        """
        grp_msgs = {}
        for idx in self.agents_id:
            grp_msgs[idx] = self.transmit_to_agent(idx)

        return grp_msgs

    def __F_local_step(
        self,
        msg: dict[int | str, torch.Tensor],
    ) -> None:
        """The local F step for an agent.

        Args:
            msg : dict[int | str, torch.Tensor]
                The message from the agent.

        Returns:
            None
        """
        # Read the message
        idx, msg1, msg2 = msg.values()
        msg1 = msg1.to(self.device)
        msg2 = msg2.to(self.device)

        # Variables
        _, n = self.agents_pilots[idx].shape
        rho = self.rho * len(self.agents_id) * n
        B = torch.linalg.inv(
            self.agents_pilots[idx] @ self.agents_pilots[idx].H
        )
        C = (rho * (self.Z - self.U) + msg2 @ self.agents_pilots[idx].H) @ B 

        self.Fk[idx] = torch.tensor(
            solve_sylvester(
                msg1.cpu().numpy(), (rho * B).cpu().numpy(), C.cpu().numpy()
            )
        ).to(self.device)

        return None

    def __F_global_step(self) -> None:
        """The global F step, in which the base station aggregates all the local F.

        Args:
            None

        Return:
            None
        """
        self.F = torch.stack(list(self.Fk.values()), dim=0).mean(dim=0)
        return None

    def __Z_step(self) -> None:
        """The Z step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        C = self.F + self.U
        tr = torch.trace(C @ C.H).real

        if tr <= self.px_cost:
            self.Z = C
        else:
            lmb = torch.sqrt(tr / self.px_cost).item() - 1
            self.Z = C / (1 + lmb)

        return None

    def __U_step(self) -> None:
        """The U step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        self.U += self.F - self.Z
        return None

    def received_from_agent(self, msg: dict[int | str, torch.Tensor]) -> None:
        """Procedure when the base line receives a message from an agent.

        Args:
            msg : dict[int | str, torch.Tensor]
                The message from the agent.

        Returns:
            None
        """
        self.__F_local_step(msg=msg)
        return None

    def step(self) -> None:
        """The step of the base station.

        Args:
            None

        Return:
            None
        """
        self.__F_global_step()
        self.__Z_step()
        self.__U_step()
        return None


class Agent:
    """A class simulating an agent.

    Args:
        id : int
            The id of the specific agent.
        pilots : torch.Tensor
            The semantic pilots of the agent.
        antennas_receiver : int
            The number of antennas at receiver side.
        channel_matrix : torch.Tensor
            The channel matrix.
        snr : float
            The Signal to Noise Ratio of the channel. Default 20.0 dB.
        device : str
            The device on which we run the simulation. Default "cpu".

    Attributes:
        self.<param_name> :
            The self. version of the passed parameter.
        self.n_pilots : int
            The number of semantic pilots.
        self.pilot_dim : int
            The dimentionality of the semantic pilots.
        self.G:
            The personal G transformation.
    """

    def __init__(
        self,
        id: int,
        pilots: torch.Tensor,
        antennas_receiver: int,
        channel_matrix: torch.Tensor,
        snr: float = 20.0,
        device: str = 'cpu',
    ) -> None:
        self.id = id
        self.pilots, self.L, self.mean = prewhiten(
            complex_compressed_tensor(pilots.T, device=device), device=device
        )
        self.antennas_receiver: int = antennas_receiver
        self.channel_matrix: torch.Tensor = channel_matrix.to(device)
        self.snr = snr
        self.device: str = device

        assert self.channel_matrix.shape[0] == self.antennas_receiver, (
            'The number of rows of the channel matrix must be equal to the given number of receiver antennas.'
        )

        # Set Variables
        self.pilot_dim, self.n_pilots = self.pilots.shape

        # Initialize G
        self.G = None

        return None

    def __G_step(
        self,
        received: torch.Tensor,
    ) -> None:
        """The local G step of the agent.

        Args:
            received : torch.Tensor
                The message from the base station.

        Returns:
            None
        """
        sigma = 0
        if self.snr:
            sigma = sigma_given_snr(
                self.snr, torch.ones(1) / math.sqrt(self.antennas_receiver)
            )

        self.G = (
            self.pilots
            @ received.H
            @ torch.linalg.inv(
                received @ received.H + self.n_pilots * sigma * (1 + 1j)
            )
        ).to(self.device)
        return None

    def step(
        self,
        received: torch.Tensor,
    ) -> dict[int | str, torch.Tensor]:
        """The agent step.

        Args:
            received : torch.Tensor
                The message from the base station.

        Returns:
            msg : dict[int | str, torch.Tensor]
                The message to send to the base station.
        """
        received = received.to(self.device)

        # Perform the local G step
        self.__G_step(received=received)

        # Construct the message to send to the base station
        A = self.G @ self.channel_matrix
        msg = {'id': self.id, 'msg1': A.H @ A, 'msg2': A.H @ self.pilots}

        return msg
    
 
class LinearBaseline:
    def __init__(self, 
                 tx_latents : torch.Tensor,
                 rx_latents : dict[id:torch.Tensor], 
                 tx_size:int,
                 antennas_transmitter:int = 194 ,
                 antennas_receiver:int = 194 ,
                 strategy :str= None,
                 device: str = "cpu", 
                 snr: float =20.0,
                 dual_var: float = None,
                 k_f:int = 1):
        """The LinearBaselineOpt class.
        """
        self.device = torch.device(device)
        self.snr = snr
        self.tx_latents: torch.Tensor = tx_latents
        self.rx_latents: list[torch.Tensor] = rx_latents
        self.tx_dim : int = tx_size 
        self.strategy = strategy
        self.antennas_transmitter = antennas_transmitter
        self.antennas_receiver = antennas_receiver #we are in the square case
        self.L, self.mean = None, None
        self.k_f = k_f
        self.n = None
        self.power_tx = 1
        self.lmbd =  torch.tensor(25) #initial lambda value
         #---------------------------------
        self.F = torch.randn( #Nr x Nt
            (self.antennas_transmitter, self.antennas_receiver),
            dtype=torch.complex64,
        ).to(self.device)
        self.G_l : dict[int:torch.Tensor] = {idx: torch.zeros_like(self.F)
                                           for idx in range(len(rx_latents))}
        self.W = {idx:None 
                  for idx in range(len(rx_latents))} #alignment matrix 
        self.prepare_latents()
        

    def prepare_latents(self):
        """ Perform complex conversion and pre-withening.
            The tx latents space after complex compression have dimension (K*N-tx x n);
           -Inputs : None, class method
           - Outputs : None
        """ 
        #print(f"TX dimensions before compression is: {self.tx_latents.shape}")
        if self.strategy == 'FK':
            self.tx_latents = self.tx_latents[:, :self.k_f * 2 * self.antennas_transmitter]
        elif self.strategy == 'TK': #input tensor is still of the form (n x feature dimension)
            column_norm = self.tx_latents.abs().sum(dim=0)
            _, indices = torch.topk(column_norm, self.k_f * 2 * self.antennas_transmitter)
            self.tx_latents = self.tx_latents[:, indices]
            
        assert self.strategy == 'TK' or self.strategy == 'FK', f"Strategy {self.strategy} is not supported, choose TK or FK"
        # Compress the rx and tx_pilots latent spaces
        self.tx_latents = complex_compressed_tensor(
            self.tx_latents.T, device=self.device
        )
        self.rx_latents = [complex_compressed_tensor(
            self.rx_latents[i].T, device=self.device
        ) for i in range(len(self.rx_latents))]
        #print(f"TX dimensions after compression is: {self.tx_latents.shape}")
        # Learn L and the mean for the Prewhitening
        self.tx_latents, self.L, self.mean = prewhiten(
            self.tx_latents, device=self.device
        ) 
        self.n = self.tx_latents.shape[1]
        print(f"The prepared X has shape (N_TX x n)={self.tx_latents.shape}") 
        print(f"The number of samples per agent is = {self.n}")       
        print(f"The prepared Y has shape (m/2 x n)={self.rx_latents[0].shape}") 
        print('--------')                                                
        return None   

    def equalization(self,
                     channel_matrixes :dict[int:torch.Tensor]=None):
        for user in range(len(self.rx_latents)):
            self.G_step(idx=user, channel= channel_matrixes)
        self.F_step(channel=channel_matrixes)
        self.update_lambda()
        return None

    def G_step(self, 
               idx:int,
               channel :dict[int:torch.Tensor]=None):
        sigma = 0
        if self.snr:
                sigma = sigma_given_snr(self.snr, torch.ones(1)/math.sqrt(self.antennas_transmitter))
        nk = self.n * self.k_f
        HFX = channel[idx] @ self.F @ self.tx_latents
        
        self.G_l[idx] = (
        self.tx_latents @ HFX.H @ torch.linalg.inv(
            HFX @ HFX.H + nk * sigma * torch.view_as_complex(
                torch.stack((torch.eye(HFX.shape[0]), torch.eye(HFX.shape[0])), dim=-1)
                )
            )
        ).to(self.device)
        assert self.G_l[idx].shape == self.F.shape, f"Dimensions of G_l are not correct"
        return  None
        
    def F_step(self, channel: dict[int, torch.Tensor]):
        _,n = self.tx_latents.shape
        c = 1 / (n * len(self.rx_latents))
        A, B = 0, 0 
        for user in range(len(self.rx_latents)):
            GH = self.G_l[user] @ channel[user]
            A += GH.H @ GH @ self.tx_latents @ self.tx_latents.H
            B +=  self.tx_latents @ self.tx_latents.H @ GH.H 
        
        self.F = torch.linalg.inv( A + (self.lmbd * n)* torch.eye(self.antennas_transmitter, dtype= torch.complex64)) @ B
        
    def update_lambda(self, lr:float = .1):
        with torch.no_grad():
            self.lmbd.data = torch.max(
            torch.tensor(0.0, device=self.device), 
            self.lmbd + lr * (torch.trace(self.F @ self.F.H).real - self.power_tx)
        )
        return None
 
    def alignment_matrix(self):
        with torch.no_grad():
            for idx in range(len(self.rx_latents)):
                self.W[idx]= torch.linalg.lstsq(self.tx_latents.H, self.rx_latents[idx].H).solution.T
        return None

    def evaluate(self, 
                 channel:list[torch.Tensor]):
        sigma_ = sigma_given_snr(snr=self.snr, signal=torch.ones(1)/math.sqrt(self.antennas_transmitter))
        # dewithening input symbols 
        #x_dew= ( self.L @ self.tx_latents ) + self.mean 
        # transmit through the channel (no noise there)
        z = {idx:  (channel[idx] @ self.F @ self.tx_latents) + awgn(sigma=sigma_, size = (self.antennas_transmitter, self.n))
                 for idx in range(len(self.rx_latents))}
        # perform semantic decoding and alignment matrix: this is the estimated symbols at user side 
        y_hat = {idx:  (self.W[idx] @ (self.G_l[idx] @ z[idx]) )
                 for idx in range(len(self.rx_latents))}
        
        y_hat = {idx : decompress_complex_tensor(y_hat[idx])
                 for idx in range(len(self.rx_latents))}
        y_true = {idx : decompress_complex_tensor(self.rx_latents[idx])
                 for idx in range(len(self.rx_latents))}
        loss = [(torch.mean((y_true[idx] - y_hat[idx])**2)) for idx in range(len(self.rx_latents))]
        return loss
    
# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================
def main() -> None:
    """The main loop."""
    # Variables Definition
    seed: int = 42
    iterations: int = 25
    dataset: str = 'cifar10'
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    base_station_model: str =   "mobilenetv3_small_100"
    agents_models: list[str] = [
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
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
        
    Base = LinearBaseline(tx_latents= datamodules[0].train_data.z_tx,
                           rx_latents= {id : datamodule.train_data.z_rx 
                                        for id,datamodule in datamodules.items()},
                           tx_size =  datamodules[0].train_data.z_tx.shape,
                           strategy="FK",
                           antennas_receiver= antennas_receiver,
                           antennas_transmitter=antennas_transmitter
                          )
    tabl = pd.DataFrame(columns=["iteration","Trace","lambda_value", "MSE" ])
    
    mse_values = {iter: None for iter in range(iterations)}

    for _ in tqdm(range(iterations)):
        Base.alignment_matrix()
        #Perform channel equalization
        Base.equalization(channel_matrixes= channel_matrixes)
        # Evaluation step MSE
        agents_loss = Base.evaluate(channel = channel_matrixes) 
        mse_values[_] = agents_loss
        # Update result table
        tabl.loc[_] = [_+1, torch.trace(Base.F@Base.F.H).real, Base.lmbd.data, agents_loss ]
    print(tabl)
    
        # Convert mse_values dictionary to DataFrame
    df_mse = pd.DataFrame.from_dict(mse_values, orient='index')
    # Rename columns to indicate agents 
    df_mse.columns = [f'Agent {i+1}' for i in range(df_mse.shape[1])]

    plt.figure(figsize=(10, 6))
    for agent in df_mse.columns:
        plt.plot(df_mse.index, df_mse[agent], label=agent)

    plt.xlabel("Iterations")
    plt.ylabel("MSE ")
    plt.title("MSE Loss over Iterations")
    plt.legend()
    plt.grid()
    plt.show()
            
    
    return None

if __name__ == '__main__':
    main()



