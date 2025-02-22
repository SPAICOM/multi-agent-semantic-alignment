import math
import torch
import numpy as np
from collections import defaultdict
from scipy.linalg import solve_sylvester

if __name__ == '__main__':
    from utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
    )
else:
    from src.utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
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
    ) -> None:
        self.dim: int = dim
        self.antennas_transmitter: int = antennas_transmitter
        self.rho: float = rho
        self.px_cost: int = px_cost
        self.channel_matrix: torch.Tensor = channel_matrix
        self.device: str = device

        # Attributes Initialization
        self.L = {}
        self.mean = {}
        self.agents_id = set()
        self.agents_pilots = {}
        self.msgs = defaultdict(int)
        self.channel_awareness = self.channel_matrix is not None

        # Set the channl matrix to the device
        if self.channel_awareness:
            self.channel_matrix = self.channel_matrix.to(device)

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

    def handshake_step(
        self,
        idx: int,
        pilots: torch.Tensor,
        c: int = 1,
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


# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================


def main() -> None:
    """Some sanity tests..."""
    print('Start performing sanity tests...')
    print()

    # Variables definition

    # Latent Spaces for TX and RX

    # TODO
    print('First test...', end='\t')
    print('[Passed]')

    print()
    return None


if __name__ == '__main__':
    main()
