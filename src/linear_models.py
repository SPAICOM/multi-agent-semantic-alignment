import math
import torch
import numpy as np
from collections import defaultdict
from scipy.linalg import solve_sylvester

if __name__ == '__main__':
    from utils import (
        complex_compressed_tensor,
        decompress_complex_tensor,
        complex_gaussian_matrix,
        prewhiten,
        awgn,
        sigma_given_snr,
        a_inv_times_b,
    )
else:
    from src.utils import (
        complex_compressed_tensor,
        decompress_complex_tensor,
        prewhiten,
        awgn,
        sigma_given_snr,
        a_inv_times_b,
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
        channel_usage : int
            The channel usage of the communication. Default 1.
        rho : float
            The rho coeficient for the admm method. Default 1e-1.
        px_cost : int
            The transmitter power constraint. Default 1.
        device : str
            The device on which we run the simulation. Default "cpu".

    Attributes:
        self.<param_name> :
            The self. version of the passed parameter.
        self.L : dict[int, torch.Tensor]
            The L matrix for prewhitening the messages for a specific agent.
        self.mean : dict[int, float]
            The mean needed for prewhitening the messages for a specific agent.
        self.agents_id : set[int]
            The set of agents IDs that are connected to this base station.
        self.agents_pilot : dict[int, torch.Tensor]
            The semantic pilots for the specific agent, already complex compressed and prewhitened.
        self.msgs : dict[str, int]
            The total messages.
        self.channel_matrixes : dict[int, torch.Tensor]
            The channel matrix for each agent.
        self.F : torch.Tensor
            The global F transformation.
        self.F_agent : dict[int, torch.Tensor]
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
        channel_usage: int = 1,
        rho: float = 1e-1,
        px_cost: int = 1,
        device: str = 'cpu',
    ) -> None:
        self.dim: int = dim
        self.antennas_transmitter: int = antennas_transmitter
        self.channel_usage: int = channel_usage
        self.rho: float = rho
        self.px_cost: int = px_cost
        self.device: str = device

        # Attributes Initialization
        self.L: dict[int, torch.Tensor] = {}
        self.mean: dict[int, float] = {}
        self.agents_id: set[int] = set()
        self.agents_pilots: dict[int, torch.Tensor] = {}
        self.msgs: defaultdict = defaultdict(int)
        self.channel_matrixes: dict[int, torch.Tensor] = {}

        # Initialize Global F at random and locals
        self.F = torch.randn(
            (
                self.antennas_transmitter * self.channel_usage,
                (self.dim + 1) // 2,
            ),
            dtype=torch.complex64,
        ).to(self.device)
        self.F_agent = {}

        # ADMM variables
        self.Z = torch.zeros_like(self.F)
        self.U = torch.zeros_like(self.F)

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
        channel_matrix: torch.Tensor,
        c: int = 1,
    ) -> None:
        """Handshaking step simulation.

        Args:
            idx : int
                The id of the agent.
            pilots : torch.Tensor
                The semantic pilots for the agent.
            channel_matrix : torch.Tensor
                The channel matrix of the communication.
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

        # Add channel matrix
        if channel_matrix is None:
            self.channel_matrixes[idx] = channel_matrix
        else:
            self.channel_matrixes[idx] = torch.kron(
                torch.eye(self.channel_usage, dtype=torch.complex64),
                channel_matrix,
            ).to(self.device)

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

    def get_trace(self) -> float:
        """Get the trace of the global F matrix.

        Args:
            None

        Return:
            float
                The trace of the global F.
        """
        return torch.trace(self.F.H @ self.F).real.item()

    def __compression_and_prewhitening(
        self,
        msg: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        """A private module used to complex compress and prewhite a message to transmit.

        Args:
            msg : torch.Tensor
                The message to compress and prewhite.
            idx: int
                The idx of the agent the base line is interested to transmit to.

        Returns:
            msg : torch.Tensor
                The message compressed and prewhitened.
        """
        # Complex compression
        msg = complex_compressed_tensor(msg, device=self.device)

        # Prewhitening step
        msg = a_inv_times_b(self.L[idx], msg - self.mean[idx])
        return msg

    def transmit_to_agent(
        self,
        idx: int,
        msg: torch.Tensor,
        alignment: bool = False,
    ) -> list[torch.Tensor]:
        """Transmit to agent i its respective FX or HFX.

        Args:
            idx : int
                The idx of the specific agent.
            msg : torch.Tensor
                A message to send to an agent.

        Returns:
            msg : torch.Tensor
                The message for the specific agent.
        """
        assert idx in self.agents_id, (
            'The passed idx is not in the connected agents.'
        )

        if not alignment:
            msg = self.__compression_and_prewhitening(msg, idx)

        # Encode the message
        msg = self.F @ msg

        # Transmit over the channel
        if self.is_channel_aware(idx):
            msg = self.channel_matrixes[idx] @ msg

        # Updating the number of trasmitting messages
        self.msgs['transmitting'] += 1

        return msg

    def group_cast(self) -> dict[int, torch.Tensor]:
        """Send a message to the whole group.

        Args:
            None

        Returns:
            grp_msgs : dict[int, torch.Tensor]
                A collection of messages to the whole agent group.
        """
        grp_msgs = {}
        for idx in self.agents_id:
            grp_msgs[idx] = self.transmit_to_agent(
                idx, self.agents_pilots[idx], alignment=True
            )

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

        self.F_agent[idx] = torch.tensor(
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
        # Check if all agents did transmit their message
        assert len(self.F_agent) == len(self.agents_id), (
            f'The following agents did not communicate with the base station:\n\t{self.agents_id - set(self.F_agent.keys())}'
        )

        # Performe aggregation of the F
        self.F = torch.stack(list(self.F_agent.values()), dim=0).mean(dim=0)

        # Clean local F
        self.F_agent = {}

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

    def is_channel_aware(
        self,
        idx: int,
    ) -> bool:
        """Check if the communication between the base station and agent idx is channel aware or not.

        Args:
            idx : int
                The index of the agent.

        Returns:
            bool
                If the base station for this communication is channel aware or not.
        """
        return self.channel_matrixes[idx] is not None

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
        model_name : str
            The name of the encoding model of the agent.
        antennas_receiver : int
            The number of antennas at receiver side.
        channel_matrix : torch.Tensor
            The channel matrix.
        channel_usage : int
            The channel usage of the communication. Default 1.
        snr : float
            The Signal to Noise Ratio of the channel. Default 20.0 dB.
        privacy : bool
            If the agent performs the privatization of the pilots or not. Default True.
        device : str
            The device on which we run the simulation. Default "cpu".

    Attributes:
        self.<param_name> :
            The self. version of the passed parameter.
        self.n_pilots : int
            The number of semantic pilots.
        self.pilot_dim : int
            The dimentionality of the semantic pilots.
        self.sigma : int
            The sigma of the additive white gaussian noise.
        self.G:
            The personal G transformation.
    """

    def __init__(
        self,
        id: int,
        pilots: torch.Tensor,
        model_name: str,
        antennas_receiver: int,
        channel_matrix: torch.Tensor,
        channel_usage: int = 1,
        snr: float = 20.0,
        privacy: bool = True,
        device: str = 'cpu',
    ) -> None:
        self.id = id
        self.antennas_receiver: int = antennas_receiver
        self.channel_matrix: torch.Tensor = torch.kron(
            torch.eye(channel_usage, dtype=torch.complex64), channel_matrix
        ).to(device)
        self.channel_usage: int = channel_usage
        self.model_name: str = model_name
        self.snr: float = snr
        self.privacy: bool = privacy
        self.device: str = device

        assert (
            self.channel_matrix.shape[0]
            == self.antennas_receiver * self.channel_usage
        ), (
            'The number of rows of the channel matrix must be equal to the given number of receiver antennas.'
        )

        if self.privacy:
            self.pilots, self.L, self.mean = prewhiten(
                complex_compressed_tensor(pilots.T, device=device),
                device=device,
            )
        else:
            self.pilots = complex_compressed_tensor(pilots.T, device=device)
            self.L = torch.eye(self.pilots.shape[0], dtype=self.pilots.dtype)
            self.mean = 0

        # Set Variables
        self.pilot_dim, self.n_pilots = self.pilots.shape

        if self.snr:
            self.sigma = sigma_given_snr(
                self.snr, torch.ones(1) / math.sqrt(self.antennas_receiver)
            )

            if self.channel_usage > 0:
                self.sigma /= self.channel_usage
        else:
            self.sigma = 0

        # Initialize G
        self.G = None

        return None

    def __dewhitening_and_decompression(
        self,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        """A private module to handle both prewhitening removal and decompression of a message.

        Args:
            msg : torch.Tensor
                The message to clean and decompress.

        Returns:
            msg : torch.Tensor
                The final cleaned message.
        """
        # Remove whitening
        msg = self.L @ msg + self.mean

        # Decompress
        msg = decompress_complex_tensor(msg, device=self.device)
        return msg

    def __G_step(
        self,
        received: torch.Tensor,
        channel_awareness: bool,
    ) -> None:
        """The local G step of the agent.

        Args:
            received : torch.Tensor
                The message from the base station.

        Returns:
            None
        """
        if not channel_awareness:
            received = self.channel_matrix @ received

        self.G = (
            self.pilots
            @ received.H
            @ torch.linalg.inv(
                received @ received.H + self.n_pilots * self.sigma * (1 + 1j)
            )
        ).to(self.device)
        return None

    def step(
        self,
        received: torch.Tensor,
        channel_awareness: bool,
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
        self.__G_step(received=received, channel_awareness=channel_awareness)

        # Construct the message to send to the base station
        A = self.G @ self.channel_matrix
        msg = {'id': self.id, 'msg1': A.H @ A, 'msg2': A.H @ self.pilots}

        return msg

    def decode(
        self,
        msg: torch.Tensor,
        channel_awareness: bool,
    ) -> torch.Tensor:
        """Decode the incoming message from the base station.

        Args:
            msg : torch.Tensor
                The incoming message from the base station.
            channel_awareness : bool
                The awareness of the base station about the channel state.

        Returns:
            msg : torch.Tensor
                The decoded message.
        """
        msg = msg.T

        # Pass through the channel if base station was not aware
        if not channel_awareness:
            msg = self.channel_matrix @ msg

        # Additive White Gaussian Noise
        if self.snr:
            w = awgn(
                sigma=self.sigma,
                size=msg.shape,
                device=self.device,
            )
            msg += w

        # Decode the message
        msg = self.G @ msg

        # Clean the message
        msg = self.__dewhitening_and_decompression(msg)
        return msg.T

    def eval(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        channel_awareness: bool,
    ) -> float:
        """Eval an input given an expected output.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            channel_awareness : bool
                The awareness of the base station about the channel state.

        Returns:
            float
                The mse loss.
        """
        assert self.G is not None, (
            'You have to first align the agent with the base station.'
        )

        input = input.to(self.device)
        output = output.to(self.device)

        decoded = self.decode(input, channel_awareness=channel_awareness)
        return torch.nn.functional.mse_loss(
            decoded, output, reduction='mean'
        ).item()


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
    n: int = 100
    iterations: int = 1
    tx_dim: int = 384
    rx_dim: int = 768
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    channel_matrix: torch.Tensor = complex_gaussian_matrix(
        mean=0, std=1, size=(antennas_receiver, antennas_transmitter)
    )

    # Get the semantic pilots
    tx_pilots: torch.Tensor = torch.randn(n, tx_dim)
    rx_pilots: torch.Tensor = torch.randn(n, rx_dim)

    print('First test...', end='\t')
    # Agents Initialization
    agents = {
        0: Agent(
            id=0,
            pilots=rx_pilots,
            antennas_receiver=antennas_receiver,
            channel_matrix=channel_matrix,
        )
    }

    # Base Station Initialization
    base_station = BaseStation(
        dim=tx_dim,
        antennas_transmitter=antennas_transmitter,
    )

    # Perform Handshaking
    for agent_id in agents:
        base_station.handshake_step(
            idx=agent_id, pilots=tx_pilots, channel_matrix=channel_matrix
        )

    # Base Station - Agent alignment
    for i in range(iterations):
        # Base Station transmits FX or HFX (depends if Base Station is channel aware or not)
        grp_msgs = base_station.group_cast()

        # (i) Agents performs local G and F steps
        # (ii) Agents send msg1 and msg2 to the base station
        for idx, agent in agents.items():
            a_msg = agent.step(
                grp_msgs[idx],
                channel_awareness=base_station.is_channel_aware(idx),
            )
            base_station.received_from_agent(msg=a_msg)

        # Base Station computes global F, Z and U steps
        base_station.step()
    print('[Passed]')

    print()
    return None


if __name__ == '__main__':
    main()
