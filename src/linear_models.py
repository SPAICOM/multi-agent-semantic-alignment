import math
import torch
import numpy as np
from collections import defaultdict
from scipy.linalg import solve_sylvester

if __name__ == '__main__':
    from utils import (
        awgn,
        prewhiten,
        a_inv_times_b,
        sigma_given_snr,
        mmse_svd_equalizer,
        complex_gaussian_matrix,
        complex_compressed_tensor,
        decompress_complex_tensor,
    )
else:
    from src.utils import (
        awgn,
        prewhiten,
        a_inv_times_b,
        sigma_given_snr,
        mmse_svd_equalizer,
        complex_compressed_tensor,
        decompress_complex_tensor,
    )


# ============================================================
#
#                    CLASSES DEFINITION
#
# ============================================================


class BaseStation:
    """A class simulating a Base Station.

    Args:
        model : str
            The model name of the base station.
        dim : int
            The dimentionality of the base station encoding space.
        antennas_transmitter : int
            The number of antennas at transmitter side.
        channel_usage : int
            The channel usage of the communication. Default 1.
        rho : float
            The rho coeficient for the admm method. Default 1e-1.
        px_cost : float
            The transmitter power constraint. Default 1.0.
        device : str
            The device on which we run the simulation. Default "cpu".
        status : str
            The stream status of BS,  "multi-link" or "shared". Default "shared".

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
        model: str,
        dim: int,
        antennas_transmitter: int,
        channel_usage: int = 1,
        rho: float = 1e-1,
        px_cost: float = 1.0,
        device: str = 'cpu',
        status: str = 'shared',
    ) -> None:
        self.model: str = model
        self.dim: int = dim
        self.antennas_transmitter: int = antennas_transmitter
        self.channel_usage: int = channel_usage
        self.rho: float = rho
        self.px_cost: float = px_cost
        self.device: str = device
        self.status: str = status

        assert self.status in {'multi-link', 'shared'}, 'The passed status is not one of the available.'

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
        if self.status=='shared':
            self.Z = torch.zeros_like(self.F)
            self.U = torch.zeros_like(self.F)
        elif status == 'multi-link':
            self.Z = {}
            self.U = {}

        return None

    def __str__(self) -> str:
        """A usefull string description of the base station status.

        Returns:
            description : str
                The description of the Base Station status in string format.
        """
        description = f"""Base Station Infos:
        Status : :{self.status},
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
            "The dimention of the semantic pilots doesn't match the dimention of the base station encodings."
        )

        # Connect the agent to the base station
        self.agents_id.add(idx)

        # Populate the initialization of F_agent, Z_agent and U_agent
        if self.status == 'multi-link':
            self.F_agent[idx] = self.F.clone()
            self.Z[idx] = torch.zeros_like(self.F)
            self.U[idx] = torch.zeros_like(self.F)

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
            alignment : bool
                Set to True if the Base Station is in alignment mode. Default False.

        Returns:
            msg : torch.Tensor
                The message for the specific agent.
        """
        assert idx in self.agents_id, (
            'The passed idx is not in the connected agents.'
        )

        if not alignment:
            msg = self.__compression_and_prewhitening(msg, idx)
        
        #Treat the multi-link scenario differently 
        if self.status == 'multi-link':
            msg = self.F_agent[idx] @ msg

        elif self.status == 'shared':
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
        msg: dict[str, torch.Tensor],
    ) -> None:
        """The local F step for an agent.

        Args:
            msg : dict[str, torch.Tensor]
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
        rho = self.rho * n
        B = torch.linalg.inv(
            self.agents_pilots[idx] @ self.agents_pilots[idx].H
        )

        if self.status == 'multi-link':
            C = (rho * (self.Z[idx] - self.U[idx]) + msg2 @ self.agents_pilots[idx].H) @ B
        else:
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
            f'The following agents are not registered in the base station:\n\t{self.agents_id - set(self.F_agent.keys())}'
        )

        # Performe aggregation of the F
        self.F = torch.stack(list(self.F_agent.values()), dim=0).mean(dim=0)

        # Clean local F
        if self.status != 'multi-link':
            self.F_agent = {}

        return None

    def __Z_step(self) -> None:
        """The Z step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        match self.status:
            case 'multi-link':
               #Z step for multi-link mode: dictionary-based Z[idx].
                for idx in self.agents_id:
                    C = self.F_agent[idx] + self.U[idx]
                    tr = torch.trace(C @ C.H).real

                    if tr <= self.px_cost:
                        self.Z[idx] = C
                    else:
                        lambd = torch.sqrt(tr / self.px_cost).item() - 1.0
                        self.Z[idx] = C / (1.0 + lambd)

            case 'shared':
                C = self.F + self.U
                tr = torch.trace(C @ C.H).real

                if tr <= self.px_cost:
                    self.Z = C
                else:
                    lmb = torch.sqrt(tr / self.px_cost).item() - 1
                    self.Z = C / (1 + lmb)

            case _:
                raise Exception('The passed status is not available.')

        return None

    def __U_step(self) -> None:
        """The U step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        match self.status:
            case 'multi-link':
              for idx in self.agents_id:
                self.U[idx] += self.F_agent[idx] - self.Z[idx]

            case 'shared':
                self.U += self.F - self.Z

            case _:
                raise Exception('The passed status is not available.')

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
            msg : dict[str, torch.Tensor]
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
        if self.status == 'shared':
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
            self.sigma = (
                sigma_given_snr(
                    self.snr, torch.ones(1) / math.sqrt(self.antennas_receiver)
                )
                / self.channel_usage
            )
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
            channel_awareness : bool
                If the baseline is channel aware or not.

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
    ) -> dict[str, torch.Tensor]:
        """The agent step.

        Args:
            received : torch.Tensor
                The message from the base station.
            channel_awareness : bool
                If the baseline is channel aware or not.

        Returns:
            msg : dict[str, torch.Tensor]
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
    ) -> float:
        """Eval an input given an expected output.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.

        Returns:
            float
                The mse loss.
        """
        assert self.G is not None, (
            'You have to first align the agent with the base station.'
        )

        input = input.to(self.device)
        output = output.to(self.device)

        return (
            torch.nn.functional.mse_loss(
                input, output, reduction='mean'
            ).item()
            / self.channel_usage
        )


class BaseStationBaseline(BaseStation):
    """A class simulating a Baseline Base Station.

    Args:
        model : str
            The model name of the base station.
        dim : int
            The dimentionality of the base station encoding space.
        antennas_transmitter : int
            The number of antennas at transmitter side.
        channel_usage : int
            The channel usage of the communication. Default 1.
        px_cost : float
            The transmitter power constraint. Default 1.0.
        lr : float
            The learning rate for estimating the right F subject to the power constraint. Default 1e-2.
        strategy : str
            The strategy to choose the packets, possible values 'First-K' or 'Top-K'. Default 'First-K'.
        iterarions : int
            The number of iterations for the constraint. Default 30.
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
        self.channel_matrixes : dict[int, torch.Tensor]
            The channel matrix for each agent.
        self.F : torch.Tensor
            The global F transformation.
        self.F_agent : dict[int, torch.Tensor]
            The local F transformation of each agent.
            The U parameter required by Scaled ADMM.
        self.agents_pilots : dict[int, dict[str, torch.Tensor | int]]
            A dictionary containing the info about the Base Station pilots for a specific agent.
    """

    def __init__(
        self,
        model: str,
        dim: int,
        antennas_transmitter: int,
        channel_usage: int = 1,
        px_cost: float = 1.0,
        lr: float = 1e-2,
        strategy: str = 'First-K',
        iterations: int = 30,
        device: str = 'cpu',
    ) -> None:
        assert strategy in ['First-K', 'Top-K'], (
            f'The passed strategy {strategy} is not supported.'
        )

        self.model: str = model
        self.dim: int = dim
        self.antennas_transmitter: int = antennas_transmitter
        self.channel_usage: int = channel_usage
        self.px_cost: float = px_cost
        self.device: str = device
        self.lr: float = lr
        self.strategy: str = strategy
        self.iterations: int = iterations
        self.__F_A: dict[int, torch.Tensor] = {}
        self.__F_B: dict[int, torch.Tensor] = {}
        self.__F_C: dict[int, torch.Tensor] = {}
        self.F_agent: dict[int, torch.Tensor] = {}

        # Attributes Initialization
        self.lmb: float = 0
        self.L: dict[int, torch.Tensor] = {}
        self.mean: dict[int, float] = {}
        self.agents_id: set[int] = set()
        self.agents_pilots: dict[int, dict[str, torch.Tensor | int]] = {}
        self.msgs: defaultdict = defaultdict(int)
        self.channel_matrixes: dict[int, torch.Tensor] = {}

        # Initialize Global F at random and locals
        self.F = torch.kron(
            torch.eye(self.channel_usage, dtype=torch.complex64),
            torch.randn(
                (
                    self.antennas_transmitter,
                    self.antennas_transmitter,
                ),
                dtype=torch.complex64,
            ),
        ).to(self.device)

        return None

    def __save_ABC(
        self,
        msg: dict[str, torch.Tensor],
    ) -> None:
        """Save A, B, C for agent idx.

        Args:
            msg : dict[str, torch.Tensor]
                The message from the agent.

        Returns:
            None
        """
        # Read the message
        idx, msg1 = msg.values()

        # Get the pilots
        # We want to equalize the channel so X -> X
        x = self.agents_pilots[idx]['pilots']

        # Variables
        self.__F_A[idx] = msg1.H @ msg1
        self.__F_B[idx] = torch.linalg.inv(x @ x.H)
        self.__F_C[idx] = (msg1.H @ x @ x.H) @ self.__F_B[idx]

        self.__F_local_step(idx)

        return None

    def __F_local_step(self, idx: int) -> None:
        """The local F step.

        Args:
            idx : int
                The agent idx.

        Returns:
            None
        """
        if self.lmb == 0:
            # Solve for F when lambda equals 0, F = A.inv @ C @ B.inv , but __F_C is C @ B.inv
            self.F_agent[idx] = (
                torch.linalg.inv(self.__F_A[idx]) @ self.__F_C[idx]
            )
        else:
            self.F_agent[idx] = torch.tensor(
                solve_sylvester(
                    self.__F_A[idx].cpu().numpy(),
                    (self.lmb * self.c_repr * self.__F_B[idx]).cpu().numpy(),
                    self.__F_C[idx].cpu().numpy(),
                )
            ).to(self.device)
        return None

    def __F_global_step(self) -> None:
        """The global step.

        Args:
            None

        Returns:
            None
        """

        def lmb_update(lmb: float) -> None:
            """Method needed to update lmb in relation to the constraint violation.

            Args:
                lmb : float
                    The lmb parameter

            Returns:
                lmb : float
                    The updated lmb.
            """
            cnst_viol = self.get_trace() - self.px_cost

            if cnst_viol > 0:
                lmb += self.lr * cnst_viol
                lmb = max(0, lmb)
            else:
                lmb = 0
            return lmb

        self.F = torch.stack(list(self.F_agent.values()), dim=0).mean(dim=0)

        # Check if the constraint is respected
        if self.get_trace() - self.px_cost > 0:
            # Constraint not respected, then we proceed finding the right lambda
            self.c_repr = sum([d['n'] for d in self.agents_pilots.values()])
            for _ in range(self.iterations):
                self.lmb = lmb_update(self.lmb)

                # Perform the local step for each agent
                for agent_id in self.F_agent:
                    self.__F_local_step(agent_id)

                self.F = torch.stack(list(self.F_agent.values()), dim=0).mean(
                    dim=0
                )
        else:
            # Constraint respected, set lambda to zero
            self.lmb = 0

        self.normalizer = math.sqrt(self.get_trace())
        self.F = self.F / self.normalizer
        return None

    def step(self) -> None:
        """The step of the base station.

        Args:
            None

        Return:
            None
        """
        self.__F_global_step()
        return None

    def __compression(
        self, input: torch.Tensor
    ) -> dict[str, torch.Tensor | int]:
        """Compress the input.

        Args:
            input : torch.Tensor
                The input as real d x n.

        Return:
            msg : dict[str, torch.Tensor | int]
                The msg containining the needed information.
        """
        # Get the number of features of the input
        size, n = input.shape

        # Features to transmit
        sent_features = 2 * self.channel_usage * self.antennas_transmitter

        if self.strategy == 'First-K':
            input = input[:sent_features, :]
            indexes = None

        elif self.strategy == 'Top-K':
            # Get the indexes based on the selected strategy
            _, indexes = torch.topk(input.abs(), sent_features, dim=0)

            # Retrieve the values based on the indexes
            input = input[indexes, torch.arange(n)]

        else:
            raise Exception('The passed strategy is not supported.')

        # Complex Compression
        input = complex_compressed_tensor(input, device=self.device)

        return {
            'pilots': input,
            'size': size,
            'sent_features': sent_features,
            'indexes': indexes,
            'strategy': self.strategy,
            'n': n,
        }

    def handshake_step(
        self,
        idx: int,
        pilots: torch.Tensor,
        received: dict[str, torch.Tensor],
        channel_matrix: torch.Tensor,
        c: int = 1,
    ) -> None:
        """Handshaking step simulation.

        Args:
            idx : int
                The id of the agent.
            pilots : torch.Tensor
                The semantic pilots for the agent.
            received : dict[str, torch.Tensor]
                The message from the agent.
            channel_matrix : torch.Tensor
                The channel matrix of the communication.
            c : int
                The handshake cost in terms of messages. Default 1.

        Returns:
            None
        """
        # All on device
        pilots = pilots.T.to(self.device)
        received['msg1'] = received['msg1'].to(self.device)

        assert idx not in self.agents_id, (
            f'Agent of id {idx} already connected to the base station.'
        )
        assert self.dim == pilots.shape[0], (
            "The dimension of the semantic pilots doesn't match the dimension of the base station encodings."
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

        # Compress the pilots & save the pilots
        self.agents_pilots[idx] = self.__compression(pilots)

        # Learn L and the mean for the Prewhitening
        self.agents_pilots[idx]['pilots'], self.L[idx], self.mean[idx] = (
            prewhiten(self.agents_pilots[idx]['pilots'], device=self.device)
        )

        # Compute the local F step
        self.__save_ABC(
            msg={
                'idx': received['idx'],
                'msg1': received['msg1'],
            }
        )

        # Update the number of messages used for handshaking
        self.msgs['handshaking'] += c

        return None

    def group_cast(self):
        pass

    def transmit_to_agent(
        self,
        idx: int,
        msg: torch.Tensor,
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
        # Compress the message
        msg = self.__compression(msg)

        # Prewhitening
        msg['pilots'] = a_inv_times_b(
            self.L[idx], msg['pilots'] - self.mean[idx]
        )

        # Encode the message
        msg['pilots'] = self.F @ msg['pilots']

        # Transmit over the channel
        if self.is_channel_aware(idx):
            msg['pilots'] = self.channel_matrixes[idx] @ msg['pilots']

        # Updating the number of trasmitting messages
        self.msgs['transmitting'] += 1

        # Normalize trace
        msg['normalizer'] = self.normalizer

        return msg


class AgentBaseline(Agent):
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
        bs_pilots: torch.Tensor,
        model_name: str,
        antennas_receiver: int,
        channel_matrix: torch.Tensor,
        channel_usage: int = 1,
        snr: float = 20.0,
        device: str = 'cpu',
    ) -> None:
        # Set Variables
        self.id = id
        self.antennas_receiver: int = antennas_receiver
        self.pilots = pilots.T
        self.pilot_dim, self.n_pilots = self.pilots.shape
        self.bs_pilots: torch.Tensor = bs_pilots
        self.channel_matrix: torch.Tensor = torch.kron(
            torch.eye(channel_usage, dtype=torch.complex64), channel_matrix
        ).to(device)
        self.channel_usage: int = channel_usage
        self.model_name: str = model_name
        self.snr: float = snr
        self.device: str = device

        # Get sigma
        if self.snr:
            self.sigma = (
                sigma_given_snr(
                    self.snr, torch.ones(1) / math.sqrt(self.antennas_receiver)
                )
                / self.channel_usage
            )
        else:
            self.sigma = 0

        # Prepare the A matrix for the alignment
        self.A: torch.Tensor = None
        self.__alignment_step(input=bs_pilots, output=pilots)

        # Compute the G step.
        self.__G_step()

        assert self.G.shape == self.channel_matrix.shape, (
            'Channel and G matrix are not of the same dimension.'
        )
        return None

    def __decompression(
        self,
        msg: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        """Decompression of the received message.

        Args:
            msg : dict[str, torch.Tensor | int]
                The received message

        Return:
            output : torch.Tensor
                The output.
        """
        received, size, sent_features, indexes, strategy, n, _ = msg.values()

        # Decompress the transmitted signal
        received = decompress_complex_tensor(received)

        output = torch.zeros(size, n)

        if strategy == 'First-K':
            output[:sent_features, :] = received

        elif strategy == 'Top-K':
            output[indexes, torch.arange(n)] = received

        else:
            raise Exception('The passed strategy is not supported.')

        return output

    def __G_step(
        self,
    ) -> None:
        """The local G step of the agent.

        Args:
            None

        Returns:
            None
        """
        G, F = mmse_svd_equalizer(self.channel_matrix, self.snr)
        self.G = G.to(self.device)
        return None

    def __alignment_step(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """The alignment step to align the semantic pilots.

        Args:
            input : torch.Tensor
                The input tensor which we want to align.
            output : torch.Tensor
                The output tensor which is the tensor we want to align to.

        Returns:
            None
        """
        self.A = torch.linalg.lstsq(
            input,
            output,
        ).solution.T
        return None

    def step(
        self,
        channel_awareness: bool,
    ) -> dict[str, torch.Tensor]:
        """Return the msg composed by the pilots plus G or GH.

        Args:
            channel_awareness : bool
                If the baseline is channel aware.

        Returns:
            msg : dict[str, torch.Tensor]
                The actual message to send to the base station.
        """
        return {
            'idx': self.id,
            'msg1': self.G @ self.channel_matrix,
        }

    def decode(
        self,
        msg: dict[str, torch.Tensor | int],
        channel_awareness: bool,
    ) -> torch.Tensor:
        """Decode the incoming message from the base station.

        Args:
            msg: dict[str, torch.Tensor | int],
                The incoming message from the base station.
            channel_awareness : bool
                The awareness of the base station about the channel state.

        Returns:
            msg : torch.Tensor
                The decoded message.
        """
        msg['pilots'] = msg['pilots'].to(self.device)

        # Pass through the channel if base station was not aware
        if not channel_awareness:
            msg['pilots'] = self.channel_matrix @ msg['pilots']

        # Additive White Gaussian Noise
        if self.snr:
            w = awgn(
                sigma=self.sigma,
                size=msg['pilots'].shape,
                device=self.device,
            )
            msg['pilots'] += w

        # Decode the message
        msg['pilots'] = (self.G * msg['normalizer']) @ msg['pilots']

        # Clean the message
        msg = self.__decompression(msg)

        # Align
        msg = self.A @ msg

        return msg.T


# ============================================================
#
#                         MAIN LOOP
#
# ============================================================


def main() -> None:
    """Some sanity tests..."""
    print('Start performing sanity tests...')
    print()

    # Variables definition
    n: int = 100
    lr: float = 1e-1
    rho: float = 1e-1
    snr: float = 20.0
    iterations: int = 100
    px_cost: float = 1.0
    tx_dim: int = 384
    rx_dim: int = 768
    channel_usage: int = 8
    strategy: str = 'Top-K'
    channel_aware: bool = True
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    status: str = 'multi-link'
    channel_matrix: torch.Tensor = complex_gaussian_matrix(
        mean=0, std=1, size=(antennas_receiver, antennas_transmitter)
    )
    privacy: bool = True
    device: str = 'cpu'

    # Get the semantic pilots
    tx_pilots: torch.Tensor = torch.randn(n, tx_dim)
    rx_pilots: torch.Tensor = torch.randn(n, rx_dim)

    print('First test...', end='\t')
    # Agents Initialization
    agents: dict[int, Agent] = {
        0: Agent(
            id=0,
            pilots=rx_pilots,
            model_name='an incredible name',
            antennas_receiver=antennas_receiver,
            channel_matrix=channel_matrix,
            channel_usage=channel_usage,
            snr=snr,
            privacy=privacy,
            device=device,
        )
    }

    # Base Station Initialization
    base_station: BaseStation = BaseStation(
        model='an incredible model',
        dim=tx_dim,
        antennas_transmitter=antennas_transmitter,
        channel_usage=channel_usage,
        rho=rho,
        px_cost=px_cost,
        status=status,
        device=device,
    )

    # Perform Handshaking
    for agent_id in agents:
        base_station.handshake_step(
            idx=agent_id,
            pilots=tx_pilots,
            channel_matrix=channel_matrix if channel_aware else None,
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
    print('Second test...', end='\t')
    # Agents Initialization
    agents: dict[int, AgentBaseline] = {
        0: AgentBaseline(
            id=0,
            pilots=rx_pilots,
            bs_pilots=tx_pilots,
            model_name='an incredible name',
            antennas_receiver=antennas_receiver,
            channel_matrix=channel_matrix,
            channel_usage=channel_usage,
            snr=snr,
            device=device,
        )
    }

    # Base Station Initialization
    base_station: BaseStationBaseline = BaseStationBaseline(
        model='an incredible model',
        dim=tx_dim,
        antennas_transmitter=antennas_transmitter,
        channel_usage=channel_usage,
        lr=lr,
        px_cost=px_cost,
        strategy=strategy,
        iterations=iterations,
        device=device,
    )

    # Perform Handshaking
    for agent_id in agents:
        a_msg: dict[str, torch.Tensor] = agents[agent_id].step(
            channel_awareness=channel_aware
        )

        base_station.handshake_step(
            idx=agent_id,
            pilots=tx_pilots,
            received=a_msg,
            channel_matrix=channel_matrix if channel_aware else None,
        )

    base_station.step()
    # msg = base_station.transmit_to_agent(agent_id, tx_pilots.T)
    # received = (
    #     agents[agent_id].decode(msg, base_station.is_channel_aware(agent_id)).T
    # )
    # print(agents[agent_id].eval(received.T, agents[agent_id].pilots.T))
    # print(base_station.get_trace())
    # print(base_station.F)
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
