import math
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.linalg import solve_sylvester

if __name__ == '__main__':
    from utils import (
        complex_gaussian_matrix,
        complex_compressed_tensor,
        decompress_complex_tensor,
        prewhiten,
        sigma_given_snr,
        awgn,
        a_inv_times_b,
    )
else:
    from src.utils import (
        complex_compressed_tensor,
        decompress_complex_tensor,
        prewhiten,
        sigma_given_snr,
        awgn,
        a_inv_times_b,
    )


class FederatedLinearOptimizer:
    """The linear optimizer for a multi agent scenario.

    Args:
        n_agents : int
            The number of agents need for the simulation.
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The Complex Gaussian Matrix simulating the communication channel.
        snr : float
            The snr in dB of the communication channel. Set to None if unaware.
        cost : float
            The transmition cost. Default 1.0.
        rho : int
            The rho coeficient for the admm method. Default 1e2.

    Attributes:
        self.<args_name>
        self.dtype : torch.dtype
            The dtype of the data.
        self.antennas_transmitter : int
            The number of antennas transmitting the signal.
        self.antennas_receiver : int
            The number of antennas receiving the signal.
        self.F : torch.Tensor
            The F matrix.
        self.G : torch.Tensor
            The G matrix.
        self.Z : torch.Tensor
            The Proximal variable for ADMM.
        self.U : torch.Tensor
            The Dual variable for ADMM.
    """

    def __init__(
        self,
        n_agents: int,
        input_dim: int,
        output_dim: int,
        channel_matrix: list[torch.Tensor],
        snr: float,
        cost: float = 1.0,
        rho: float = 1e2,
        device: str = 'cpu',
    ):
        assert len(channel_matrix[0].shape) == 2, (
            'The matrix must be 2 dimesional.'
        )
        self.n_agents = n_agents
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.channel_matrix: list = [t.to(device) for t in channel_matrix]
        self.snr: float = snr
        self.cost: float = cost
        self.rho: float = rho
        self.device: str = device
        self.dtype = channel_matrix[0].dtype
        self.antennas_receiver, self.antennas_transmitter = (
            self.channel_matrix[0].shape
        )

        # Variables
        self.F: torch.Tensor = torch.randn(
            size=(self.antennas_transmitter, (self.input_dim + 1) // 2)
        )
        self.F_k = {k: self.F for k in range(self.n_agents)}
        self.G_k = {k: None for k in range(self.n_agents)}

        # ADMM variables
        self.Z = torch.zeros(
            self.antennas_transmitter, (self.input_dim + 1) // 2
        ).to(self.device)
        self.U = torch.zeros(
            self.antennas_transmitter, (self.input_dim + 1) // 2
        ).to(self.device)

        return None

    def __G_step(
        self,
        idx: int,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """The G step that minimize the Lagrangian.

        Args:
            idx: int
                The user index.
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.

        Returns:
            None
        """
        # Get the number of samples
        _, n = input.shape
        # Get the auxiliary matrix A
        A = self.channel_matrix[idx] @ self.F @ input

        # Get sigma
        sigma = 0
        if self.snr:
            sigma = sigma_given_snr(
                self.snr, torch.ones(1) / math.sqrt(self.antennas_transmitter)
            )

        self.G_k[idx] = (
            output
            @ A.H
            @ torch.linalg.inv(
                A @ A.H
                + n
                * sigma
                * torch.view_as_complex(
                    torch.stack(
                        (torch.eye(A.shape[0]), torch.eye(A.shape[0])), dim=-1
                    )
                ).to(self.device)
            )
        ).to(self.device)
        return None

    def __F_step(
        self,
        idx: int,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """The F step that minimize the Lagrangian.

        Args:
            idx: int
                The user index.
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.

        Returns:
            None
        """
        # Get the number of samples
        _, n = input.shape

        # Get the auxiliary matrixes
        rho = self.rho * n
        O_k = self.G_k[idx] @ self.channel_matrix[idx]
        A = O_k.H @ O_k
        B = rho * torch.linalg.inv(input @ input.H)
        C = (rho * (self.Z - self.U) + O_k.H @ output @ input.H) @ (B / rho)

        self.F_k[idx] = torch.tensor(
            solve_sylvester(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()),
            device=self.device,
            dtype=self.dtype,
        )
        return None

    def __F_aggregate(self):
        """ "
        Perform the mean of F transformation
        """
        stacked_F = torch.stack(list(self.F_k.values()), dim=0)
        self.F = stacked_F.mean(dim=0)

        assert self.F_k[0].shape == self.F.shape, (
            'The F aggregation step has wrong dimensions of output.'
        )
        return None

    def __Z_step(self) -> None:
        """The Z step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        # Get the auxiliary matrix C
        C = self.F + self.U
        tr = torch.trace(C @ C.H).real

        if tr <= self.cost:
            self.Z = C
        else:
            lmb = torch.sqrt(tr / self.cost).item() - 1
            self.Z = C / (1 + lmb)

        return None

    def __U_step(self) -> None:
        """The U step for the Scaled ADMM.

        Args:
            None

        Returns:
            None
        """
        self.U = self.U + self.F - self.Z
        return None

    def fit(
        self,
        n_agents: int,
        datamodule_list: list[torch.Tensor],
        iterations: int = None,
    ) -> tuple[list[float], list[float]]:
        """Fitting the F and G to the passed data.

        Args:
            n_agents : int
                The total number of agents.
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default None.

        Returns:
            (losses, traces) : tuple[list[float], list[float]]
                The losses and the traces during training.
        """
        input = datamodule_list[0].train_data.z  # Transpose
        output = {
            k: datamodule_list[k].train_data.z_decoder
            for k in range(self.n_agents)
        }

        # Inizialize the F matrix at random (this will inplace modify also F_k dictionary)
        self.F = torch.view_as_complex(
            torch.stack(
                (
                    torch.randn(
                        self.antennas_transmitter, (self.input_dim + 1) // 2
                    ),
                    torch.randn(
                        self.antennas_transmitter, (self.input_dim + 1) // 2
                    ),
                ),
                dim=-1,
            )
        ).to(self.device)

        with torch.no_grad():
            #  Save the decompressed version
            old_input = input
            old_output = list(output.values())

            # Complex compression
            input = complex_compressed_tensor(input.T, device=self.device).H
            output = [
                complex_compressed_tensor(o.T, device=self.device).H
                for k, o in output.items()
            ]

            # Perform the prewhitening step
            input, self.L_input, self.mean_input = prewhiten(
                input, device=self.device
            )
            # print(input.shape, self.L_input.shape, self.mean_input.shape)
            loss = np.inf
            losses = []
            traces = []
            if iterations:
                for _ in tqdm(range(iterations)):
                    for agent in range(self.n_agents):
                        print('Initiliazing the G process;')
                        self.__G_step(
                            idx=agent, input=input, output=output[agent]
                        )

                    for agent in range(self.n_agents):
                        print('Initiliazing the F process;')
                        self.__F_step(
                            idx=agent, input=input, output=output[agent]
                        )

                    # Compute base station computation:
                    print('Aggregating F process;')
                    self.__F_aggregate()
                    self.__Z_step()
                    self.__U_step()
                    loss = self.eval(old_input, old_output)
                    trace = torch.trace(self.F.H @ self.F).real.item()
                    losses.append(loss)
                    traces.append(trace)

            else:
                print('No iterations specified!')
            #     while loss > 1e-1:
            #         self.__G_step(input=input, output=output)
            #         self.__F_step(input=input, output=output)
            #         self.__Z_step()
            #         self.__U_step()
            #         loss = self.eval(old_input, old_output)
            #         trace = torch.trace(self.F.H@self.F).real.item()
            #         losses.append(loss)
            #         traces.append(trace)

        return losses, traces

    def transform(
        self,
        input: torch.Tensor,
        agent: int,
    ) -> torch.Tensor:
        """Transform the passed input.

        Args:
            input : torch.Tensor
                The input tensor.

        Returns:
            output : torch.Tensor
                The transformed version of the input.
        """
        with torch.no_grad():
            # Transpose
            input = input.T

            # Complex Compress the input
            input = complex_compressed_tensor(input.T, device=self.device).H

            # Perform the prewhitening step
            input = a_inv_times_b(
                self.L_input, input - self.mean_input
            )  # white signal

            # Transmit the input through the channel H
            z = self.channel_matrix[agent] @ self.F @ input

            # Add the additive white gaussian noise
            if self.snr:
                sigma = sigma_given_snr(
                    snr=self.snr,
                    signal=torch.ones(1)
                    / math.sqrt(self.antennas_transmitter),
                )

                w = awgn(sigma=sigma, size=z.shape, device=self.device)
                z += w

            output_complex = self.G_k[agent] @ z

            # Decompress the transmitted signal
            output_decompressed = decompress_complex_tensor(output_complex.H).T

        return output_decompressed.T

    def eval(
        self,
        input: torch.Tensor,
        output: list[torch.Tensor],
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
        # Check if self.F and self.G are fitted
        # assert (self.F is not None)&(self.G_k is not None), "You have to fit the solver first by calling the '.fit()' method."
        loss = 0.0
        for agent in range(self.n_agents):
            preds_k = self.transform(input, agent=agent)
            loss += torch.nn.functional.mse_loss(
                preds_k, output[agent], reduction='mean'
            ).item()
        L = loss / self.n_agents
        print(f'The average Loss function is: {L}')
        return L


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
    cost: int = 1
    snr: float = 20
    iterations: int = 10
    input_dim: int = 384
    output_dim: int = 768
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    n_agent: int = 3
    list_channel_matrix: list[torch.Tensor] = [
        complex_gaussian_matrix(
            mean=0,
            std=1,
            size=(antennas_receiver, antennas_transmitter),
        )
        for _ in range(n_agent)
    ]

    # Latent Spaces for TX and RX
    input = torch.randn(100, input_dim)
    output = [torch.randn(100, output_dim) for _ in range(n_agent)]

    print('Test for Federated Linear Opt..', end='\t')
    system = FederatedLinearOptimizer(
        n_agents=n_agent,
        input_dim=input_dim,
        output_dim=output_dim,
        channel_matrix=list_channel_matrix,
        snr=snr,
        cost=cost,
        device='cpu',
    )
    system.fit(n_agent, input, output, iterations=iterations)

    # for agent in range(n_agent):
    #    system.transform(input, agent)
    print(system.eval(input, output))
    print('[Passed]')

    print()
    return None


if __name__ == '__main__':
    main()
