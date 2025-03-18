import math
import torch
import wandb
import pandas as pd
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    from datamodules import DataModule
    from utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
        complex_gaussian_matrix,
        awgn,
        decompress_complex_tensor,
    )
else:
    from src.datamodules import DataModule
    from src.utils import (
        complex_compressed_tensor,
        prewhiten,
        sigma_given_snr,
        complex_gaussian_matrix,
        awgn,
        decompress_complex_tensor,
    )

# ============================================================
#
#                    BASELINE DEFINITION
#
# ============================================================
# Da sostituire con Hydra
config_dict = {
    'seed': 42,
    'dataset': 'cifar10',
    'antennas_transmitter': 192,
    'antennas_receiver': 192,
    'base_station_model': 'mobilenetv3_small_075',
    'agents_models': [
        'levit_128s.fb_dist_in1k',
        'mobilenetv3_large_100',
        'vit_base_patch32_clip_224',
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
    ],
    'channel_usage': 1,
}

wandb.init(project='Multi Agent MIMO Semantic Alignment', config=config_dict)


class LinearBaseline:
    def __init__(
        self,
        tx_latents: dict[id : torch.Tensor],
        rx_latents: dict[id : torch.Tensor],
        tx_size: int,
        antennas_transmitter: int,
        antennas_receiver: int,
        strategy: str,
        device: str = 'cpu',
        snr: float = 20.0,
        dual_var: float = None,
        channel_usage: int = 1,
    ):
        """The LinearBaselineOpt class."""
        self.device = torch.device(device)
        self.snr = snr
        self.tx_latents: dict[id : torch.Tensor] = tx_latents
        self.rx_latents: dict[id : torch.Tensor] = rx_latents
        self.tx_dim: int = tx_size
        self.strategy = strategy
        self.antennas_transmitter = antennas_transmitter
        self.antennas_receiver = antennas_receiver  # we are in the square case
        self.L, self.mean = [], []
        self.channel_usage = channel_usage
        self.n = None
        self.power_tx = 1
        self.lmbd = torch.tensor(
            1e-2, dtype=torch.float32
        )  # initial lambda value
        # ---------------------------------
        self.F = torch.randn(  # Nr x Nt
            (self.antennas_transmitter, self.antennas_receiver),
            dtype=torch.complex64,
        ).to(self.device)
        self.G_l: dict[int : torch.Tensor] = {
            idx: torch.zeros_like(self.F) for idx in range(len(rx_latents))
        }
        self.W = {
            idx: None for idx in range(len(rx_latents))
        }  # alignment matrix
        self.alignment_matrix()
        self.prepare_latents()

    def alignment_matrix(self):
        with torch.no_grad():
            for idx in range(len(self.rx_latents)):
                self.W[idx] = torch.linalg.lstsq(
                    self.tx_latents[idx], self.rx_latents[idx]
                ).solution
                self.tx_latents[idx] = self.tx_latents[idx] @ self.W[idx]
        return None

    def prepare_latents(self):
        """Perform complex conversion and pre-withening.
         The tx latents space after complex compression have dimension (K*N-tx x n);
        -Inputs : None, class method
        - Outputs : None
        """
        self.n, self.tx_size = self.tx_latents[0].shape
        if self.strategy == 'FK':
            self.tx_latents = [
                self.tx_latents[idx][
                    :, : self.channel_usage * 2 * self.antennas_transmitter
                ]
                for idx in range(len(self.tx_latents))
            ]
        if self.strategy == 'TK':
            column_norm = {
                id: tensor.abs().sum(dim=0)
                for id, tensor in self.tx_latents.items()
            }
            # Select top-k indices from the first tensor
            first_key = next(iter(column_norm))  # Get first key
            k = self.channel_usage * 2 * self.antennas_transmitter
            if column_norm[first_key].numel() < k:
                raise ValueError(
                    f'Not enough features in column_norm[{first_key}]: {column_norm[first_key].numel()} available, but {k} requested.'
                )
            _, indices = torch.topk(column_norm[first_key], k)
            # Apply feature selection while keeping the dictionary structure
            self.tx_latents = {
                id: tensor[:, indices]
                for id, tensor in self.tx_latents.items()
            }

        assert self.strategy == 'TK' or self.strategy == 'FK', (
            f'Strategy {self.strategy} is not supported, choose TK or FK'
        )
        # Compress the rx and tx_pilots latent spaces
        self.tx_latents = [
            complex_compressed_tensor(self.tx_latents[i].T, device=self.device)
            for i in range(len(self.rx_latents))
        ]

        self.rx_latents = [
            complex_compressed_tensor(self.rx_latents[i].T, device=self.device)
            for i in range(len(self.rx_latents))
        ]

        self.tx_latents, self.L, self.mean = zip(
            *[
                prewhiten(tensor, device=self.device)
                for tensor in self.tx_latents
            ]
        )

        return None

    def equalization(self, channel_matrixes: dict[int : torch.Tensor] = None):
        for user in range(len(self.rx_latents)):
            self.G_step(idx=user, channel=channel_matrixes)
        self.F_step(channel=channel_matrixes)
        # self.update_lambda()
        return None

    def G_step(self, idx: int, channel: dict[int : torch.Tensor]):
        """ """
        U, S, Vt = torch.linalg.svd(channel[idx])
        S = torch.diag(S).to(torch.complex64)
        B = U @ S

        if self.snr:
            self.G_l[idx] = (
                torch.kron(
                    torch.eye(self.channel_usage, dtype=torch.complex64), B.H
                )
                @ torch.linalg.inv(
                    B @ B.H
                    + (1 / self.snr)
                    * torch.view_as_complex(
                        torch.stack(
                            (torch.eye(B.shape[0]), torch.eye(B.shape[0])),
                            dim=-1,
                        )
                    )
                )
            ) * torch.linalg.norm(Vt.H)
        else:
            self.G_l[idx] = (torch.linalg.inv(S) @ U.H) * torch.linalg.norm(
                Vt.H
            )

        assert self.G_l[idx].shape == self.F.shape, (
            'Dimensions of G_l are not correct'
        )
        return None

    def F_step(self, channel: dict[int, torch.Tensor], iter: int = 100):
        """Updates F using iterative method for λ."""
        c = 1 / (self.n * len(self.rx_latents))
        A, B = 0, 0
        for user in range(len(self.rx_latents)):
            GH = self.G_l[user] @ channel[user]
            A += GH.H @ GH @ self.tx_latents[user] @ self.tx_latents[user].H
            B += self.tx_latents[user] @ self.tx_latents[user].H @ GH.H

        self.F = (
            torch.linalg.inv(
                A
                + (self.lmbd.item() / c)
                * torch.eye(self.antennas_transmitter, dtype=torch.complex64)
            )
            @ B
        )

        for _ in range(iter):
            # 1. Find optimal λ using gradient ascent iterative method
            self.update_lambda()
            # 2. Update F with new λ
            self.F = (
                torch.linalg.inv(
                    A
                    + (self.lmbd.item() / c)
                    * torch.eye(
                        self.antennas_transmitter, dtype=torch.complex64
                    )
                )
                @ B
            )

        print(
            f'Trace: {torch.trace(self.F @ self.F.H).real.item()}, Lambda: {self.lmbd.item()}'
        )
        return None

    def update_lambda(self, lr: float = 1e-2):
        with torch.no_grad():
            constraint_violation = (
                torch.trace(self.F @ self.F.H).real - self.power_tx
            )

            if constraint_violation > 0:
                # Se la potenza è troppo alta ed il vincolo e' violato, aumentiamo lambda
                self.lmbd.data += lr * constraint_violation

            elif constraint_violation < 0:
                # per rispettare il vincolo lambda>=0
                self.lmbd.data = torch.max(
                    torch.tensor(0.0, device=self.device), self.lmbd
                )
        return None

    def evaluate(self, channel: list[torch.Tensor]):
        sigma_ = sigma_given_snr(
            snr=self.snr,
            signal=torch.ones(1) / math.sqrt(self.antennas_transmitter),
        )
        # transmit through the channel symbols that are already withened and compressed in prepare_latents() function;
        z = {
            idx: (channel[idx] @ self.F @ self.tx_latents[idx])
            + awgn(sigma=sigma_, size=(self.antennas_transmitter, self.n))
            for idx in range(len(self.rx_latents))
        }
        # perform semantic decoding and alignment matrix: this is the estimated symbols at user side
        x_hat = {
            idx: (self.G_l[idx] @ z[idx])
            for idx in range(len(self.rx_latents))
        }

        # dewithening step
        y_hat = {
            idx: (self.L[idx] @ x_hat[idx]) + self.mean[idx]
            for idx in range(len(self.rx_latents))
        }
        # pad and decompress symbols to get back to (n x d) original dimensions
        y_pad = {
            idx: torch.cat(
                [
                    tensor,
                    torch.zeros(
                        (
                            self.rx_latents[idx].shape[0] - tensor.shape[0],
                            self.n,
                        ),
                        dtype=torch.complex64,
                    ),  # Complex padding
                ],
                dim=0,
            )
            for idx, tensor in y_hat.items()
        }

        y_hat = {
            idx: decompress_complex_tensor(y_pad[idx])
            for idx in range(len(self.rx_latents))
        }

        y_true = {
            idx: decompress_complex_tensor(self.rx_latents[idx])
            for idx in range(len(self.rx_latents))
        }

        loss = [
            (torch.mean((y_true[idx] - y_hat[idx]) ** 2))
            for idx in range(len(self.rx_latents))
        ]

        return loss, y_hat


config = wandb.config


# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================
def main() -> None:
    """The main loop."""
    seed_everything(config.seed, workers=True)
    tabl = pd.DataFrame(
        columns=['compression_factor', 'Trace', 'lambda_value', 'MSE']
    )
    # Channel Initialization
    channel_matrixes: dict[int : torch.Tensor] = {
        idx: complex_gaussian_matrix(
            0,
            1,
            (
                config.antennas_receiver,
                config.antennas_transmitter,
            ),
        )
        for idx, _ in enumerate(config.agents_models)
    }

    # Datamodules Initialization
    datamodules = {
        idx: DataModule(
            dataset=config.dataset,
            tx_enc=config.base_station_model,
            rx_enc=agent_model,
        )
        for idx, agent_model in enumerate(config.agents_models)
    }
    for datamodule in datamodules.values():
        datamodule.prepare_data()
        datamodule.setup()

    Base = LinearBaseline(
        tx_latents={
            id: datamodule.train_data.z_tx
            for id, datamodule in datamodules.items()
        },
        rx_latents={
            id: datamodule.train_data.z_rx
            for id, datamodule in datamodules.items()
        },
        tx_size=datamodules[0].train_data.z_tx.shape,
        strategy='TK',
        antennas_receiver=config.antennas_receiver,
        antennas_transmitter=config.antennas_transmitter,
    )

    compression_factor = (
        Base.channel_usage * Base.antennas_transmitter / (Base.tx_size / 2)
    )
    # -----------------------------------------------------
    Base.equalization(channel_matrixes=channel_matrixes)

    # mse: list of mse for each user language;
    # y_preds: dictionary of {idx:y_hat} where y_hat should be used to the classification task
    mse, y_preds = Base.evaluate(channel=channel_matrixes)
    wandb.log({'mse_list': mse})
    # -----------------------------------------------------
    tabl = [
        compression_factor,
        torch.trace(Base.F @ Base.F.H).real,
        Base.lmbd.data,
        mse,
    ]
    wandb.log({'table': tabl})
    print(tabl)

    return None


if __name__ == '__main__':
    main()

wandb.finish()
