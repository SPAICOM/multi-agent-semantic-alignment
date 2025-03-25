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
import random
import wandb
from scipy.linalg import solve_sylvester
from scipy.optimize import bisect

#To be removed after trials 
if __name__ == "__main__":
    from datamodules import DataModule
else:
    from src.datamodules import DataModule
    
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
    from utils import (
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
            "seed": 42,
            "dataset": 'cifar10',
            "antennas_transmitter": 4,
            "antennas_receiver": 4,
            "base_station_model": "vit_tiny_patch16_224",#"mobilenetv3_large_100",#"rexnet_100",
            "agents_models": [
                #"vit_tiny_patch16_224",
                "vit_small_patch16_224",
                #"mobilenetv3_large_100",
                "rexnet_100",
                "rexnet_130",
                "rexnet_150",
                "vit_small_patch32_224",
                "vit_base_patch16_224",
                "vit_base_patch32_clip_224",
                 "mobilenetv3_small_100"],
                 #"levit_128s.fb_dist_in1k"],
            "channel_usage": [1],#,2,4,6,16],
            "strategy": "TK"}

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
        channel_usage: int = None
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
        self.lmbd = torch.tensor(1e-1, dtype=torch.float32) #initial lambda value
         #---------------------------------
        self.F = torch.randn( #KNr x KNt
            (self.channel_usage*self.antennas_transmitter, self.channel_usage*self.antennas_receiver),
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
        print("Alignment performed before channel equalization;")
        return None

    def prepare_latents(self):
        """Perform complex conversion and pre-withening.
         The tx latents space after complex compression have dimension (K*N-tx x n);
        -Inputs : None, class method
        - Outputs : None
        """
        self.n, self.tx_size = self.tx_latents[0].shape
        #for i in range(len(self.tx_latents)):
            #print(f"TX shape:{self.tx_latents[i].shape}")

        if self.strategy == 'FK':
            self.tx_latents = [
                self.tx_latents[idx][
                    :, : 2 * self.antennas_transmitter * self.channel_usage
                ]
                for idx in range(len(self.tx_latents))
            ]
        #IL PROBLEMA E' CHE DOPO L'ALIGNMENT CI SONO DIVERSE DIMENSIONALITA', 
        #DEVI FARE UNA SELZIONE DI TOP-K PER OGNI SPAZIO LATENETE 
        #E RICONTROLLARE NEL CODICE CHE NON PRENDI IL PRIMO SPAZIO PER FARE TUTTO (MA FARE TUTTO A CICLO!!!!!!!!!)
        if self.strategy == 'TK':
            column_norm = {
                id: tensor.abs().sum(dim=0)
                for id, tensor in self.tx_latents.items()
            }
            # Select top-k indices from the first tensor
            first_key = next(iter(column_norm))  # Get first key
            k =  2 * self.antennas_transmitter * self.channel_usage
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
        # Once that the features are selected, let's 
        # prepare X for the channel usage --> x \in (KN_tx,n): 
        #self.tx_latents = [ self.tx_latents[i].repeat(self.channel_usage,1)
        #                    for i in range(len(self.tx_latents))]
        #self.L = [self.L[i].repeat(self.channel_usage, self.channel_usage)
        #          for i in range(len(self.L))]
        #self.mean = [self.mean[i].repeat(self.channel_usage,1)
        #             for i in range(len(self.mean))]
        return None

    def equalization(self,
                     channel_matrixes :dict[int:torch.Tensor]=None,
                     iterations:int = 10,
                     tol:float = 1e-1):
        kron_channels = {user: torch.kron(torch.eye(self.channel_usage, dtype=torch.complex64), channel_matrixes[user])
                         for user in range(len(self.rx_latents))}
        
        for user in range(len(self.rx_latents)):
            self.G_step(idx=user, channel= kron_channels)
        self.F_step(channel=kron_channels)
        for _ in range(iterations):
           trace_value = self.update_lambda()
           self.F_step(channel=kron_channels) 
        #print(trace_value)
        #Scaling by Trace to ensure power at the TX 
        self.F = self.F/math.sqrt(trace_value)
        self.G_l = {idx: tensor * (math.sqrt(trace_value)) for idx, tensor in self.G_l.items()}
        print(f"Trace value after scaling is:{torch.trace(self.F @ self.F.H).real.item()}")
        return None

    def G_step(self, idx: int, channel: dict[int : torch.Tensor]):
        """ """
        U, S, Vt = torch.linalg.svd(channel[idx])
        S = torch.diag(S).to(torch.complex64)
        B = U @ S

        if self.snr:
            self.G_l[idx] = ( B.H  @ torch.linalg.inv(B@B.H + (1/self.snr) * torch.view_as_complex(torch.stack((torch.eye(B.shape[0]), torch.eye(B.shape[0])), dim=-1))) ) * torch.linalg.norm(Vt.H)
        else:
            self.G_l[idx] = ( torch.linalg.inv(S) @ U.H ) * torch.linalg.norm(Vt.H)
        
        assert self.G_l[idx].shape == self.F.shape, f"Dimensions of G_l are not correct"
        return  None
    
    def F_step(self, channel: dict[int, torch.Tensor]):
        """Updates F using iterative method for λ."""
        c = 1 / (self.n * len(self.rx_latents) * self.channel_usage)
        
        A, B, C = 0, 0, 0
        for user in range(len(self.rx_latents)):
            GH = self.G_l[user] @ channel[user]
            A += GH.H @ GH 
            B += self.tx_latents[user] @ self.tx_latents[user].H 
            C += GH.H @ (self.tx_latents[user] @ self.tx_latents[user].H) 

        w = self.lmbd / c
        self.F = torch.tensor(
            solve_sylvester(
                A.numpy(),
                (w * torch.linalg.inv(B)).numpy(),
                (torch.linalg.inv(B) @ C).numpy()
            )
        ).to(self.device)


        return None

    def update_lambda(self, lr: float = 1e2):
        with torch.no_grad():
            trace_value = torch.trace(self.F @ self.F.H).real.item()
            constraint_violation = trace_value - self.power_tx
            #print(f"Value of the constraint: {constraint_violation}")
            if constraint_violation > 0:  
                # Increase lambda if the constraint is violated (power is too high)
                self.lmbd += lr * constraint_violation 
            elif constraint_violation < 0:
                # Ensure lambda stays non-negative
                self.lmbd = (torch.tensor(0.0)) #torch.max(torch.tensor(0.0, device=self.device), self.lmbd)
            #print(f"Lambda value: {self.lmbd}") 
        return trace_value
  
    def evaluate(self, 
                 channel: dict[int, torch.Tensor]):
        kron_channels = {user: torch.kron(torch.eye(self.channel_usage, dtype=torch.complex64), channel[user])
                         for user in range(len(self.rx_latents))}
        sigma_ = sigma_given_snr(snr=self.snr, signal=torch.ones(1)/math.sqrt(self.antennas_transmitter)) 
        # transmit through the channel symbols that are already withened and compressed in prepare_latents() function;
        z = {
            idx: (kron_channels[idx] @ self.F @ self.tx_latents[idx])
            + awgn(sigma=sigma_, size=(self.channel_usage* self.antennas_transmitter, self.n))
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
        
    Base = LinearBaseline(tx_latents=  {id : datamodule.train_data.z_tx 
                                        for id,datamodule in datamodules.items()},

                        rx_latents= {id : datamodule.train_data.z_rx 
                                        for id,datamodule in datamodules.items()},
                        tx_size =  datamodules[0].train_data.z_tx.shape,
                        strategy= config.strategy,
                        antennas_receiver= config.antennas_receiver,
                        antennas_transmitter=config.antennas_transmitter,
                        channel_usage = config.channel_usage)
    
    compression_factor = (Base.channel_usage*Base.antennas_transmitter/ (Base.tx_size/2) )
    print(f"Compression Factor in this run is: {compression_factor}")
    #-----------------------------------------------------
    Base.equalization(channel_matrixes= channel_matrixes)

    # mse: list of mse for each user language;
    # y_preds: dictionary of {idx:y_hat} where y_hat should be used to the classification task
    mse, y_preds = Base.evaluate(channel = channel_matrixes) 
    #wandb.log({"mse_list": mse})
    #-----------------------------------------------------
    #tabl = [compression_factor, torch.trace(Base.F@Base.F.H).real, Base.lmbd.data]
    losses = pd.DataFrame({'Usage':config.channel_usage,'Model': config.agents_models, 'MSE': mse})

    print(losses)
    #wandb.log({"table": losses})
    #trace = torch.trace(Base.F @ Base.F.H).real.item() - 1
#
    #def f(lam, trace_val):
    #    return lam * (trace_val)
#
    #lambdas = np.logspace(-10, 1, num=200)
    #f_values = f(lambdas, trace)
#
    #lambda_vals = np.logspace(-6, 6, num=200)
    #trace_vals = []
#
    ## Loop over the lambda values and compute trace values
    #for lam in lambda_vals:
    #    Base.lmbd = lam  # update the lambda in your instance
    #    Base.F_step(channel_matrixes)  # update F based on the current lambda
    #    current_trace = torch.trace(Base.F @ Base.F.H).item()  # compute trace
    #    trace_vals.append(current_trace)
#
    ## Create a figure with 1 row and 2 columns of subplots
    #fig, axes = plt.subplots(1, 2, figsize=(16, 5))
#
    ## First subplot: Plot f(lambda) vs lambda
    #axes[0].semilogx(lambdas, f_values, label=r'$f(\lambda) = \lambda\,(\mathrm{trace}(F F^H)-1)$')
    #axes[0].set_xlabel(r'$\lambda$', fontsize=12)
    #axes[0].set_ylabel('f(lambda)', fontsize=12)
    #axes[0].set_title('Plot of $f(\\lambda)$ vs $\\lambda$', fontsize=14)
    #axes[0].legend()
    #axes[0].grid(True, which="both", ls="--", linewidth=0.5)
#
    ## Second subplot: Plot trace(F @ F.H) vs lambda
    #axes[1].semilogx(lambda_vals, trace_vals, label=r'$\operatorname{trace}(F F^H)$')
    #axes[1].set_xlabel(r'$\lambda$', fontsize=12)
    #axes[1].set_ylabel(r'$\operatorname{trace}(F F^H)$', fontsize=12)
    #axes[1].set_title('Trace of $F F^H$ vs. $\lambda$', fontsize=14)
    #axes[1].legend()
    #axes[1].grid(True, which="both", ls="--", linewidth=0.5)

    #plt.tight_layout()
    #wandb.log({"plot": wandb.Image(fig)})
    #plt.show()
    return None


def main_prova() -> None:
    """The main loop."""
    seed_everything(config.seed, workers=True)
    
    # Channel Initialization
    channel_matrixes: dict[int, torch.Tensor] = {
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
    
    # List to collect results for each channel usage value.
    results = []
    
    # Iterate over each channel usage value.
    for usage in config.channel_usage:
        # Create a LinearBaseline instance with the current channel usage.
        Base = LinearBaseline(
            tx_latents={id: datamodule.train_data.z_tx 
                        for id, datamodule in datamodules.items()},
            rx_latents={id: datamodule.train_data.z_rx 
                        for id, datamodule in datamodules.items()},
            tx_size=datamodules[0].train_data.z_tx.shape,
            strategy=config.strategy,
            antennas_receiver=config.antennas_receiver,
            antennas_transmitter=config.antennas_transmitter,
            channel_usage=usage  # Use the current channel usage value.
        )
        
        # Calculate compression factor.
        compression_factor = (usage  / (Base.tx_size)/2)
#        print(f"Compression Factor in this run with channel usage {usage} is: {compression_factor}")
#        
#        # Perform equalization and evaluate.
#        Base.equalization(channel_matrixes=channel_matrixes)
#        mse, y_preds = Base.evaluate(channel=channel_matrixes)
#        
#        mse_dict = {model: mse_value.item() if hasattr(mse_value, "item") else mse_value
#                    for model, mse_value in zip(config.agents_models, mse)}
#        mse_dict['Channel Usage'] = usage  # Add the current channel usage.
#        results.append(mse_dict)
#    
#    # Create a DataFrame with all results.
#    losses = pd.DataFrame(results)
#    print(losses)
#    plt.figure(figsize=(10,6))
#    for model in config.agents_models:
#        plt.plot(
#        losses['Channel Usage'], 
#        losses[model], 
#        marker='o', 
#        label=model
#    )
#
#    plt.xlabel("Channel Usage")
#    plt.ylabel("MSE")
#    plt.title("MSE vs Channel Usage per Model")
#    plt.legend(title="Latent spaces logic")
#    plt.grid(True)
#    plt.show()

if __name__ == '__main__':
    main_prova()

#wandb.finish()


