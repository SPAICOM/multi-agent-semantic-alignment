import math
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.linalg import solve_sylvester
import sys
import os
# Aggiungi il percorso della directory radice 
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Risale di un livello
sys.path.insert(0, root_dir)
from src.datamodules import CustomDataset
from src.utils import complex_compressed_tensor, decompress_complex_tensor, prewhiten, sigma_given_snr, awgn, a_inv_times_b
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from gdown import download
from zipfile import ZipFile
from dotenv import dotenv_values
from src.utils import complex_gaussian_matrix
from src.users import Agents
from src.base_station import Base_Station

class Linear_Protocol(Base_Station, Agents):
    def __init__(self, 
                 dataset:str, 
                 model_name:list, 
                 iterations:int = 10, 
                 snr :float =20, 
                 rho:int = 1e2,
                 device:str="cpu",
                 cost:float = 1.0,
                 **args):
        self.snr=snr
        self.cost= cost
        self.iter = iterations
        self.rho=rho 
        # Initialize Base_Station and agents 
        Base_Station.__init__(self, **args)
        Agents.__init__(self, 
                        dataset=dataset, 
                        model_name=model_name,
                        **args)
        self.n = self.train_bs.latent_space.shape[0]
        self.channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(self.antennas_receiver, self.antennas_transmitter))
        
    def linear_optimizer(self,iterations:int):
        
        
        self.X, self.L, self.mean_X = prewhiten(self.X, device=self.device)
        #self.X = a_inv_times_b(self.L, self.X - self.mean_X)
        self.F = torch.view_as_complex(torch.stack((torch.randn(self.antennas_transmitter, (self.train_bs.latent_space_size[-1] + 1) // 2), torch.randn(self.antennas_transmitter, (self.train_bs.latent_space_size[-1] + 1) // 2)), dim=-1)).to(self.device)
        
        with torch.no_grad():
                
            for i in tqdm(range(iterations)):
                #################################
                #          G_k Step             #
                #################################
                HFX = self.AP_message_broadcast(channel_matrix = self.channel_matrix)
                #print(f"HFX at iteration {i} is: {HFX}")
                sigma = sigma_given_snr(self.snr, torch.ones(1)/math.sqrt(self.antennas_transmitter))
                #G_k = Y(HFX)((HFX)(HFX)+nKΣ)^-1
                self.G_k = {idx: (self.Y[idx] @ HFX.H @ torch.linalg.inv(HFX @ HFX.H + self.n  * sigma * torch.view_as_complex(torch.stack((torch.eye(HFX.shape[0]), torch.eye(HFX.shape[0])), dim=-1)).to(self.device))).to(self.device)
                            for idx in range(self.num_users)}
                #Dimensions check 
                assert self.G_k[0].shape == ((self.train_users[0].latent_space.shape[-1] + 1) // 2, self.antennas_receiver), f"Expected G_k shape has dimension issues"
                
                #################################
                #    Fk Step & Aggregation      #
                #################################
                self.bs_buffer:list = self.user_message_broadcast(channel_matrix = self.channel_matrix) #a list that contain for each idx a list of A and (GH)^H(Y)
                rho_n = self.rho*self.n
                for user in range(self.num_users):
                    A = self.bs_buffer[user][0]
                    B = rho_n * torch.linalg.inv(self.X @ self.X.H)
                    C = (rho_n * (self.Z - self.U)) + (self.bs_buffer[user][1] @ self.X.H) @ (B/(rho_n)) #quest'ultimo termine va controllato 
                    self.F_k[user] = torch.tensor(solve_sylvester(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()), device=self.device)
                    print(np.linalg.norm(self.F_k[user]))
                    #residual = A.numpy().dot(self.F.numpy()) + self.F.numpy().dot(B.numpy()) - C.numpy()
                    #print(np.linalg.norm(residual))
                    
                self._F_aggregation()
                
                #################################
                #     Dual variables update     #
                #################################
                C = self.F + self.U
                tr = torch.trace(C @ C.H).real
                
                if tr <= self.cost:
                    self.Z = C
                else:
                    lmb = torch.sqrt(tr / self.cost).item() -1
                    self.Z = C / (1 + lmb)
                    
                self.U = self.U + self.F - self.Z
                #self.residuals = torch.linalg.matrix_norm(self.F - self.Z)
                #print(f"Residuals at iter {i} is: {self.residuals}")
        return None
    
    
    

if __name__ == "__main__":
    lp = Linear_Protocol(dataset="cifar10", 
                        model_name=["vit_small_patch16_224", "vit_small_patch32_224"])
    lp.linear_optimizer(iterations=10)
    
    
    