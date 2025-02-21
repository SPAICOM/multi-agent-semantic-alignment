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


class Base_Station(LightningDataModule):
    """This class define the Base Station features,
       actions and the message that AP exchange with users.
    """
    def __init__(self, 
                 dataset:str = "cifar10",
                 model_name:str = "vit_base_patch16_224",
                 batch_size:int =128,
                 antennas_transmitter:int= 192,
                 num_users:int = 1, #temporary as that
                 device:str = "cpu"
                 ):
        self.device = device
        self.antennas_transmitter = antennas_transmitter
        self.dataset = "cifar10"
        self.language = model_name
        self.batch_size = batch_size
        self.bs_buffer = []
        self.load_data(self.language, self.dataset)
        self.X = complex_compressed_tensor(self.train_bs.latent_space, device=self.device).H
        # Variables needed in the protocol variable exchange
        self.F = torch.view_as_complex(torch.stack((torch.randn(self.antennas_transmitter, (self.train_bs.latent_space_size[-1] + 1) // 2), torch.randn(self.antennas_transmitter, (self.train_bs.latent_space_size[-1] + 1) // 2)), dim=-1)).to(self.device)
        self.F_k = {k:None for k in range(num_users)}
        #self.H = torch.view_as_complex(
                 #torch.stack((torch.randn(self.antennas_transmitter, self.antennas_transmitter), 
                             # torch.randn(self.antennas_transmitter, self.antennas_transmitter)), dim=-1)).to(self.device)
        self.Z = torch.zeros(self.antennas_transmitter,(self.train_bs.latent_space_size[-1] + 1) // 2).to(self.device)
        self.U = torch.zeros(self.antennas_transmitter,(self.train_bs.latent_space_size[-1] + 1) // 2).to(self.device) 
        

    def load_data(self, model_name, dataset="cifar10"):
        """ Simple function to load latent spaces (absolute) from GoogleDrive[GiuseppeID].
            The function is called at AP class initialization;
        Args:
            model_name (str): Name of the model.
            dataset (str, optional): Name of the dataset. Defaults to "cifar10".
        Return: None
        """
        CURRENT = Path('.')
        DATA_DIR = CURRENT / 'data'
        ZIP_PATH = DATA_DIR / 'latents.zip'
        DIR_PATH = DATA_DIR / 'latents/'
        CHECK_DIR = DATA_DIR / 'latents'
        if not CHECK_DIR.exists():
            # Make sure that DATA_DIR exists
            DATA_DIR.mkdir(exist_ok=True) 
            ID = dotenv_values()['DATA_ID']
            print("Downloading zip file...")
            download(id=ID, output=str(ZIP_PATH), quiet=False)
            print("Download complete!")
            
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Latents download complete")
            
        else:
            print("Latents folder already exists. Skipping download and extraction.")
        GENERAL_PATH: Path = CURRENT / 'data/latents'/dataset
        #Extract train/test/val latent spaces of the AP;
        self.train_bs = CustomDataset(path=GENERAL_PATH / 'train' / f'{model_name}.pt')
        self.test_bs = CustomDataset(path=GENERAL_PATH / 'test' / f'{model_name}.pt')
        self.val_bs = CustomDataset(path=GENERAL_PATH / 'val' / f'{model_name}.pt')
        print(f"Latent spaces tr/val/ts are correctly loaded!")
        assert self.train_bs.latent_space_size[-1] == self.test_bs.latent_space_size[-1] and self.train_bs.latent_space_size[-1] == self.val_bs.latent_space_size[-1], "Input size must match between train, test and val data."
        #print(f"The AP latent space has dimension {self.train_data.latent_space_size}")
        return None
    
    def AP_message_broadcast(self, channel_matrix, prewithening=False):
        """ Broadcasts the message [HF^(t)X] to all users in the network.
        
        Args: None
        Return: message(Torch.tensor)--> the result of (HFX)^H
        """
        self.H = channel_matrix
        if prewithening:
           self.X = prewhiten(self.X)
        message = (self.H @ self.F @ self.X)
        return message
    
    def _F_aggregation(self):
        """ Aggregates the F matrices of all users in the network
            and update the global F variable, maintaining it at base station side;
        Args: None
        Return: None
        """
        self.F = torch.mean(torch.stack(list(self.F_k.values()), dim=0),dim=0)
        return None

if __name__ == "__main__":
   bs = Base_Station(dataset="cifar10", model_name="vit_small_patch16_224")
   bs.AP_message_broadcast() 
   bs._F_aggregation() 