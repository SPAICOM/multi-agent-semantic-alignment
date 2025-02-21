from pytorch_lightning import LightningDataModule
import torch
import math
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
from gdown import download
from zipfile import ZipFile
from dotenv import dotenv_values
from src.base_station import Base_Station

class Agents(LightningDataModule):
    """ Class for agents of the system communicating with Base Station;
        Latent spaces and labels associated to this class are called
        train/test/val_users and contains latent spaces and labels;
        
    """
    def __init__(self,
                 dataset:str,
                 model_name:list,
                 device:str="cpu",
                 antennas_receiver:int=192):
        self.device=device
        self.languages= {user:model for user,model in zip(range(len(model_name)),model_name)}
        self.num_users = len(self.languages.keys())
        self.antennas_receiver = antennas_receiver
        self.train_data = {user:None for user in range(self.num_users)}
        self.load_users_data()
        self.Y = {k : complex_compressed_tensor(self.train_users[k].latent_space, device=self.device).H
                  for k in range(self.num_users)}
        
        
        self.buffer_users = {k:[] for k in range(self.num_users)}
        self.H = None #torch.view_as_complex(
                      #torch.stack((torch.randn(self.antennas_receiver, self.antennas_receiver), 
                              #torch.randn(self.antennas_receiver, self.antennas_receiver)), dim=-1)).to(self.device)
                 
        #Initialize G variable of SAE at receiver side 
        # G_k:[Torch.tensor]-->(N_RX x m/2)
        self.G_k = None
        # self.G_k = {
        #             user: torch.view_as_complex(
        #                 torch.stack([
        #                     torch.zeros((self.train_users[0].latent_space.shape[-1] + 1) // 2, self.antennas_receiver),
        #                     torch.zeros((self.train_users[0].latent_space.shape[-1] + 1) // 2, self.antennas_receiver)
        #                 ], dim=-1)
        #             ).to(self.device)
        #             for user in range(self.num_users)
        #         }

    def load_users_data(self,
                        dataset="cifar10"):
       
        user_models= list(self.languages.values())
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
        #Extract train/test/val latent spaces for each users
        #and store it in a dict of the form {user_idx : latent_space_data}
        self.train_users = {k: CustomDataset(path=GENERAL_PATH / 'train' / f'{lang_name}.pt') 
                           for k,lang_name in zip(range(self.num_users),self.languages.values())}
        self.test_users = {k: CustomDataset(path=GENERAL_PATH / 'test' / f'{lang_name}.pt') 
                           for k,lang_name in zip(range(self.num_users),self.languages.values())}
        self.val_users = {k: CustomDataset(path=GENERAL_PATH / 'val' / f'{lang_name}.pt') 
                           for k,lang_name in zip(range(self.num_users),self.languages.values())}
        print(f"Latent spaces of all the agents are correctly loaded!")
        assert self.train_users[0].latent_space_size[-1] == self.test_users[0].latent_space_size[-1] and self.train_users[0].latent_space_size[-1] == self.val_users[0].latent_space_size[-1], "Input size must match between train, test and val data."
        print(f"The users latent space has dimension {self.train_users[0].latent_space.shape}")
        return None
    
    def user_message_broadcast(self, channel_matrix, prewithening=False):
        self.H = channel_matrix
        messages=[]
        for idx in range(self.num_users):
            m = (self.H.H @ self.G_k[idx].H) @ (self.G_k[idx] @ self.H) 
            if prewithening:
                self.Y[idx] = prewhiten(self.Y[idx])
            p = (self.H.H @ self.G_k[idx].H) @ self.Y[idx]
            #print(f"The shape of message for agent {idx} with sem pilot is {p.shape}")
            messages.append([m,p])
        return messages
        
 