# `from common_import import *` is importing all the functions, classes, and variables defined in the
# `common_import` module into the current module or script. This allows you to use those imported
# items directly without prefixing them with the module name. It's a way to bring in functionality
# from another module to use in the current code.
from common_import import *

class Agent(LightningDataModule):
    """ Class defining the user in the comm. system;
        Latent spaces and labels associated to this class are called
        train/test/val_users and contains latent spaces and labels;
        
    """
    def __init__(self,
                 language_name:str,
                 dataset:str= "cifar10",
                 device:str="cpu",
                 antennas_receiver:int = 192
                 ):
        
        super().__init__()
        self.dataset = dataset
        self.language_name = language_name
        self.device=device
        self.antennas_receiver = antennas_receiver
        self.load_users_data()
        self.Y = complex_compressed_tensor(self.train_user.latent_space, device=self.device).H
        self.G_k = None
                 
    def load_users_data(self,
                        dataset="cifar10"):
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
        self.train_user = CustomDataset(path=GENERAL_PATH / 'train' / f'{self.language_name}.pt')    
        self.test_user = CustomDataset(path=GENERAL_PATH / 'test' / f'{self.language_name}.pt') 
        self.val_user  = CustomDataset(path=GENERAL_PATH / 'val' / f'{self.language_name}.pt') 
        print(f"Latent spaces of all the agents are correctly loaded!")
        assert self.train_user.latent_space_size[-1] == self.test_user.latent_space_size[-1] and self.train_user.latent_space_size[-1] == self.val_user.latent_space_size[-1], "Input size must match between train, test and val data."
        print(f"The users latent space has dimension {self.train_user.latent_space.shape}")
        return None
    
    def user_message_broadcast(self, channel_matrix):
        H = channel_matrix
        m = (H.H @ self.G_k.H) @ (self.G_k @ H) 
        p = (H.H @ self.G_k.H) @ self.Y
            #print(f"The shape of message for agent {idx} with sem pilot is {p.shape}")
        return m,p
        
if __name__ == "__main__":
     user = Agent(language_name= "vit_small_patch32_224")