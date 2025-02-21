from common_import import *


class Base_Station(LightningDataModule):
    """This class define the Base Station features,
       actions and the message that AP exchange with users.
    """
    def __init__(self, 
                 num_users: int = 2, #momentaneamente assegnato cosi
                 dataset:str = "cifar10",
                 model_name:str = "vit_base_patch16_224",
                 batch_size:int =128,
                 antennas_transmitter:int= 192,
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
    
    def AP_message_broadcast(self, channel_matrix):
        """ Broadcasts the message [HF^(t)X] to all users in the network.
        
        Args: None
        Return: message(Torch.tensor)--> the result of (HFX)^H
        """
        H = channel_matrix
        message = (H @ self.F @ self.X)
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
   bs = Base_Station(dataset="cifar10")
   H = torch.view_as_complex(
    torch.stack(
        (
            torch.randn(bs.antennas_transmitter, bs.antennas_transmitter),
            torch.randn(bs.antennas_transmitter, bs.antennas_transmitter)
        ),
        dim=-1
    )
)
   
   m=bs.AP_message_broadcast(channel_matrix=H) 
   #bs._F_aggregation() 