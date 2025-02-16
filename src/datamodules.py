"""In this python module we define class that handles the dataset:
    - CustomDataset: a custom Pytorch Dataset for encoding from an absolute representation to a relative one.
    - DatasetClassifier: a custom Pytorch Dataset for classifing images.
    - DataModule: a Pytorch Lightning Data Module for the Relative Encoder.
    - DataModuleClassifier: a Pytorch Lightning Data Module for classifing the images.
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


# =====================================================
#
#                 DATASETS DEFINITION
#
# =====================================================

class CustomDataset(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        encoder_path : Path
            The path to the encoder.
        decider_path : Path
            The path to the decoder.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.z : torch.Tensor
            The absolute representation of the Dataset encoder side.
        self.labels : torch.Tensor
            The labels of the Dataset.
        self.z_decoder : torch.Tensor
            The absolute representation of the Dataset decoder side.
        self.input_size : int
            The size of the input of the network.
        self.output_size : int
            The size of the output of the network.
    """
    def __init__(self,
                 encoder_path: Path,
                 decoder_path: Path):
        self.encoder_path: Path = encoder_path
        self.decoder_path: Path = decoder_path
        
        # =================================================
        #                 Encoder Stuff
        # =================================================
        encoder_blob = torch.load(self.encoder_path, weights_only=True)

        # Retrieve the absolute representation from the encoder
        self.z = encoder_blob['absolute']

        # Retrieve the labels
        self.labels = encoder_blob['labels']

        del encoder_blob

        # =================================================
        #                 Decoder Stuff
        # =================================================
        decoder_blob = torch.load(self.decoder_path, weights_only=True)

        # Retrieve the absolute representation from the decoder
        self.z_decoder = decoder_blob['absolute']

        del decoder_blob
        
        # =================================================
        #         Get the input and the output size
        # =================================================
        # When the input is only the absolute representation
        self.input_size = self.z.shape[-1]
        self.output_size = self.z_decoder.shape[-1]


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int
                Length of the Dataset.
        """
        return len(self.z)


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx : int
                The index of the wanted row.

        Returns:
            (input, r_i) : tuple[torch.Tensor, torch.Tensor]
                The inputs and target as a tuple of tensors.
        """
        # Get the absolute representation of element idx
        input = self.z[idx]

        # Get the absolute representation of element idx
        output = self.z_decoder[idx]

        return input, output


class DatasetClassifier(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        path : Path
            The path to the data.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.input: torch.Tensor
            The absolute representation the decoder.
        self.labels : torch.Tensor
            The labels of the Dataset.
        self.input_size : int
            The size of the input of the network.
        self.num_classes : int
            The size of the number of classes.
    """
    def __init__(self,
                 path: Path):
        self.path: Path = path
                
        # =================================================
        #                 Get the Data
        # =================================================
        decoder_blob = torch.load(self.path, weights_only=True)

        # Retrieve the absolute representation from the decoder
        self.input = decoder_blob['absolute']
        
        # Retrieve the labels
        self.labels = decoder_blob['labels']

        del decoder_blob

        # =================================================
        #         Get the input and the output size
        # =================================================
        self.input_size = self.input.shape[-1]
        self.num_classes = self.labels.unique().shape[-1]


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int
            Length of the Dataset.
        """
        return len(self.input)


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx : int
                The index of the wanted row.

        Returns:
            (input_i, l_i) : tuple[torch.Tensor, torch.Tensor]
                The inputs and target as a tuple of tensors.
        """
        # Get the i input
        input_i = self.input[idx]

        # Get the label of the element at idx
        l_i = self.labels[idx]
        
        return input_i, l_i


# =====================================================
#
#                DATAMODULES DEFINITION
#
# =====================================================
    
class DataModule(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset : str
            The name of the dataset.
        encoder : str
            The name of the encoder.
        decoder : str
            The name of the decoder.
        batch_size : int
            The size of a batch. Default 128.
        num_workers : int
            The number of workers. Setting it to 0 means that the data will be
            loaded in the main process. Default 0.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """
    def __init__(self,
                 dataset: str,
                 encoder: str,
                 decoder: str,
                 batch_size: int = 128,
                 num_workers: int = 0) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.encoder: str = encoder
        self.decoder: str = decoder
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers


    def prepare_data(self) -> None:
        """This function prepares the dataset (Download and Unzip).

        Returns:
            None
        """
        from gdown import download
        from zipfile import ZipFile
        from dotenv import dotenv_values

        CURRENT = Path('.')
        DATA_DIR = CURRENT / 'data'
        ZIP_PATH = DATA_DIR / 'latents.zip'
        DIR_PATH = DATA_DIR / 'latents/'

        # Make sure that DATA_DIR exists
        DATA_DIR.mkdir(exist_ok=True)

        # Check if the zip file is already in the path
        if not ZIP_PATH.exists():
            # Get from the .env file the zip file Google Drive ID
            ID = dotenv_values()['DATA_ID']

            # Download the zip file
            download(id=ID, output=str(ZIP_PATH))
            
        # Check if the directory exists
        if not DIR_PATH.is_dir():
            # Unzip the zip file
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(ZIP_PATH.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a Dataset for our data.

        Args:
            stage : str
                The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        self.train_data = CustomDataset(encoder_path=GENERAL_PATH / 'train' / f'{self.encoder}.pt',
                                        decoder_path=GENERAL_PATH / 'train' / f'{self.decoder}.pt')
        self.test_data = CustomDataset(encoder_path=GENERAL_PATH / 'test' / f'{self.encoder}.pt',
                                       decoder_path=GENERAL_PATH / 'test' / f'{self.decoder}.pt')
        self.val_data = CustomDataset(encoder_path=GENERAL_PATH / 'val' / f'{self.encoder}.pt',
                                      decoder_path=GENERAL_PATH / 'val' / f'{self.decoder}.pt')

        assert self.train_data.input_size == self.test_data.input_size and self.train_data.input_size == self.val_data.input_size, "Input size must match between train, test and val data."
        assert self.train_data.output_size == self.test_data.output_size and self.train_data.output_size == self.val_data.output_size, "Output size must match between train, test and val data."

        self.input_size = self.train_data.input_size
        self.output_size = self.train_data.output_size

        return None


    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def predict_dataloader(self) -> DataLoader:
        """The function returns the predict DataLoader.

        Returns:
            DataLoader
                The predict DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class DataModuleClassifier(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset : str
            The name of the dataset.
        decoder : str
            The name of the decoder.
        batch_size : int
            The size of a batch. Default 128.
        num_workers : int
            The number of workers. Setting it to 0 means that the data will be
            loaded in the main process. Default 0.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """
    def __init__(self,
                 dataset: str,
                 decoder: str,
                 batch_size: int = 128,
                 num_workers: int = 0) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.decoder: str = decoder
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers


    def prepare_data(self) -> None:
        """This function prepare the dataset (Download and Unzip).

        Returns:
            None
        """
        from gdown import download
        from zipfile import ZipFile
        from dotenv import dotenv_values

        CURRENT = Path('.')
        DATA_DIR = CURRENT / 'data'
        ZIP_PATH = DATA_DIR / 'latents.zip'
        DIR_PATH = DATA_DIR / 'latents/'

        # Make sure that DATA_DIR exists
        DATA_DIR.mkdir(exist_ok=True)

        # Check if the zip file is already in the path
        if not ZIP_PATH.exists():
            # Get from the .env file the zip file Google Drive ID
            ID = dotenv_values()['DATA_ID']

            # Download the zip file
            download(id=ID, output=str(ZIP_PATH))
            
        # Check if the directory exists
        if not DIR_PATH.is_dir():
            # Unzip the zip file
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(ZIP_PATH.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a DatasetRelativeDecoder for our data.

        Args:
            stage : str
                The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        self.train_data = DatasetClassifier(path=GENERAL_PATH / 'train' / f'{self.decoder}.pt')
        self.test_data = DatasetClassifier(path=GENERAL_PATH / 'test' / f'{self.decoder}.pt')
        self.val_data = DatasetClassifier(path=GENERAL_PATH / 'val' / f'{self.decoder}.pt')

        assert self.train_data.input_size == self.test_data.input_size and self.train_data.input_size == self.val_data.input_size, "Input size must match between train, test and val data."
        assert self.train_data.num_classes == self.test_data.num_classes and self.train_data.num_classes == self.val_data.num_classes, "The number of classes must match between train, test and val data."

        self.input_size = self.train_data.input_size
        self.num_classes = self.train_data.num_classes
        return None


    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def predict_dataloader(self) -> DataLoader:
        """The function returns the predict DataLoader.

        Returns:
            DataLoader
                The predict DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



def main() -> None:
    """The main script loop in which we perform some sanity tests.
    """
    print("Start performing sanity tests...")
    print()
    
    # Setting inputs
    dataset = 'cifar10'
    encoder = 'vit_small_patch16_224'
    decoder = 'vit_base_patch16_224'
    
    print("Running first test...", end='\t')
    data = DataModule(dataset=dataset,
                      encoder=encoder,
                      decoder=decoder)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    print("Running second test...", end='\t')
    data = DataModuleClassifier(dataset=dataset,
                                decoder=decoder)
    
    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    return None


if __name__ == "__main__":
    main()
