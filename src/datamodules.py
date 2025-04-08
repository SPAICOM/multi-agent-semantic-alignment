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


if __name__ == '__main__':
    from download_utils import (
        download_zip_from_gdrive,
    )
else:
    from src.download_utils import (
        download_zip_from_gdrive,
    )

# =====================================================
#
#                 DATASETS DEFINITION
#
# =====================================================


class CustomDataset(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        tx_path : Path
            The path to the tx encoding model.
        rx_path : Path
            The path to the rx encoding model.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.z_tx : torch.Tensor
            The absolute representation of the Dataset transmitter side.
        self.labels : torch.Tensor
            The labels of the Dataset.
        self.z_rx : torch.Tensor
            The absolute representation of the Dataset receiver side.
        self.input_size : int
            The size of the input of the network.
        self.output_size : int
            The size of the output of the network.
    """

    def __init__(
        self,
        tx_path: Path,
        rx_path: Path,
    ):
        self.tx_path: Path = tx_path
        self.rx_path: Path = rx_path

        # =================================================
        #                 Encoder Stuff
        # =================================================
        tx_blob = torch.load(self.tx_path, weights_only=True)

        # Retrieve the absolute representation from the transmitter
        self.z_tx = tx_blob['absolute']

        # Retrieve the labels
        self.labels = tx_blob['labels']

        del tx_blob

        # =================================================
        #                 Decoder Stuff
        # =================================================
        rx_blob = torch.load(self.rx_path, weights_only=True)

        # Retrieve the absolute representation from the receiver
        self.z_rx = rx_blob['absolute']

        del rx_blob

        # =================================================
        #         Get the input and the output size
        # =================================================
        # When the input is only the absolute representation
        self.input_size = self.z_tx.shape[-1]
        self.output_size = self.z_rx.shape[-1]

    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int
                Length of the Dataset.
        """
        return len(self.z_tx)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx : int
                The index of the wanted row.

        Returns:
            (input, output) : tuple[torch.Tensor, torch.Tensor]
                The inputs and target as a tuple of tensors.
        """
        # Get the absolute representation of element idx
        input = self.z_tx[idx]

        # Get the absolute representation of element idx
        output = self.z_rx[idx]

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

    def __init__(
        self,
        path: Path,
    ):
        self.path: Path = path

        # =================================================
        #                 Get the Data
        # =================================================
        rx_blob = torch.load(self.path, weights_only=True)

        # Retrieve the absolute representation from the receiver
        self.input = rx_blob['absolute']

        # Retrieve the labels
        self.labels = rx_blob['labels']

        del rx_blob

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

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        tx_enc : str
            The name of the encoder transmitter side.
        rx_enc : str
            The name of the encoder transmitter side.
        train_subset_ratio : float
            The ratio of dataset to use for training. Default 1.0.
        batch_size : int
            The size of a batch. Default 128.
        num_workers : int
            The number of workers. Setting it to 0 means that the data will be
            loaded in the main process. Default 0.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """

    def __init__(
        self,
        dataset: str,
        tx_enc: str,
        rx_enc: str,
        train_subset_ratio: float = 1.0,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.tx_enc: str = tx_enc
        self.rx_enc: str = rx_enc
        self.train_subset_ratio: float = train_subset_ratio
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    def prepare_data(self) -> None:
        """This function prepares the dataset (Download and Unzip).

        Returns:
            None
        """
        from dotenv import dotenv_values

        # Get from the .env file the zip file Google Drive ID
        ID = dotenv_values()['DATA_ID']

        # Download and unzip the data
        download_zip_from_gdrive(ID, name='latents', path='data')

        return None

    def setup(
        self,
        stage: str = None,
    ) -> None:
        """This function setups a Dataset for our data.

        Args:
            stage : str
                The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        # ================================================================
        #                         Train Data
        # ================================================================
        self.train_data = CustomDataset(
            tx_path=GENERAL_PATH / 'train' / f'{self.tx_enc}.pt',
            rx_path=GENERAL_PATH / 'train' / f'{self.rx_enc}.pt',
        )

        full_size = len(self.train_data)
        subset_size = int(full_size * self.train_subset_ratio)
        indices = torch.randperm(full_size)[:subset_size]

        self.train_data.z_tx = self.train_data.z_tx[indices]
        self.train_data.z_rx = self.train_data.z_rx[indices]
        self.train_data.labels = self.train_data.labels[indices]

        # ================================================================
        #                         Test Data
        # ================================================================
        self.test_data = CustomDataset(
            tx_path=GENERAL_PATH / 'test' / f'{self.tx_enc}.pt',
            rx_path=GENERAL_PATH / 'test' / f'{self.rx_enc}.pt',
        )

        # ================================================================
        #                         Val Data
        # ================================================================
        self.val_data = CustomDataset(
            tx_path=GENERAL_PATH / 'val' / f'{self.tx_enc}.pt',
            rx_path=GENERAL_PATH / 'val' / f'{self.rx_enc}.pt',
        )

        assert (
            self.train_data.input_size == self.test_data.input_size
            and self.train_data.input_size == self.val_data.input_size
        ), 'Input size must match between train, test and val data.'
        assert (
            self.train_data.output_size == self.test_data.output_size
            and self.train_data.output_size == self.val_data.output_size
        ), 'Output size must match between train, test and val data.'

        self.input_size = self.train_data.input_size
        self.output_size = self.train_data.output_size

        return None

    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """The function returns the predict DataLoader.

        Returns:
            DataLoader
                The predict DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class DataModuleClassifier(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset : str
            The name of the dataset.
        rx_enc : str
            The name of the receiver encoder.
        batch_size : int
            The size of a batch. Default 128.
        num_workers : int
            The number of workers. Setting it to 0 means that the data will be
            loaded in the main process. Default 0.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """

    def __init__(
        self,
        dataset: str,
        rx_enc: str,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.rx_enc: str = rx_enc
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    def prepare_data(self) -> None:
        """This function prepare the dataset (Download and Unzip).

        Returns:
            None
        """
        from dotenv import dotenv_values

        # Get from the .env file the zip file Google Drive ID
        ID = dotenv_values()['DATA_ID']

        # Download and unzip the data
        download_zip_from_gdrive(ID, name='latents', path='data')

        return None

    def setup(self, stage: str = None) -> None:
        """This function setups a DatasetRelativeDecoder for our data.

        Args:
            stage : str
                The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        self.train_data = DatasetClassifier(
            path=GENERAL_PATH / 'train' / f'{self.rx_enc}.pt'
        )
        self.test_data = DatasetClassifier(
            path=GENERAL_PATH / 'test' / f'{self.rx_enc}.pt'
        )
        self.val_data = DatasetClassifier(
            path=GENERAL_PATH / 'val' / f'{self.rx_enc}.pt'
        )

        assert (
            self.train_data.input_size == self.test_data.input_size
            and self.train_data.input_size == self.val_data.input_size
        ), 'Input size must match between train, test and val data.'
        assert (
            self.train_data.num_classes == self.test_data.num_classes
            and self.train_data.num_classes == self.val_data.num_classes
        ), 'The number of classes must match between train, test and val data.'

        self.input_size = self.train_data.input_size
        self.num_classes = self.train_data.num_classes
        return None

    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """The function returns the predict DataLoader.

        Returns:
            DataLoader
                The predict DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def main() -> None:
    """The main script loop in which we perform some sanity tests."""
    print('Start performing sanity tests...')
    print()

    # Setting inputs
    dataset = 'cifar10'
    encoder = 'vit_small_patch16_224'
    decoder = 'vit_base_patch16_224'

    print('Running first test...', end='\t')
    data = DataModule(dataset=dataset, tx_enc=encoder, rx_enc=decoder)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    print('Running second test...', end='\t')
    data = DataModuleClassifier(dataset=dataset, rx_enc=decoder)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
