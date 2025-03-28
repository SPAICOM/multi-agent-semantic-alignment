"""The following python module contains available methods for handling downloads."""

from pathlib import Path
from gdown import download
from zipfile import ZipFile

# =======================================================
#
#                 METHODS DEFINITION
#
# =======================================================


def download_zip_from_gdrive(
    id: str,
    name: str,
    path: str,
) -> None:
    """A method to download a zip file containing all the needed data.
    The method will save the data in the data/<name>/ directory.

    Args:
        id : str
            The gdown id of the zip file.
        name : str
            The name of the subdirectory inside data.
        path: str
            The path where to download the zip.

    Returns:
        None
    """
    CURRENT = Path('.')
    DATA_DIR = CURRENT / path
    ZIP_PATH = DATA_DIR / f'{name}.zip'
    DIR_PATH = DATA_DIR / f'{name}/'

    # Make sure that DATA_DIR exists
    DATA_DIR.mkdir(exist_ok=True)

    # Check if the zip file is already in the path
    if not ZIP_PATH.exists():
        # Download the zip file
        download(id=id, output=str(ZIP_PATH))

    # Check if the directory exists
    if not DIR_PATH.is_dir():
        # Unzip the zip file
        with ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ZIP_PATH.parent)

    return None


# =======================================================
#
#                     MAIN LOOP
#
# =======================================================


def main() -> None:
    """Test loop."""
    print('Start performing sanity tests...')
    print()

    from dotenv import dotenv_values

    print('Running first test...', end='\t')
    id = dotenv_values()
    download_zip_from_gdrive(id=id, path='data', name='latents')
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
