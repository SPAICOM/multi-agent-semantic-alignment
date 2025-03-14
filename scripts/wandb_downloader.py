"""This python module handles the download of the trained models from wandb.

To check available parameters run 'python /path/to/wandb_downloader.py --help'.
"""

import wandb
import shutil
from pathlib import Path


# ============================================================
#
#                     METHODS DEFINITION
#
# ============================================================


def find_and_rename_ckpt_files(original_path: str, new_path: str) -> None:
    """Save the results in a convenient directory structure.

    Args:
        original_path : str
            The directory where the files are currently saved.
        new_path : str
            The new directory where to save the files.

    Returns:
        None
    """
    original_path = Path(original_path)
    new_path = Path(new_path)

    # Ensure the new directory exists
    new_path.mkdir(parents=True, exist_ok=True)

    # Find all .ckpt files in subdirectories
    ckpt_files = list(original_path.rglob('*.ckpt'))

    new_path = f'{new_path}.ckpt'
    shutil.move(str(ckpt_files[0]), str(new_path))
    print(f'Moved: {ckpt_files[0]} to {new_path}')

    return None


def download_classifier(
    org: str,
    rx_enc: str,
    dataset: str,
    seed: int,
    session,
) -> None:
    """
    Args:
        org : str
            The name of the wandb organization.
        rx_enc : str
            The receiver encoder name.
        dataset : str
            The dataset name.
        seed : int
            The seed of the model.
        session :
            The wandb session.

    Return:
        None
    """
    session.use_artifact(
        f'{org}/multi_agent_semantic_alignment__classifier/model-{rx_enc}_{seed}_{dataset}:best',
        type='model',
    ).download()
    find_and_rename_ckpt_files(
        'artifacts',
        f'models/classifiers/{dataset}/{rx_enc}/seed_{seed}/seed_{seed}',
    )
    return None


# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================


def main() -> None:
    """The main loop."""
    import argparse

    description = """
    This python module handles the download of the trained models from wandb.

    To check available parameters run 'python /path/to/wandb_downloader.py --help'.
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-o',
        '--org',
        help='The organization of wandb where the models are saved.',
        type=str,
    )

    args = parser.parse_args()

    # Init wandb
    run = wandb.init()

    # Define needed variables
    datasets = ['cifar10']
    seeds = [27, 42, 100, 123, 144, 200]
    rx_encs = [
        'vit_tiny_patch16_224',
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
        'vit_base_patch32_clip_224',
        'rexnet_100',
        'mobilenetv3_small_075',
        'mobilenetv3_large_100',
        'mobilenetv3_small_100',
        'efficientvit_m5.r224_in1k',
        'levit_128s.fb_dist_in1k',
    ]

    # Donwload Classifiers
    for dataset in datasets:
        for rx_enc in rx_encs:
            for seed in seeds:
                download_classifier(
                    org=args.org,
                    rx_enc=rx_enc,
                    dataset=dataset,
                    seed=seed,
                    session=run,
                )

    # Close wandb
    wandb.finish()

    return None


if __name__ == '__main__':
    main()
