"""
This python module handles the training of the linear optimizer for SAE in Federated scenario.
"""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import torch
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from src.utils import complex_gaussian_matrix
from src.linear_models import FederatedLinearOptimizer
from src.datamodules import DataModule


def main():
    """The main loop."""
    import argparse

    description = """
    This python module handles the training of the linear optimizer for SAE.

    To check available parameters run 'python /path/to/example_training_linear.py --help'.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    # parser.add_argument('-a',
    #                     '--agents',
    #                     help="Agents cardinality.",
    #                     type=int,
    #                     required=True,
    #                     default=None)
    # decoders_str = 'vit_base_patch16_224.pt, vit_small_patch16_224.pt, vit_small_patch32_224.pt, vit_tiny_patch16_224.pt'

    parser.add_argument(
        '-d', '--dataset', help='The dataset.', type=str, required=True
    )

    parser.add_argument(
        '--encoder', help='The encoder.', type=str, required=True
    )

    parser.add_argument(
        '--decoder',
        nargs='+',
        help='List of strings of different decoder models.',
        type=str,
        default=['vit_small_patch16_224'],
    )

    parser.add_argument(
        '-s',
        '--snr',
        help='The snr of the communication channel in dB. Set to None if unaware. Default None.',
        type=float,
        default=None,
    )

    parser.add_argument(
        '--transmitter',
        help='The number of antennas for the transmitter.',
        type=int,
        required=True,
    )

    parser.add_argument(
        '--receiver',
        help='The number of antennas for the receiver.',
        type=int,
        required=True,
    )

    parser.add_argument(
        '-i',
        '--iterations',
        help='The number of fitting iterations. Default None.',
        type=int,
        default=None,
    )

    parser.add_argument(
        '-c',
        '--cost',
        help='Transmission cost. Default 1.',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--rho',
        help='The rho parameter for admm. Default 1e2.',
        default=1e2,
        type=float,
    )

    parser.add_argument(
        '--seed',
        help='The seed for the analysis. Default 42.',
        default=42,
        type=int,
    )

    args = parser.parse_args()
    # Setting the seed
    seed_everything(args.seed, workers=True)
    agents = len(args.decoder)
    # Get the channel matrix
    # channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(args.receiver, args.transmitter))
    list_channel_matrix: list[torch.Tensor] = [
        complex_gaussian_matrix(
            mean=0,
            std=1,
            size=(args.receiver, args.transmitter),
        )
        for _ in range(agents)
    ]

    # =========================================================
    #                     Get the dataset
    # =========================================================

    # Initialize the datamodule
    list_datamodule = []
    for model_name in args.decoder:
        print('Datamodule list runned')
        datamodule = DataModule(
            dataset=args.dataset, encoder=args.encoder, decoder=model_name
        )

        # Prepare and setup the data
        datamodule.prepare_data()
        datamodule.setup()
        list_datamodule.append(datamodule)
        print('Datamodule list runned')

    # =========================================================
    #               Define the Linear Optimizer
    # =========================================================
    opt = FederatedLinearOptimizer(
        n_agents=agents,
        input_dim=list_datamodule[0].input_size,
        output_dim=list_datamodule[0].output_size,
        channel_matrix=list_channel_matrix,
        snr=args.snr,
        cost=args.cost,
        rho=args.rho,
    )

    # Fit the linear optimizer
    losses, traces = opt.fit(
        n_agents=len(args.decoder),
        datamodule_list=list_datamodule,
        iterations=args.iterations,
    )

    # Eval the linear optimizer
    print(
        'loss:',
        opt.eval(
            input=datamodule.test_data.z, output=datamodule.test_data.z_decoder
        ),
    )

    print('trace(FF^H):', torch.trace(opt.F @ opt.F.H).real.item())
    if args.cost:
        print(torch.trace(opt.F.H @ opt.F).real.item() <= args.cost)

    pl.DataFrame(
        {
            'Iterations': range(0, len(losses)),
            'Losses': losses,
            'Traces': traces,
        }
    ).write_parquet('convergence.parquet')

    print(pl.read_parquet('convergence.parquet'))

    fig, axs = plt.subplots(ncols=2, nrows=2)
    sns.lineplot(x=range(0, len(losses)), y=losses, ax=axs[0, 0]).set(
        title='Convergence', ylabel='MSE Loss', xlabel='Iteration'
    )
    sns.lineplot(x=range(0, len(losses)), y=losses, ax=axs[0, 1]).set(
        title='Convergence Log Scaled',
        ylabel='MSE Loss',
        xlabel='Iteration',
        yscale='log',
    )
    sns.lineplot(x=range(0, len(traces)), y=traces, ax=axs[1, 0]).set(
        title='Convergence', ylabel='tr FF^H', xlabel='Iteration'
    )
    sns.lineplot(x=range(0, len(traces)), y=traces, ax=axs[1, 1]).set(
        title='Convergence Log Scaled',
        ylabel='tr FF^H',
        xlabel='Iteration',
        yscale='log',
    )
    plt.show()

    return None


if __name__ == '__main__':
    main()
