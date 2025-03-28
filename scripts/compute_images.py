"""A usefull script used to compute the needed plots.

The script expects to have the results saved in the following structure (both linear and baseline results):
|_ results/
    |_ linear_model/
    |   |_ r1.parquet
    |   |_ ...
    |   |_ rk.parquet
    |_ baseline/
        |_ r1.parquet
        |_ ...
        |_ rk.parquet
"""

import polars as pl
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


def main() -> None:
    """The main loop."""
    # Defining some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'
    IMG_PATH: Path = CURRENT / 'img'

    # Create image Path
    IMG_PATH.mkdir(exist_ok=True)

    # Set style
    plt.rcParams.update(
        {
            'figure.figsize': (12, 8),
            'font.size': 22,
            'axes.titlesize': 24,
            'axes.labelsize': 24,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 20,
            'legend.title_fontsize': 22,
            'lines.markersize': 18,
            'lines.linewidth': 3,
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times'],
        }
    )

    # Retrieve Data
    df: pl.DataFrame = (
        pl.read_parquet(RESULTS_PATH / 'linear_model/*.parquet')
        .with_columns(
            (
                (pl.col('Channel Usage') / pl.col('Latent Complex Dim').max())
                * 100
            )
            .round(2)
            .alias('Compression Factor'),
        )
        .group_by(
            [
                'Case',
                'Antennas Transmitter',
                'Channel Usage',
                'Compression Factor',
                'Seed',
                'SNR',
            ]
        )
        .agg(pl.col('Accuracy').mean())
        .sort('Antennas Transmitter')
        .vstack(
            pl.read_parquet(RESULTS_PATH / 'baseline/*.parquet')
            .with_columns(
                (
                    (
                        pl.col('Channel Usage')
                        / pl.col('Latent Complex Dim').max()
                    )
                    * 100
                )
                .round(2)
                .alias('Compression Factor'),
            )
            .group_by(
                [
                    'Case',
                    'Antennas Transmitter',
                    'Channel Usage',
                    'Compression Factor',
                    'Seed',
                    'SNR',
                ]
            )
            .agg(pl.col('Accuracy').mean())
            .sort(['Case', 'Antennas Transmitter'])
        )
        .rename(
            {
                'Antennas Transmitter': 'Channel',
            }
        )
        .with_columns(
            (
                (pl.col('Channel')).cast(pl.String)
                + 'x'
                + (pl.col('Channel')).cast(pl.String)
            ).alias('Channel'),
        )
    )

    # ===================================================================================
    #                          Accuracy Vs Compression Factor
    # ===================================================================================
    filter = pl.col('SNR') == 20.0

    ax = sns.lineplot(
        df.filter(filter),
        x='Compression Factor',
        y='Accuracy',
        style='Channel',
        hue='Case',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.3),
    )
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsCompression_factor.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsCompression_factor.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                          Accuracy Vs Signal to Noise Ratio
    # ===================================================================================
    filter = pl.col('Channel') == '4x4'
    ch_usage = (df.filter(filter & (pl.col('Case').str.contains('Linear'))))[
        'Channel Usage'
    ].max()

    ax = sns.lineplot(
        df.filter(filter & (pl.col('Channel Usage') <= ch_usage)),
        x='SNR',
        y='Accuracy',
        hue='Case',
        style='Compression Factor',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.3),
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsSNR.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsSNR.png'),
        bbox_inches='tight',
    )

    return None


if __name__ == '__main__':
    main()
