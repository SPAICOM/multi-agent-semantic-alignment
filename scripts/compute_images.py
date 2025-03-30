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

    # Set sns style
    sns.set_style('whitegrid')

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
                'Simulation',
            ]
        )
        .agg(
            pl.col('Accuracy').mean(),
            pl.col('Loss').sum(),
        )
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
                    'Simulation',
                ]
            )
            .agg(
                pl.col('Accuracy').mean(),
                pl.col('Loss').sum(),
            )
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

    # Define ticks
    ticks = list(map(int, df['Compression Factor'].unique().to_list()))

    # ===================================================================================
    #                          Accuracy Vs Compression Factor
    # ===================================================================================
    filter = pl.col('Simulation') == 'compr_fact'

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
    plt.xticks(ticks, labels=ticks)
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
    #                          MSE Vs Compression Factor
    # ===================================================================================
    filter = pl.col('Simulation') == 'compr_fact'

    ax = sns.lineplot(
        df.filter(filter),
        x='Compression Factor',
        y='Loss',
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
    plt.ylabel('MSE')
    plt.xticks(ticks, labels=ticks)
    plt.savefig(
        str(IMG_PATH / 'MSEVsCompression_factor.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'MSEVsCompression_factor.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                          Accuracy Vs Signal to Noise Ratio
    # ===================================================================================
    filter = pl.col('Simulation') == 'snr'

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
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                          MSE & Accuracy - Homogeneous Vs Heterogeneous
    # ===================================================================================
    filter = (pl.col('Simulation') == 'homogeneous') | (
        pl.col('Simulation') == 'heterogeneous'
    )

    plot_df = (
        df.filter(filter)
        .drop('Case')
        .rename({'Simulation': 'Case'})
        .sort(['Case'], descending=True)
    )

    ax = sns.lineplot(
        plot_df,
        x='Compression Factor',
        y='Loss',
        style='Case',
        hue='Case',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.2),
    )
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.ylabel('MSE')
    plt.xticks(ticks, labels=ticks)
    plt.savefig(
        str(IMG_PATH / 'AlignmentStruggle.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AlignmentStruggle.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    ax = sns.lineplot(
        plot_df,
        x='Compression Factor',
        y='Accuracy',
        style='Case',
        hue='Case',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.2),
    )
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.xticks(ticks, labels=ticks)
    plt.savefig(
        str(IMG_PATH / 'AccuracyGroups.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyGroups.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


if __name__ == '__main__':
    main()
