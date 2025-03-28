"""
A useful script used to compute the needed plots.

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
    |_ baseline_TK/
        |_ r1.parquet
        |_ ...
        |_ rk.parquet
"""

import polars as pl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


def process_parquet_files(path_pattern: Path) -> pl.DataFrame:
    """
    Reads and processes all parquet files in the given folder (using glob for '*.parquet').
    Applies the same transformation to each file:
      - Compute Compression Factor,
      - Cast SNR to Float64,
      - Group by columns and aggregate Accuracy.
    Returns a concatenated DataFrame sorted by 'Antennas Transmitter'.
    """
    files = list(path_pattern.glob("*.parquet"))
    df_list = []
    for file in files:
        df_temp = (
            pl.read_parquet(file, allow_missing_columns=True)
            .with_columns([
                (((pl.col('Channel Usage') / pl.col('Latent Complex Dim').max()) * 100)
                    .round(2)).alias('Compression Factor'),
                pl.col("SNR").cast(pl.Float64)
            ])
            .group_by([
                'Case',
                'Antennas Transmitter',
                'Channel Usage',
                'Compression Factor',
                'Seed',
                'SNR',
            ])
            .agg(pl.col('Accuracy').mean())
        )
        df_list.append(df_temp)
    if df_list:
        return pl.concat(df_list).sort('Antennas Transmitter')
    else:
        return pl.DataFrame()


def main() -> None:
    """The main loop."""
    # Defining some useful paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results'
    IMG_PATH: Path = CURRENT / 'img'
    IMG_PATH.mkdir(exist_ok=True)

    # Set matplotlib style
    plt.rcParams.update({
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
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
    })

    # Process each results folder separately
    df_linear = process_parquet_files(RESULTS_PATH / 'linear_model')
    df_baseline = process_parquet_files(RESULTS_PATH / 'baseline')
    df_baseline_TK = process_parquet_files(RESULTS_PATH / 'baseline_TK')

    # Stack all DataFrames together
    df = pl.concat([df_linear, df_baseline, df_baseline_TK])
    
    # Rename and reformat the channel column:
    # Rename 'Antennas Transmitter' to 'Channel' and then format it as 'NxN'
    df = df.rename({'Antennas Transmitter': 'Channel'})
    df = df.with_columns(
        (pl.col('Channel').cast(pl.Utf8) + 'x' + pl.col('Channel').cast(pl.Utf8)).alias('Channel')
    )

    # ===================================================================================
    #                      Accuracy Vs Compression Factor
    # ===================================================================================
    # Filter for SNR == 20.0
    snr_filter = pl.col('SNR') == 20.0
    df_plot = df.filter(snr_filter)

    # Convert to pandas for seaborn
    df_plot_pd = df_plot.to_pandas()
    
    ax = sns.lineplot(
        data=df_plot_pd,
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
    plt.savefig(str(IMG_PATH / 'AccuracyVsCompression_factor.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(IMG_PATH / 'AccuracyVsCompression_factor.png'), bbox_inches='tight')
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                      Accuracy Vs Signal to Noise Ratio
    # ===================================================================================
    channel_filter = pl.col('Channel') == '4x4'
    # We filter for a specific channel and then determine the maximum Channel Usage value for the 'Linear' case.
    ch_usage = df.filter(channel_filter & (pl.col('Case').str.contains('Linear')))['Channel Usage'].max()
    df_plot2 = df.filter(channel_filter & (pl.col('Channel Usage') <= ch_usage)).to_pandas()
    
    ax = sns.lineplot(
        data=df_plot2,
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
    plt.savefig(str(IMG_PATH / 'AccuracyVsSNR.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(IMG_PATH / 'AccuracyVsSNR.png'), bbox_inches='tight')
    plt.clf()
    plt.cla()

    return None


if __name__ == '__main__':
    main()
