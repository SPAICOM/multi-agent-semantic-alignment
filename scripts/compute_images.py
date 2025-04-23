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
import matplotlib.lines as mlines


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
    plt.style.use('.conf/plotting/plt.mplstyle')

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
                'Training Subset Ratio',
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
                    'Training Subset Ratio',
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
                'Training Subset Ratio': 'Semantic Pilots Percentage',
            }
        )
        .sort('Semantic Pilots Percentage', descending=False)
        .with_columns(
            (
                (pl.col('Channel')).cast(pl.String)
                + 'x'
                + (pl.col('Channel')).cast(pl.String)
            ).alias('Channel'),
            (pl.col('Semantic Pilots Percentage') * 100).cast(pl.String)
            + r' \%',
        )
    )

    # Define ticks
    ticks = list(map(int, df['Compression Factor'].unique().to_list()))

    # ===================================================================================
    #                          Accuracy Vs Compression Factor
    # ===================================================================================
    filter = pl.col('Simulation') == 'compr_fact'

    case_order = [
        'Federated Semantic Alignment',
        'Multi-Link Semantic Alignment',
        'Baseline First-K',
        'Baseline Top-K',
    ]

    ax = sns.lineplot(
        df.filter(filter),
        x='Compression Factor',
        y='Accuracy',
        style='Channel',
        hue='Case',
        hue_order=case_order,
        markers=True,
    )
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Separate by Case and Channel
    case_labels = case_order
    channel_labels = ['1x1', '2x2', '4x4']

    # Match labels to handles
    case_handles = [handles[labels.index(cl)] for cl in case_labels]
    channel_handles = [handles[labels.index(cl)] for cl in channel_labels]

    # First legend: Channel
    legend1 = ax.legend(
        channel_handles,
        channel_labels,
        title='Channel',
        loc='center right',
        bbox_to_anchor=(1, 0.6),
        ncol=1,
        frameon=True,
        framealpha=1,
    )

    # Second legend: Case
    ax.legend(
        case_handles,
        case_labels,
        title='Case',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=True,
    )

    # Add both legends manually
    ax.add_artist(legend1)
    # sns.move_legend(
    #     ax,
    #     'upper center',
    #     ncol=2,
    #     frameon=True,
    #     bbox_to_anchor=(0.5, 1.3),
    # )
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

    ch_usage = (df.filter(filter & (pl.col('Case').str.contains('Semantic'))))[
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
        style='Semantic Pilots Percentage',
        hue='Case',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.2),
    )
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.ylabel('Network MSE')
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
        style='Semantic Pilots Percentage',
        hue='Case',
        markers=True,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=3,
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

    # ===================================================================================
    #                          Both MSE & Accuracy - Homogeneous Vs Heterogeneous
    # ===================================================================================

    # Create figure and main axis
    fig, ax1 = plt.subplots()

    # For Loss plots, we'll use solid lines
    sns.lineplot(
        data=plot_df,
        x='Compression Factor',
        y='Loss',
        hue='Case',
        style='Semantic Pilots Percentage',
        markers=True,
        dashes=False,  # Use solid lines for Loss
        ax=ax1,
        legend=False,  # Don't show the automatic legend
    )

    # Manually set all lines for Loss to be solid
    for line in ax1.get_lines():
        line.set_linestyle('-')  # Set all Loss lines to solid

    ax1.set_ylabel('Network MSE', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create secondary y-axis for Accuracy
    ax2 = ax1.twinx()

    # For Accuracy plots, we'll use dashed lines
    sns.lineplot(
        data=plot_df,
        x='Compression Factor',
        y='Accuracy',
        hue='Case',
        style='Semantic Pilots Percentage',
        markers=True,
        dashes=True,  # Use dashed lines for Accuracy
        ax=ax2,
        legend=False,  # Don't show the automatic legend
    )

    # Manually set all lines for Accuracy to be dashed
    for line in ax2.get_lines():
        line.set_linestyle('--')  # Set all Accuracy lines to dashed

    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(None, 0.9)  # Set the Accuracy y-axis range

    # X-axis settings
    ax1.set_xlabel(r'Compression Factor $\zeta$ (\%)')

    # Get unique values for Case and Semantic Pilots Percentage
    # case_values = plot_df['Case'].unique().sort(descending=False)
    case_values = ['homogeneous', 'heterogeneous']
    dataset_perc_values = plot_df['Semantic Pilots Percentage'].unique()

    # Extract the actual colors used by seaborn for each case
    color_palette = dict(
        zip(case_values, sns.color_palette(n_colors=len(case_values)))
    )

    # Extract actual markers used by seaborn
    all_lines = ax1.get_lines()
    style_mapping = {}

    # Create a dictionary to map Semantic Pilots Percentage to the marker used
    for line in all_lines:
        # Try to get the style attribute from the line's properties
        if hasattr(line, '_style_kw'):
            dataset_perc = line._style_kw.get('label', None)
            if dataset_perc in dataset_perc_values:
                style_mapping[dataset_perc] = {'marker': line.get_marker()}

    # If we couldn't extract styles, use default ones
    dataset_perc_values = sorted(
        dataset_perc_values, key=lambda x: float(x.split(' ')[0])
    )
    if not style_mapping:
        markers = ['o', 'X', 's']
        style_mapping = {
            dp: {'marker': markers[i % len(markers)]}
            for i, dp in enumerate(dataset_perc_values)
        }

    # Create custom legend handles for Case (colors only)
    case_handles = []
    for case in case_values:
        case_handles.append(
            mlines.Line2D(
                [],
                [],
                color=color_palette.get(case),
                marker=None,
                markersize=0,
                linestyle='-',
                label=case,
            )
        )

    # Create custom legend handles for Semantic Pilots Percentage (markers only)
    dataset_handles = []
    for perc in dataset_perc_values:
        style = style_mapping.get(perc, {'marker': 'o'})
        dataset_handles.append(
            mlines.Line2D(
                [],
                [],
                color='black',  # Use neutral color
                marker=style['marker'],
                linestyle='',
                label=perc,
            )
        )

    # # Create custom legend handles for Loss vs Accuracy (line styles)
    metric_handles = [
        mlines.Line2D(
            [],
            [],
            color='black',
            marker=None,
            linestyle='-',
            label='Network MSE',
        ),
        mlines.Line2D(
            [],
            [],
            color='black',
            marker=None,
            linestyle='--',
            label='Accuracy',
        ),
    ]

    # Add three separate legends
    case_legend = ax1.legend(
        handles=case_handles,
        loc='upper center',
        bbox_to_anchor=(0.22, 1.22),
        ncol=len(case_values),
        title='Case',
    )

    metric_legend = ax1.legend(
        handles=metric_handles,
        loc='center right',
        ncol=1,
        framealpha=1,
        title='Metric',
    )

    ax1.legend(
        handles=dataset_handles,
        loc='upper center',
        bbox_to_anchor=(0.8, 1.22),
        ncol=len(dataset_perc_values),
        columnspacing=0.5,
        title='Semantic Pilots Percentage',
    )

    # Add all legends back (this is a trick to show multiple legends)
    ax1.add_artist(case_legend)
    ax1.add_artist(metric_legend)

    # Show and/or save
    plt.tight_layout()
    plt.xticks(ticks, labels=ticks)
    plt.savefig(
        str(IMG_PATH / 'MSE&Accuracy_struggle.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'MSE&Accuracy_struggle.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


if __name__ == '__main__':
    main()
