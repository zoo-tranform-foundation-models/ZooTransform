import matplotlib.pyplot as plt
import seaborn as sns


def plot_loglikelihood_vs_dmscore(df):
    """
    Plots LogLikelihood and LogLikelihoodPre against DMS_score for each species,
    with improved aesthetics and layout.
    """

    unique_species = df['Species'].unique()
    num_species = len(unique_species)

    # Create subplots
    fig, axes = plt.subplots(1, num_species, figsize=(6 * num_species, 5), sharey=True)

    if num_species == 1:
        axes = [axes]

    for ax, sp in zip(axes, unique_species):
        species_data = df[df['Species'] == sp]

        # Scatter plots with alpha transparency and consistent color palette
        sns.scatterplot(
            x='LogLikelihood', y='DMS_score',
            data=species_data, ax=ax,
            label='LogLikelihood', s=60, alpha=0.7, color='#1f77b4'
        )
        sns.scatterplot(
            x='LogLikelihoodPre', y='DMS_score',
            data=species_data, ax=ax,
            label='LogLikelihoodPre', s=60, alpha=0.7, color='#ff7f0e'
        )

        sns.regplot(
            x='LogLikelihood', y='DMS_score',
            data=species_data, ax=ax,
            scatter=False, color='#1f77b4', ci=None, line_kws={'lw': 2, 'ls': '--'}
        )
        sns.regplot(
            x='LogLikelihoodPre', y='DMS_score',
            data=species_data, ax=ax,
            scatter=False, color='#ff7f0e', ci=None, line_kws={'lw': 2, 'ls': '--'}
        )

        ax.set_title(sp, fontsize=16, weight='bold')
        ax.set_xlabel('LogLikelihood', fontsize=14)
        ax.set_ylabel('DMS Score', fontsize=14)
        ax.legend(frameon=True, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
