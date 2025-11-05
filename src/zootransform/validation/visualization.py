import matplotlib.pyplot as plt
import seaborn as sns


def plot_loglikelihood_vs_dmscore(df):
    """
    Plots LogLikelihood and LogLikelihoodPre against DMS_score for each species,
    arranging species plots in two rows: first with 5, then with 4.
    """
    unique_species = df['Species'].unique()
    num_species = len(unique_species)

    # Determine how to split species into rows
    first_row = 5
    second_row = 4

    if num_species < 9:
        first_row = num_species // 2 + num_species % 2
        second_row = num_species - first_row

    nrows = 2 if num_species > 1 else 1
    ncols = max(first_row, second_row)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)

    # Flatten axes for easier indexing, handle both (2D) and (1D) cases
    if nrows == 1:
        axes = [axes] if num_species == 1 else axes
    else:
        axes = axes.flatten()

    # Pad species list to match grid size for proper looping (some axes are unused)
    padded_species = list(unique_species) + [None]*(nrows*ncols - num_species)

    for i, sp in enumerate(padded_species):
        ax = axes[i]
        if sp is None:
            ax.set_visible(False)
            continue

        species_data = df[df['Species'] == sp]

        sns.scatterplot(
            x='Likelihood', y='DMS_score',
            data=species_data, ax=ax,
            label='Likelihood', s=60, alpha=0.7, color='#1f77b4'
        )
        sns.scatterplot(
            x='LikelihoodPre', y='DMS_score',
            data=species_data, ax=ax,
            label='LikelihoodPre', s=60, alpha=0.7, color='#ff7f0e'
        )

        sns.regplot(
            x='Likelihood', y='DMS_score',
            data=species_data, ax=ax,
            scatter=False, color='#1f77b4', ci=None, line_kws={'lw': 2, 'ls': '--'}
        )
        sns.regplot(
            x='LikelihoodPre', y='DMS_score',
            data=species_data, ax=ax,
            scatter=False, color='#ff7f0e', ci=None, line_kws={'lw': 2, 'ls': '--'}
        )

        ax.set_title(sp, fontsize=16, weight='bold')
        ax.set_xlabel('Likelihood', fontsize=14)
        ax.set_ylabel('DMS Score', fontsize=14)
        ax.legend(frameon=True, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
