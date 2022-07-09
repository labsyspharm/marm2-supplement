import sys

from MARM.analysis import (
    read_all_analysis_dataframes, read_settings, extend_drug_adapted,
    get_signal_deconvolution_df, get_signal_transduction_df, write_timestamp
)
from MARM.paths import get_figure_dir
from MARM.visualize import (
    plot_gains, plot_deconvolution,  plot_contextualized_graph,
    plot_raf_states, plot_drug_free_monomers, plot_raf_dimerization,
)
from MARM.estimation import RAFI, PANRAFI

sxs = read_settings(sys.argv, index=False, threads=False)
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])
drug = sys.argv[4]
perturbations = ''

drug_label = f'{drug.replace("_", "")} [$\mu$M]'
df = read_all_analysis_dataframes(sxs, f'transduction__{drug}__{perturbations}')

if drug in RAFI + PANRAFI:
    states = ['RAF_marginal_RAFi', 'RAF_marginal_RAS', 'RAF_marginal_MEK',
              'RAF_marginal_RAF',
              'baseline_R', 'baseline_IR', 'baseline_RR', 'baseline_RRI',
              'baseline_IRRI']

    total_RAF = df[['baseline_R', 'baseline_IR']].sum(axis=1) + 2 * \
                df[['baseline_RRI', 'baseline_IRRI']].sum(axis=1)

    for state in states:
        if state in df.columns:
            df[state] = df[state].div(total_RAF, axis=0)

    plot_raf_states(df, f'{drug}_0', drug_label, figdir,
                    f'rafstates_{drug}_{perturbations}.pdf')

plot_drug_free_monomers(df, f'{drug}_0', drug_label, figdir,
                        f'drugfree_{drug}_{perturbations}.pdf')
if drug in MEKI:
    plot_drug_free_monomers(df, f'{drug}_0', drug_label, figdir,
                            f'drugfree_{drug}_{perturbations}_channel.pdf',
                            channel=True)
plot_raf_dimerization(df, f'{drug}_0', drug_label, figdir,
                      f'dimerization_{drug}_{perturbations}.pdf')

df = extend_drug_adapted(df, 0.1)

for cond, cond_label in zip([lambda f: f.EGFR_crispr == 1.0,
                             lambda f: f.EGFR_crispr == 10.0],
                            ['EGFRwt', 'EGFRa']):
    subset = df[(df.time <= 1) & cond(df)]

    df_deconv = get_signal_deconvolution_df(subset, f'{drug}_0', 0)
    plot_deconvolution(
        df_deconv, f'{drug}_0', figdir,
        f'deconv_{cond_label}_{drug}_{perturbations}.pdf'
    )

    for mode in ['peak', 'int_log10']:
        df_gains = get_signal_transduction_df(
            sxs,
            f'transduction__{drug}__{perturbations}',
            lambda frame: (frame.time <= 1) & cond(frame),
            [f'{drug}_0'],
            mode
        )
        plot_gains(
            df_gains, f'{drug}_0', drug_label, figdir,
            f'gains_{mode}_{cond_label}_{drug}_{perturbations}.pdf'
        )

        plot_contextualized_graph(
            df_gains, f'{drug}_0', 5, figdir,
            f'rewiring_{mode}_{cond_label}_{drug}_{perturbations}.pdf'
        )

write_timestamp(figdir, f'transduction__{drug}__{perturbations}')
