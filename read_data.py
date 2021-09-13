import os
import sys
import pandas as pd
import numpy as np
import re
import MARM

from MARM.estimation import RAFI, MEKI, PANRAFI

DATAFILES_ESTIMATION = ['D1', 'D4', 'D5', 'D6', 'D7', 'D8', 'DP1', 'DP2',
                        'DT1', 'DT2', 'DT3']
DATAFILES_PREDICTION_COMBO = ['combo_egf', 'combo_no_egf', 'combo_egf_crispr',
                              'combo_no_egf_crispr']
DATAFILES_PANRAFCOMBO = ['20180407_EGF_RAFi_MEKi_data_extended.xls']
DATAFILES_PREDICTION_SINGLE = ['D4', 'D6', 'D7', 'D8']
DATAFILES_MUTRAS_ENGINEERED = [
    '20200117_A375_NRASQ61K_unidimensional_pERK_DOX_RAFi_MEKi',
]
DATAFILES_MUTRAS_ENGINEERED_COMBO = [
    '20200119_A375_NRASQ61K_matrix_pERK_DOX_RAFi_MEKi',
    '20200130_A375_NRASQ61K_matrix_pERK_DOX_RAFifixed_panRAFi_MEKi.xls'
]
DATAFILES_HT29 = ['20200128_HT29_matrix_EGF_RAFi_MEKi']
DATAFOLDER = os.path.join(os.path.dirname(MARM.__file__), 'data')

pERK_IF_std = 0.2
pMEK_IF_std = 0.2


def fill_data_dict(data, data_dict):
    for drug in RAFI + PANRAFI + MEKI:
        for key in [f'{drug}_0', f'{drug}_0_preeq',
                    f'{drug}_0_presim']:
            if key not in data_dict.keys():
                data_dict[key] = 0.0

    for key in data.columns.values:
        if key not in data_dict.keys():
            data_dict[key] = float('nan')


def read_IF_experiment(data, df, specs):
    for spec in specs:
        xx = df.values[spec['pERK_mean_anchor'][0] - 1,
                       (spec['pERK_mean_anchor'][1]):(spec['pERK_mean_anchor'][1]+spec['x_points'])]
        yy = df.values[(spec['pERK_mean_anchor'][0]):(spec['pERK_mean_anchor'][0]+spec['y_points']),
                        spec['pERK_mean_anchor'][1] - 1]
        for x_index, x in enumerate(xx):
            if spec['x_axis'] in spec and float(x) != spec[spec['x_axis']]:
                continue
            for y_index, y in enumerate(yy):
                if spec['y_axis'] in spec and float(y) != spec[spec['y_axis']]:
                    continue
                # read in data from dataframe
                data_dict = {
                    'pERK_IF_obs':
                        float(df.values[spec['pERK_mean_anchor'][0] + y_index,
                                  spec['pERK_mean_anchor'][1] + x_index]),
                }

                if 'pERK_std_anchor' in spec:
                    data_dict['pERK_IF_obs_std'] = \
                        float(df.values[spec['pERK_std_anchor'][0] + y_index,
                                  spec['pERK_std_anchor'][1] + x_index])
                elif 'pERK_IF_obs_std' in spec:
                    data_dict['pERK_IF_obs_std'] = \
                        spec['pERK_IF_obs_std']

                if 'pMEK_mean_anchor' in spec:
                    data_dict['pMEK_IF_obs'] = \
                        float(df.values[spec['pMEK_mean_anchor'][0] + y_index,
                                  spec['pMEK_mean_anchor'][1] + x_index])

                if 'pMEK_std_anchor' in spec:
                    data_dict['pMEK_IF_obs_std'] = \
                        float(df.values[spec['pMEK_std_anchor'][0] + y_index,
                                  spec['pMEK_std_anchor'][1] + x_index])
                elif 'pMEK_IF_obs_std' in spec:
                    data_dict['pMEK_IF_obs_std'] = \
                        spec['pMEK_IF_obs_std']

                data_dict[spec['x_axis']] = float(x)
                data_dict[spec['y_axis']] = float(y)
                # read in data from experiment specification
                for val in ['time', 't_presim',
                            'EGF_0', 'EGF_0_preeq', 'EGF_0_presim',
                            'Vemurafenib_0', 'Vemurafenib_0_preeq',
                            'Vemurafenib_0_presim',
                            'Cobimetinib_0', 'Cobimetinib_0_preeq',
                            'Cobimetinib_0_presim',
                            'EGFR_crispr', 'EGFR_crispr_preeq',
                            'EGFR_crispr_presim']:
                    if val not in [spec['x_axis'], spec['y_axis']]:
                        data_dict[val] = spec[val]

                fill_data_dict(data, data_dict)

                # append datapoint to dataframe
                data.loc[len(data)] = data_dict


protein_aliases = {
    'BRAF':     'BRAF',
    'CRAF':     'CRAF',
    'MEK1':     'MEK',
    'MAP2K1':   'MEK',
    'MEK2':     'MEK',
    'MAP2K2':   'MEK',
    'EGFR':     'EGFR',
    'HRAS':     'RAS',
    'KRAS':     'RAS',
    'NRAS':     'RAS',
    'ERK1':     'ERK',
    'MAPK1':    'ERK',
    'ERK2':     'ERK',
    'MAPK3':    'ERK',
    'SOS1':     'SOS',
    'DUSP4':    'DUSP',
    'DUSP6':    'DUSP',
    'SPRY2':    'SPRY',
    'SPRY4':    'SPRY',
    'CBL':      'CBL',
}


def read_MS_experiment(data, df, specs):
    for spec in specs:
        xx = df.values[
             spec['mean_anchor'][0] - 1,
             (spec['mean_anchor'][1]):(spec['mean_anchor'][1]+spec['x_points'])
             ]
        for x_index, x in enumerate(xx):
            # read in data from dataframe
            data_dict = {
            }
            for obs in data.columns.values:
                rows = []
                if spec.get('proteomics', False):
                    # DP1
                    rows = [
                        row
                        for row in df.index.values
                        if (str(row[1]) in protein_aliases.keys() and
                            f't{protein_aliases[str(row[1])]}_obs' == obs)
                        or f't{row[1]}_obs' == obs
                    ]
                else:
                    # DP2
                    if obs == 'pERK_obs':
                        rows = [
                            row
                            for row in df.index.values
                            if re.match('^ERK[1|2]+$', str(row[1]))
                               and (row[3] == 'Y187+T185+T185&Y187' or row[
                            3] == 'Y204+T202+T202&Y204')
                        ]
                    elif obs == 'pS1134SOS1_obs':
                        rows = [
                            row
                            for row in df.index.values
                            if re.match('^SOS1$', str(row[1]))
                            and row[3] == 'S1134'
                        ]
                if rows:
                    m = df.loc[rows, :].values[
                            :, spec['mean_anchor'][1]+x_index
                    ].sum()

                    s = np.sqrt(np.power(df.loc[rows, :].values[
                        :, spec['std_anchor'][1]+x_index
                    ], 2.0).sum(axis=0))
                    if spec.get('proteomics', False):
                        data_dict[obs] = np.log(m)
                        data_dict[obs + '_std'] = s/m
                    else:
                        data_dict[obs] = m
                        data_dict[obs + '_std'] = s

            for val in spec['x_axis']:
                data_dict[val] = x
            # read in data from experiment specification
            for val in [
                'time', 't_presim',
                'EGF_0', 'EGF_0_preeq', 'EGF_0_presim',
                'Vemurafenib_0', 'Vemurafenib_0_preeq', 'Vemurafenib_0_presim',
                'EGFR_crispr', 'EGFR_crispr_preeq', 'EGFR_crispr_presim'
            ]:
                if val not in spec['x_axis']:
                    data_dict[val] = spec[val]

            fill_data_dict(data, data_dict)

            # append datapoint to dataframe
            data.loc[len(data)] = data_dict


def read_T_experiment(data, df, specs):
    for spec in specs:
        xx = df.values[
             spec['mean_anchor'][0] - 1,
             (spec['mean_anchor'][1]):(spec['mean_anchor'][1]+spec['x_points'])
             ]
        for x_index, x in enumerate(xx):
            # read in data from dataframe
            data_dict = {
            }
            for obs in data.columns.values:
                rows = [
                    row
                    for row in df.index.values
                    if (str(row[1]) in protein_aliases.keys() and
                        f'tm{protein_aliases[str(row[1])]}_obs' == obs)
                    or f'tm{row[1]}_obs' == obs
                ]

                if rows:
                    m = df.loc[rows, :].values[
                        :, spec['mean_anchor'][1] + x_index
                    ].sum()

                    s = np.sqrt(np.power(df.loc[rows, :].values[
                        :, spec['std_anchor'][1] + x_index
                    ], 2.0).sum(axis=0))

                    data_dict[obs] = np.log(m)
                    data_dict[obs + '_std'] = s/m
            for val in spec['x_axis']:
                data_dict[val] = x
            # read in data from experiment specification
            for val in [
                'time', 't_presim',
                'EGF_0', 'EGF_0_preeq', 'EGF_0_presim',
                'Vemurafenib_0', 'Vemurafenib_0_preeq', 'Vemurafenib_0_presim',
                'Cobimetinib_0', 'Cobimetinib_0_preeq', 'Cobimetinib_0_presim',
                'EGFR_crispr', 'EGFR_crispr_preeq', 'EGFR_crispr_presim'
            ]:
                if val not in spec['x_axis'] and val in spec:
                    data_dict[val] = spec[val]

            fill_data_dict(data, data_dict)

            # append datapoint to dataframe
            data.loc[len(data)] = data_dict


def read_combo_experiment(data, df, egf, egfr_crispr, t):
    for iV in range(len(df)-1):
        rafi_conc = float(df.values[1 + iV, 0])
        for iC in range(len(df.columns.values)-1):
            meki_conc = float(df.values[0, 1 + iC])

            data_dict = {
                'time': t,
                't_presim': 0.0,
                'EGFR_crispr': egfr_crispr,
                'EGFR_crispr_preeq': egfr_crispr,
                'EGFR_crispr_presim': egfr_crispr,
                'EGF_0': egf,
                'EGF_0_preeq': 0,
                'EGF_0_presim': 0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'Vemurafenib_0': rafi_conc,
                'Vemurafenib_0_preeq': rafi_conc,
                'Vemurafenib_0_presim': rafi_conc,
                'Cobimetinib_0': meki_conc,
                'Cobimetinib_0_preeq': meki_conc,
                'Cobimetinib_0_presim': meki_conc,
                'pERK_IF_obs': float(df.values[1 + iV, 1 + iC]),
                'pERK_IF_obs_std': pERK_IF_std,
            }

            fill_data_dict(data, data_dict)

            # append datapoint to dataframe
            data.loc[len(data)] = data_dict


def load_experiment(IDs):
    cols = []
    for drug in RAFI + PANRAFI + MEKI:
        cols.extend([f'{drug}_0', f'{drug}_0_preeq', f'{drug}_0_presim'])
    cols.extend([
        'time', 't_presim',
        'EGF_0', 'EGF_0_preeq', 'EGF_0_presim',
        'EGFR_crispr',    'EGFR_crispr_preeq', 'EGFR_crispr_presim',
        'NRAS_Q61mut',    'NRAS_Q61mut_preeq', 'NRAS_Q61mut_presim',
        'pERK_IF_obs',    'pERK_IF_obs_std',
        'pMEK_IF_obs',    'pMEK_IF_obs_std',
        'tCRAF_obs',      'tCRAF_obs_std',
        'tBRAF_obs',      'tBRAF_obs_std',
        'tDUSP_obs',      'tDUSP_obs_std',
        'tEGFR_obs',      'tEGFR_obs_std',
        'tmDUSP_obs',     'tmDUSP_obs_std',
        'tmEGFR_obs',     'tmEGFR_obs_std',
        'tGRB2_obs',      'tGRB2_obs_std',
        'tMEK_obs',       'tMEK_obs_std',
        'tERK_obs',       'tERK_obs_std',
        'tRAS_obs',       'tRAS_obs_std',
        'tSOS1_obs',      'tSOS1_obs_std',
        'tSPRY_obs',      'tSPRY_obs_std',
        'tmSPRY_obs',     'tmSPRY_obs_std',
        'tCBL_obs',       'tCBL_obs_std',
        'pERK_obs',       'pERK_obs_std',
        'pS1134SOS1_obs', 'pS1134SOS1_obs_std',
    ])
    data = pd.DataFrame(columns=cols)

    for ID in IDs:
        if ID.endswith('.xls'):
            filename = os.path.join(DATAFOLDER, f'{ID}')
        else:
            filename = os.path.join(DATAFOLDER, f'{ID}.csv')

        if ID == 'D1':
            df = pd.read_csv(filename,
                             )
            spec = {
                't_presim': 0.0,
                'EGF_0': 100.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'Vemurafenib_0': 1.0,
                'Vemurafenib_0_preeq': 1.0,
                'Vemurafenib_0_presim': 1.0,
                'Cobimetinib_0': 0.0,
                'Cobimetinib_0_preeq': 0.0,
                'Cobimetinib_0_presim': 0.0,
                'pERK_mean_anchor': [1, 1],
                #'pERK_std_anchor': [1, 13],
                'pERK_IF_obs_std': pERK_IF_std,
                #'pMEK_mean_anchor': [10, 1],
                #'pMEK_std_anchor': [10, 13],
                #'pMEK_IF_obs_std': pMEK_IF_std,
                'x_axis': 'time',
                'x_points': 10,
                'y_axis': 'EGF_0',
                'y_points':  6,
            }
            read_IF_experiment(data, df, [spec])

        if ID == 'D4':
            df = pd.read_csv(filename)

            df_MEK = df.loc[df.pMEK == 1]
            df_ERK = df.loc[df.pERK == 1]
            n_conc = 10

            conc_anchor = list(df.columns).index('Concentration (uM)')
            mean_anchor = list(df.columns).index('Mean')
            std_anchor = list(df.columns).index('Std')

            for row in range(len(df_ERK.index)):
                for conc_idx in range(n_conc):
                    drug_conc = df_ERK.values[row, conc_anchor + conc_idx]
                    data_dict = {
                        'time': df.Time_EGF.values[0],
                        't_presim': 0.0,
                        'EGFR_crispr': 1.0,
                        'EGFR_crispr_preeq': 1.0,
                        'EGFR_crispr_presim': 1.0,
                        'NRAS_Q61mut': 0,
                        'NRAS_Q61mut_preeq': 0,
                        'NRAS_Q61mut_presim': 0,
                        'EGF_0': df_ERK.EGF.values[row],
                        'EGF_0_preeq': 0.0,
                        'EGF_0_presim': 0.0,
                        'pERK_IF_obs': df_ERK.values[row, mean_anchor +
                                                     conc_idx],
                        'pERK_IF_obs_std': pERK_IF_std,
                    }
                    for drug in RAFI + PANRAFI + MEKI:
                        if df_ERK[drug].values[row] == -1:
                            conc = drug_conc
                        else:
                            conc = float(df_ERK[drug].values[row])
                        data_dict[f'{drug}_0'] = conc
                        data_dict[f'{drug}_0_preeq'] = conc
                        data_dict[f'{drug}_0_presim'] = conc

                    if len(df_MEK):
                        data_dict.update({
                            'pMEK_IF_obs': df_MEK.values[row, mean_anchor +
                                                         conc_idx],
                            'pMEK_IF_obs_std': pMEK_IF_std,
                        })

                    for key in data.columns.values:
                        if key not in data_dict.keys():
                            data_dict[key] = float('nan')

                    data.loc[len(data)] = data_dict

        if ID in ['D7', 'D8']:
            df = pd.read_csv(filename)
            if ID == 'D8':
                # don't consider dynamic dose response data from EGFRa since
                # we cannot explain response at low drug concentrations
                df = df.loc[df.Cell_Line == 'A375']

            n_conc = 10

            conc_anchor = list(df.columns).index('Concentration (uM)')
            mean_anchor = list(df.columns).index('Mean')
            std_anchor = list(df.columns).index('Std')

            if ID == 'D7':
                df_MEK = df.loc[df.pMEK == 1]

            df = df.loc[df.pERK == 1]

            for row in range(len(df.index)):
                for conc_idx in range(n_conc):
                    drug_conc = df.values[row, conc_anchor + conc_idx]
                    egfr_crispr = 1.0 \
                        if df.Cell_Line.values[row] == 'A375' \
                        else pow(2,  3.2)
                    data_dict = {
                        'time': df.Time_EGF.values[row],
                        't_presim': 0.0,
                        'EGF_0': df.EGF.values[row],
                        'EGF_0_preeq': 0.0,
                        'EGF_0_presim': 0.0,
                        'EGFR_crispr': egfr_crispr,
                        'EGFR_crispr_preeq': egfr_crispr,
                        'EGFR_crispr_presim': egfr_crispr,
                        'NRAS_Q61mut': 0,
                        'NRAS_Q61mut_preeq': 0,
                        'NRAS_Q61mut_presim': 0,
                        'pERK_IF_obs': df.values[row, mean_anchor +
                                                 conc_idx],
                        'pERK_IF_obs_std': pERK_IF_std,
                        'pMEK_IF_obs': df_MEK.values[row, mean_anchor +
                                                     conc_idx]
                            if ID == 'D7' else float('nan'),
                        'pMEK_IF_obs_std': pMEK_IF_std
                            if ID == 'D7' else float('nan'),
                        # 'pERK_IF_obs': max(df.values[row, mean_anchor +
                        #                              conc_idx], 0.01),
                        # 'pERK_IF_obs_std': max(df.values[row, std_anchor +
                        #                                  conc_idx], 0.01),
                    }

                    drugs = {
                        'D7': ['Vemurafenib', 'Dabrafenib',  'LY3009120',
                                 'PLX8394', 'Trametinib', 'Cobimetinib'],
                        'D8': ['Vemurafenib', 'Cobimetinib']
                    }

                    for drug in drugs[ID]:
                        if df[drug].values[row] == -1:
                            conc = drug_conc
                        else:
                            conc = float(df[drug].values[row])
                        data_dict[f'{drug}_0'] = conc
                        data_dict[f'{drug}_0_preeq'] = conc
                        data_dict[f'{drug}_0_presim'] = conc

                    fill_data_dict(data, data_dict)

                    data.loc[len(data)] = data_dict

        if ID == 'D6':
            df = pd.read_csv(filename, header=[2], index_col=[0])
            for ir, row in df.iterrows():
                for t in df.keys():
                    rafi_conc = 1.0
                    meki_conc = 0.0
                    egfr_crispr = pow(2, -5.3) if ir == 'A375_Cri_1/9' else \
                                  pow(2,  3.2) if ir == 'A375_Cra_F7_5/13' \
                                      else \
                                  pow(2, 2.0) if ir == 'A375_Cra_F7_6/14' \
                                      else \
                                  1.0
                    data_dict = {
                        'time': float(t),
                        't_presim': 0.0,
                        'EGF_0': 100,
                        'EGF_0_preeq': 0.0,
                        'EGF_0_presim': 0.0,
                        'EGFR_crispr': egfr_crispr,
                        'EGFR_crispr_preeq': egfr_crispr,
                        'EGFR_crispr_presim': egfr_crispr,
                        'NRAS_Q61mut': 0,
                        'NRAS_Q61mut_preeq': 0,
                        'NRAS_Q61mut_presim': 0,
                        'Vemurafenib_0': rafi_conc,
                        'Vemurafenib_0_preeq': rafi_conc,
                        'Vemurafenib_0_presim': rafi_conc,
                        'Cobimetinib_0': meki_conc,
                        'Cobimetinib_0_preeq': meki_conc,
                        'Cobimetinib_0_presim': meki_conc,
                        'pERK_IF_obs': row[t],
                        'pERK_IF_obs_std': pERK_IF_std,
                        'pMEK_IF_obs': float('nan'),
                        'pMEK_IF_obs_std': float('nan'),
                    }

                    fill_data_dict(data, data_dict)

                    data.loc[len(data)] = data_dict

        if ID == 'D5':
            df = pd.read_csv(filename,
                             header=[0],
                             index_col=[0]
                             )

            df = pd.concat([df.iloc[0:6, :-1], df.iloc[9:15]],
                           axis=1).transpose()

            for ir, row in df.iterrows():

                data_dict = {
                    'time': row['Time (h) after GF addition'],
                    't_presim': row['Time (h) of GF addition after '
                                    'Vemurafenib (1uM)'],
                    'EGF_0': 100.0,
                    'EGF_0_preeq': 0,
                    'EGF_0_presim': 0,
                    'EGFR_crispr': 1.0,
                    'EGFR_crispr_preeq': 1.0,
                    'EGFR_crispr_presim': 1.0,
                    'NRAS_Q61mut': 0,
                    'NRAS_Q61mut_preeq': 0,
                    'NRAS_Q61mut_presim': 0,
                    'Vemurafenib_0': 1.0,
                    'Vemurafenib_0_preeq': 0.0,
                    'Vemurafenib_0_presim': 1.0,
                    'pERK_IF_obs': row.EGF,
                    'pERK_IF_obs_std': pERK_IF_std,
                }

                fill_data_dict(data, data_dict)

                data.loc[len(data)] = data_dict

        if ID == 'DP1':
            df = pd.read_csv(filename,
                             header=[0, 1],
                             index_col=[0, 1, 2, 3, 4]
                             )
            spec = {
                'proteomics': True,
                'EGF_0': 0.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'time': 0.0,
                't_presim': 0.0,
                'mean_anchor': [1, 0],
                'std_anchor': [1, 6],
                'x_axis': ['Vemurafenib_0',
                           'Vemurafenib_0_preeq',
                           'Vemurafenib_0_presim'],
                'x_points':  5,
            }
            read_MS_experiment(data, df, [spec])

        if ID == 'DP2':
            df = pd.read_csv(filename,
                             header=[0, 1],
                             index_col=[0, 1, 2, 3, 4]
                             )
            spec = {
                'phosphoproteomics': True,
                'EGF_0': 0.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'time': 0.0,
                't_presim': 0.0,
                'mean_anchor': [1, 0],
                'std_anchor': [1, 6],
                'x_axis': ['Vemurafenib_0',
                           'Vemurafenib_0_preeq',
                           'Vemurafenib_0_presim'],
                'x_points':  5,
            }
            read_MS_experiment(data, df, [spec])

        if ID in DATAFILES_PREDICTION_COMBO:
            df = pd.read_csv(filename,
                             header=[0],
                             )
            if 'no_egf' in ID:
                egf = 0
            else:
                egf = 100

            if '_crispr' in ID:
                egfr_crispr = pow(2,  3.2)
                t = 8
            else:
                egfr_crispr = 1.0
                t = 0.0833

            read_combo_experiment(data, df, egf, egfr_crispr, t)

        if ID == 'DT1':
            df = pd.read_csv(filename,
                             header=[0, 1],
                             index_col=[0, 1, 2])
            spec = {
                'EGF_0': 0.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'time': 0.0,
                't_presim': 0.0,
                'mean_anchor': [1, 0],
                'std_anchor': [1, 9],
                'x_axis': ['Vemurafenib_0',
                           'Vemurafenib_0_preeq',
                           'Vemurafenib_0_presim'],
                'x_points': 8,
            }
            read_T_experiment(data, df, [spec])

        if ID == 'DT2':
            df = pd.read_csv(filename,
                             header=[0, 1],
                             index_col=[0, 1, 2])
            spec = {
                'EGF_0': 100.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'Vemurafenib_0': 1.0,
                'Vemurafenib_0_preeq': 1.0,
                'Vemurafenib_0_presim': 1.0,
                't_presim': 0.0,
                'mean_anchor': [1, 0],
                'std_anchor': [1, 8],
                'x_axis': ['time'],
                'x_points': 7,
            }
            read_T_experiment(data, df, [spec])

        if ID == 'DT3':
            df = pd.read_csv(filename,
                             header=[0, 1],
                             index_col=[0, 1, 2])
            spec = {
                'EGF_0': 100.0,
                'EGF_0_preeq': 0.0,
                'EGF_0_presim': 0.0,
                'EGFR_crispr': 1.0,
                'EGFR_crispr_preeq': 1.0,
                'EGFR_crispr_presim': 1.0,
                'NRAS_Q61mut': 0,
                'NRAS_Q61mut_preeq': 0,
                'NRAS_Q61mut_presim': 0,
                'Vemurafenib_0': 1.0,
                'Vemurafenib_0_preeq': 1.0,
                'Vemurafenib_0_presim': 1.0,
                'Cobimetinib_0': 1.0,
                'Cobimetinib_0_preeq': 1.0,
                'Cobimetinib_0_presim': 1.0,
                't_presim': 0.0,
                'mean_anchor': [1, 0],
                'std_anchor': [1, 8],
                'x_axis': ['time'],
                'x_points': 7,
            }
            read_T_experiment(data, df, [spec])

        if ID in [*DATAFILES_PANRAFCOMBO,
                  *DATAFILES_HT29,
                  '20200119_A375_NRASQ61K_matrix_pERK_DOX_RAFi_MEKi',
                  '20200130_A375_NRASQ61K_matrix_pERK_DOX_RAFifixed_panRAFi_MEKi.xls']:
            if ID in DATAFILES_PANRAFCOMBO:
                df = pd.read_excel(filename,
                                   sheet_name='pERK')
                df = df[
                    (df['Drug A'] == 'LY3009120') |
                    (df['Drug B'] == 'LY3009120')
                ]
            elif ID.endswith('.xls'):
                df = pd.read_excel(filename, sheet_name='pERK')
            else:
                df = pd.read_csv(filename)

            for ir, row in df.iterrows():
                egfr_crispr = 1.0 \
                    if row.Cell_line in ['A375', 'HT29',
                                         'A375_NRAS_Q61K_DOXind'] \
                    else pow(2,  3.2)
                drug_a = row['Drug A']
                drug_b = row['Drug B']

                data_dict = {
                    'time': row.get('Time EGF (h)', 0.0),
                    't_presim': 0.0,
                    'EGF_0': row.get('EGF (ng/mL)', 0.0),
                    'EGF_0_preeq': 0,
                    'EGF_0_presim': 0,
                    'EGFR_crispr': egfr_crispr,
                    'EGFR_crispr_preeq': egfr_crispr,
                    'EGFR_crispr_presim': egfr_crispr,
                    'NRAS_Q61mut': row.get('DOX (ng/mL)', 0)/1000,
                    'NRAS_Q61mut_preeq': row.get('DOX (ng/mL)', 0)/1000,
                    'NRAS_Q61mut_presim': row.get('DOX (ng/mL)', 0)/1000,
                    f'{drug_a}_0': row['Concentration A (uM)'],
                    f'{drug_a}_0_preeq': row['Concentration A (uM)'],
                    f'{drug_a}_0_presim': row['Concentration A (uM)'],
                    f'{drug_b}_0': row['Concentration B (uM)'],
                    f'{drug_b}_0_preeq': row['Concentration B (uM)'],
                    f'{drug_b}_0_presim': row['Concentration B (uM)'],
                    'pERK_IF_obs': row['pERK'],
                    'pERK_IF_obs_std': pERK_IF_std,
                }

                if ID == '20200130_A375_NRASQ61K_matrix_pERK_DOX_RAFifixed_panRAFi_MEKi.xls':
                    data_dict['Vemurafenib_0'] = 1.0
                    data_dict['Vemurafenib_0_preeq'] = 1.0
                    data_dict['Vemurafenib_0_presim'] = 1.0

                fill_data_dict(data, data_dict)

                data.loc[len(data)] = data_dict

        if ID == 'M1_Naive_pERK_DoseResponse':
            df = pd.read_csv(filename)
            df.loc[df.Drug == 'AZ628', 'Drug'] = 'AZ_628'
            df = df[df.Drug.apply(lambda x: x in RAFI + PANRAFI + MEKI + [
                'DMSO'])]

            for ir, row in df.iterrows():
                data_dict = {
                    'time': 0.0,
                    't_presim': 0.0,
                    'EGF_0': 0.0,
                    'EGF_0_preeq': 0.0,
                    'EGF_0_presim': 0.0,
                    'EGFR_crispr': 1.0,
                    'EGFR_crispr_preeq': 1.0,
                    'EGFR_crispr_presim': 1.0,
                    'NRAS_Q61mut': 1.0
                    if row.CellLine == 'M1(NRAS)' else 0.0,
                    'NRAS_Q61mut_preeq': 1.0
                    if row.CellLine == 'M1(NRAS)' else 0.0,
                    'NRAS_Q61mut_presim': 1.0
                    if row.CellLine == 'M1(NRAS)' else 0.0,
                    'pERK_IF_obs': row['pERK (Raw fluorescence)'],
                    'pERK_IF_obs_std': pERK_IF_std,
                }
                if row.Drug != 'DMSO':
                    for suffix in ['', '_preeq', '_presim']:
                        data_dict[f'{row.Drug}_0{suffix}'] = \
                            row['Concentration (µM)']

                fill_data_dict(data, data_dict)

                data.loc[len(data)] = data_dict

        if ID == '20200117_A375_NRASQ61K_unidimensional_pERK_DOX_RAFi_MEKi':
            df = pd.read_csv(filename)

            df_MEK = df.loc[df.pMEK == 1]
            df_ERK = df.loc[df.pERK == 1]
            n_conc = 9

            conc_anchor = list(df.columns).index('Concentration (uM)')
            mean_anchor = list(df.columns).index('Mean')
            std_anchor = list(df.columns).index('Std')

            for row in range(len(df_ERK.index)):
                for conc_idx in range(n_conc):
                    drug_conc = df_ERK.values[row, conc_anchor + conc_idx]
                    nras_mut = float(int(df_ERK.loc[row, 'DOX'] == 1000))
                    data_dict = {
                        'time': 0.0,
                        't_presim': 0.0,
                        'EGFR_crispr': 1.0,
                        'EGFR_crispr_preeq': 1.0,
                        'EGFR_crispr_presim': 1.0,
                        'NRAS_Q61mut': nras_mut,
                        'NRAS_Q61mut_preeq': nras_mut,
                        'NRAS_Q61mut_presim': nras_mut,
                        'EGF_0': 0.0,
                        'EGF_0_preeq': 0.0,
                        'EGF_0_presim': 0.0,
                        'pERK_IF_obs': df_ERK.values[row, mean_anchor +
                                                     conc_idx],
                        'pERK_IF_obs_std': pERK_IF_std,
                    }
                    for drug in RAFI + PANRAFI + MEKI:
                        if drug in ['Vemurafenib', 'Cobimetinib']:
                            if df_ERK[drug].values[row] == -1:
                                conc = drug_conc
                            else:
                                conc = float(df_ERK[drug].values[row])
                        else:
                            conc = 0.0
                        data_dict[f'{drug}_0'] = conc
                        data_dict[f'{drug}_0_preeq'] = conc
                        data_dict[f'{drug}_0_presim'] = conc

                    if len(df_MEK):
                        data_dict.update({
                            'pMEK_IF_obs': df_MEK.values[row, mean_anchor +
                                                         conc_idx],
                            'pMEK_IF_obs_std': pMEK_IF_std,
                        })

                    for key in data.columns.values:
                        if key not in data_dict.keys():
                            data_dict[key] = float('nan')

                    data.loc[len(data)] = data_dict

    if not len(data):
        raise Exception("No data available for specified experiment")

    return data


def filter_experiments(data, instances):
    instances_lower = [instance.lower() for instance in instances]
    if 'rafi' not in instances_lower:
        data = data[(data[[f'{drug}_0'
                           for drug in RAFI + PANRAFI]] == 0).all(axis=1)]
    if 'meki' not in instances_lower:
        data = data[(data[[f'{drug}_0' for drug in MEKI]] == 0.0).all(axis=1)]
    if 'egf' not in instances_lower:
        data = data[data.EGF_0 == 0.0]
    if 'egfr' not in instances_lower:
        data = data[data.EGFR_crispr == 1.0]

    subset = (
        (data[[f'{drug}_0' for drug in MEKI + RAFI + PANRAFI]] > 0.0).sum(
            axis=1
        ) <= 1
    ) & (
         ((data[[f'{drug}_0' for drug in PANRAFI]] > 0.0).any(axis=1)
          & (data['EGF_0'] == 0.0))
         |
         (data[[f'{drug}_0' for drug in PANRAFI]] == 0.0).all(axis=1)
    )

    if 'singleprediction' not in instances_lower \
            and 'comboprediction' not in instances_lower \
            and 'panrafcomboprediction' not in instances_lower \
            and 'engineered' not in instances_lower \
            and 'ht29' not in instances_lower:
        data = data[subset]

    if 'singleprediction' in instances_lower:
        data = data[subset.apply(lambda x: not x) | (data.EGFR_crispr != 1.0)]

    if 'engineered' in instances_lower and 'mutrastraining' in instances_lower:
        ras_subset = (
                (data.NRAS_Q61mut == 1.0) &
                ((data[f'Cobimetinib_0'] > 0.0001) +
                 (data[f'Vemurafenib_0'] > 0.001) == 0)
        )
    else:
        ras_subset = (
            (data.NRAS_Q61mut == 0.0) |
            ((data[[f'{drug}_0' for drug in MEKI + RAFI + PANRAFI]] > 0.0).sum(
                    axis=1
                ) == 0)
        )

    if 'mutrastraining' in instances_lower:
        data = data[ras_subset]

    return data


if __name__ == '__main__':
    instances = sorted(sys.argv[1].split('_'))
    outfile = f'processed_{"_".join(instances)}.csv'
    if 'comboprediction' in instances:
        data = load_experiment(DATAFILES_PREDICTION_COMBO)
    elif 'singleprediction' in instances:
        data = load_experiment(DATAFILES_PREDICTION_SINGLE)
    elif 'panrafcomboprediction' in instances:
        data = load_experiment(DATAFILES_PANRAFCOMBO)
    elif 'ht29' in instances:
        data = load_experiment(DATAFILES_HT29)
    elif 'mutrastraining' in instances or 'mutrasprediction' in instances:
        if 'engineered' in instances:
            data = load_experiment(DATAFILES_MUTRAS_ENGINEERED)
    elif 'mutrascomboprediction' in instances:
        data = load_experiment(DATAFILES_MUTRAS_ENGINEERED_COMBO)
    else:
        data = load_experiment(DATAFILES_ESTIMATION)

    data = filter_experiments(data, instances)
    data.loc[data.time == 0.083, 'time'] = 0.0833
    data.to_csv(os.path.join(DATAFOLDER, outfile))
