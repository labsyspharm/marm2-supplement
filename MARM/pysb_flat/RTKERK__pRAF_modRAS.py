# exported from PySB model 'RTKERK__base'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, EnergyPattern, ANY, WILD, as_complex_pattern
from sympy import exp, log

Model()

Monomer('BRAF', ['AA600', 'RBD', 'mek', 'raf', 'rafi', 'T753'], {'AA600': ['E'], 'T753': ['u', 'p']})
Monomer('CRAF', ['RBD', 'mek', 'raf', 'rafi', 'S642'], {'S642': ['u', 'p']})
Monomer('RAS', ['raf', 'sos1', 'state'], {'state': ['gdp', 'gtp']})
Monomer('RAFi', ['raf'])
Monomer('PRAFi', ['raf'])
Monomer('MEK', ['Dsite', 'meki', 'phospho', 'raf'], {'phospho': ['p', 'u']})
Monomer('MEKi', ['mek'])
Monomer('ERK', ['CD', 'phospho'], {'phospho': ['p', 'u']})
Monomer('DUSP', ['erk'])
Monomer('mDUSP')
Monomer('EGF', ['rtk'])
Monomer('EGFR', ['KD', 'Tyr', 'rtkf', 'ub'], {'Tyr': ['p', 'u'], 'ub': ['false', 'true']})
Monomer('mEGFR')
Monomer('GRB2', ['SH2', 'SH3'])
Monomer('SPRY', ['SH3m'])
Monomer('mSPRY')
Monomer('SOS1', ['S1134', 'SH3m', 'ras'], {'S1134': ['p', 'u']})
Monomer('CBL', ['SH3m'])

Parameter('ep_RAF_RAF_mod_RASgtp_double_ddG', 1000.0)
Parameter('ep_RAF_RAS_mod_pRAF_ddG', 1000.0)
Parameter('ep_RAF_RAF_mod_pRAF_ddG', 1000.0)
Parameter('bind_RASgtp_RAF_kf', 10.0)
Parameter('bind_RASgtp_RAF_dG', 0.01)
Parameter('bind_RASgtp_RAF_phi', 1.0)
Parameter('bind_RAF_RAF_kf', 10.0)
Parameter('bind_RAF_RAF_dG', 0.01)
Parameter('bind_RAF_RAF_phi', 1.0)
Parameter('bind_RAFi_RAF_kf', 10.0)
Parameter('bind_RAFi_RAF_dG', 0.01)
Parameter('bind_RAFi_RAF_phi', 1.0)
Parameter('bind_PRAFi_RAF_kf', 10.0)
Parameter('bind_PRAFi_RAF_dG', 0.01)
Parameter('bind_PRAFi_RAF_phi', 1.0)
Parameter('ep_RAF_RAF_mod_RAFi_single_ddG', 0.001)
Parameter('ep_RAF_RAF_mod_RAFi_double_ddG', 1000.0)
Parameter('bind_RAF_uMEK_kf', 10.0)
Parameter('bind_RAF_uMEK_dG', 0.01)
Parameter('bind_RAF_uMEK_phi', 1.0)
Parameter('catalyze_RAF_RAFrafiNone_MEK_p_kcat', 10.0)
Parameter('catalyze_RAFrafiNone_MEKmekiNone_p_kcat', 10.0)
Parameter('catalyze_RAFrafiNone_MEKmeki_MEKi_p_kcatr', 0.0)
Parameter('bind_MEKi_MEK_kf', 10.0)
Parameter('bind_MEKi_MEK_dG', 0.01)
Parameter('bind_MEKi_MEK_phi', 1.0)
Parameter('ep_pMEK_MEKi_ddG', 100.0)
Parameter('bind_MEK_uERK_kf', 10.0)
Parameter('bind_MEK_uERK_dG', 0.01)
Parameter('bind_MEK_uERK_phi', 1.0)
Parameter('catalyze_MEKmekiNone_phosphop_ERK_p_kcat', 10.0)
Parameter('bind_DUSP_pERK_kf', 10.0)
Parameter('bind_DUSP_pERK_dG', 0.01)
Parameter('bind_DUSP_pERK_phi', 1.0)
Parameter('mDUSP_eq', 10000.0)
Parameter('DUSP_eq', 10000.0)
Parameter('synthesize_pERK_DUSP_ERK_gexpslope', 1000.0)
Parameter('msynthesize_pERK_DUSP_kdeg', 10.0)
Parameter('psynthesize_pERK_DUSP_kdeg', 10.0)
Parameter('synthesize_pERK_DUSP_ERK_kM', 0.0)
Parameter('bind_EGF_EGFR_kf', 10.0)
Parameter('bind_EGF_EGFR_dG', 0.01)
Parameter('bind_EGF_EGFR_phi', 1.0)
Parameter('bind_EGFR_EGFR_kf', 10.0)
Parameter('bind_EGFR_EGFR_dG', 100.0)
Parameter('bind_EGFR_EGFR_phi', 1.0)
Parameter('ep_EGFR_EGFR_mod_EGF_single_ddG', 100.0)
Parameter('catalyze_EGFR_EGFR_p_kcat', 10.0)
Parameter('catalyze_EGFR_u_kcatr', 10.0)
Parameter('catalyze_EGFR_endo_kcat', 10.0)
Parameter('catalyze_EGFR_deg_kM', 10.0)
Parameter('catalyze_EGFR_recycl_kcat', 10.0)
Parameter('catalyze_EGFR_deg_kcat', 10.0)
Parameter('mEGFR_eq', 10000.0)
Parameter('EGFR_eq', 10000.0)
Parameter('synthesize_pERK_EGFR_ERK_gexpslope', 1000.0)
Parameter('msynthesize_pERK_EGFR_kdeg', 10.0)
Parameter('psynthesize_pERK_EGFR_kdeg', 10.0)
Parameter('EGFR_crispr', 1.0)
Parameter('synthesize_pERK_EGFR_ERK_kM', 0.0)
Parameter('bind_pEGFR_GRB2_kf', 10.0)
Parameter('bind_pEGFR_GRB2_dG', 100.0)
Parameter('bind_pEGFR_GRB2_phi', 1.0)
Parameter('bind_SPRY_GRB2_kf', 10.0)
Parameter('bind_SPRY_GRB2_dG', 0.01)
Parameter('bind_SPRY_GRB2_phi', 1.0)
Parameter('mSPRY_eq', 10000.0)
Parameter('SPRY_eq', 10000.0)
Parameter('synthesize_pERK_SPRY_ERK_gexpslope', 1000.0)
Parameter('msynthesize_pERK_SPRY_kdeg', 10.0)
Parameter('psynthesize_pERK_SPRY_kdeg', 10.0)
Parameter('synthesize_pERK_SPRY_ERK_kM', 0.0)
Parameter('bind_GRB2_SOS1_kf', 10.0)
Parameter('bind_GRB2_SOS1_dG', 0.01)
Parameter('bind_GRB2_SOS1_phi', 1.0)
Parameter('catalyze_pERK_SOS1_pS1134_kbase', 0.0)
Parameter('catalyze_pERK_SOS1_pS1134_kcat', 10.0)
Parameter('catalyze_pERK_CRAF_pS642_kcat', 10.0)
Parameter('catalyze_pERK_BRAF_pT753_kcat', 10.0)
Parameter('ep_SOS1S1134p_GRB2_ddG', 100.0)
Parameter('bind_SOS1_RAS_kf', 10.0)
Parameter('bind_SOS1_RAS_dG', 0.01)
Parameter('catalyze_SOS1_RAS_gtp_kcat', 0.01)
Parameter('bind_CBL_GRB2_kf', 10.0)
Parameter('bind_CBL_GRB2_dG', 0.01)
Parameter('bind_CBL_GRB2_phi', 1.0)
Parameter('catalyze_CBL_EGFR_ub_kcat', 10.0)
Parameter('catalyze_EGFR_dub_kcatr', 10.0)
Parameter('catalyze_PP2A_MEK_u_kcatr', 1.0)
Parameter('catalyze_DUSP_ERK_u_kcatr', 1.0)
Parameter('catalyze_phosphatase_SOS1_uS1134_kcatr', 1.0)
Parameter('catalyze_phosphatase_CRAF_uS642_kcatr', 1.0)
Parameter('catalyze_phosphatase_BRAF_uT753_kcatr', 1.0)
Parameter('catalyze_NF1_RAS_gdp_kcatr', 100.0)
Parameter('BRAF_0', 0.0)
Parameter('CRAF_0', 0.0)
Parameter('RAS_0', 0.0)
Parameter('RAFi_0', 0.0)
Parameter('PRAFi_0', 0.0)
Parameter('MEK_0', 0.0)
Parameter('MEKi_0', 0.0)
Parameter('ERK_0', 0.0)
Parameter('EGF_0', 0.0)
Parameter('GRB2_0', 0.0)
Parameter('SOS1_0', 0.0)
Parameter('CBL_0', 0.0)
Parameter('pMEK_IF_scale', 1.0)
Parameter('pMEK_IF_offset', 0.1)
Parameter('pERK_IF_scale', 1.0)
Parameter('pERK_IF_offset', 0.1)

Parameter('N_Avogadro', 6.02214076000000e+23)
Parameter('volume', 1.00000000000000e-12)
Parameter('m_Da_EGF', 6200.00000000000)

Expression('Ea0_bind_RASgtp_BRAF', -bind_RASgtp_RAF_phi*bind_RASgtp_RAF_dG - log(bind_RASgtp_RAF_kf))
Expression('Ea0_bind_RASgtp_CRAF', -bind_RASgtp_RAF_phi*bind_RASgtp_RAF_dG - log(bind_RASgtp_RAF_kf))
Expression('Ea0_bind_RAF_RAF', -bind_RAF_RAF_phi*bind_RAF_RAF_dG - log(bind_RAF_RAF_kf))
Expression('Ea0_bind_RAFi_RAF', -bind_RAFi_RAF_phi*bind_RAFi_RAF_dG - log(bind_RAFi_RAF_kf))
Expression('Ea0_bind_PRAFi_RAF', -bind_PRAFi_RAF_phi*bind_PRAFi_RAF_dG - log(bind_PRAFi_RAF_kf))
Expression('ep_RAF_RAF_mod_RAFi_double_Gf', ep_RAF_RAF_mod_RAFi_double_ddG - ep_RAF_RAF_mod_RAFi_single_ddG)
Expression('Ea0_bind_BRAF_uMEK', -bind_RAF_uMEK_phi*bind_RAF_uMEK_dG - log(bind_RAF_uMEK_kf))
Expression('Ea0_bind_CRAF_uMEK', -bind_RAF_uMEK_phi*bind_RAF_uMEK_dG - log(bind_RAF_uMEK_kf))
Expression('catalyze_RAFrafiNone_MEKmekiANY_p_kcat', catalyze_RAFrafiNone_MEKmekiNone_p_kcat*catalyze_RAFrafiNone_MEKmeki_MEKi_p_kcatr)
Expression('Ea0_bind_MEKi_MEK', -bind_MEKi_MEK_phi*bind_MEKi_MEK_dG - log(bind_MEKi_MEK_kf))
Expression('Ea0_bind_MEK_uERK', -bind_MEK_uERK_phi*bind_MEK_uERK_dG - log(bind_MEK_uERK_kf))
Expression('Ea0_bind_DUSP_pERK', -bind_DUSP_pERK_phi*bind_DUSP_pERK_dG - log(bind_DUSP_pERK_kf))
Expression('msynthesize_pERK_DUSP_kbase', mDUSP_eq*psynthesize_pERK_DUSP_kdeg)
Expression('psynthesize_pERK_DUSP_kbase', 1000000.0*DUSP_eq*psynthesize_pERK_DUSP_kdeg/(N_Avogadro*volume*mDUSP_eq))
Expression('psynthesize_pERK_DUSP_ksyn', psynthesize_pERK_DUSP_kbase)
Expression('Ea0_bind_EGF_EGFR', -bind_EGF_EGFR_phi*bind_EGF_EGFR_dG - log(bind_EGF_EGFR_kf))
Expression('Ea0_bind_EGFR_EGFR', -bind_EGFR_EGFR_phi*bind_EGFR_EGFR_dG - log(bind_EGFR_EGFR_kf))
Expression('catalyze_EGFR_u_kcat', catalyze_EGFR_EGFR_p_kcat*catalyze_EGFR_u_kcatr)
Expression('msynthesize_pERK_EGFR_kbase', mEGFR_eq*psynthesize_pERK_EGFR_kdeg)
Expression('psynthesize_pERK_EGFR_kbase', 1000000.0*EGFR_eq*psynthesize_pERK_EGFR_kdeg/(N_Avogadro*volume*mEGFR_eq))
Expression('psynthesize_pERK_EGFR_ksyn', psynthesize_pERK_EGFR_kbase)
Expression('Ea0_bind_pEGFR_GRB2', -bind_pEGFR_GRB2_phi*bind_pEGFR_GRB2_dG - log(bind_pEGFR_GRB2_kf))
Expression('Ea0_bind_SPRY_GRB2', -bind_SPRY_GRB2_phi*bind_SPRY_GRB2_dG - log(bind_SPRY_GRB2_kf))
Expression('msynthesize_pERK_SPRY_kbase', mSPRY_eq*psynthesize_pERK_SPRY_kdeg)
Expression('psynthesize_pERK_SPRY_kbase', 1000000.0*SPRY_eq*psynthesize_pERK_SPRY_kdeg/(N_Avogadro*volume*mSPRY_eq))
Expression('psynthesize_pERK_SPRY_ksyn', psynthesize_pERK_SPRY_kbase)
Expression('Ea0_bind_GRB2_SOS1', -bind_GRB2_SOS1_phi*bind_GRB2_SOS1_dG - log(bind_GRB2_SOS1_kf))
Expression('bind_SOS1_RAS_kr', exp(bind_SOS1_RAS_dG)*bind_SOS1_RAS_kf)
Expression('Ea0_bind_CBL_GRB2', -bind_CBL_GRB2_phi*bind_CBL_GRB2_dG - log(bind_CBL_GRB2_kf))
Expression('catalyze_EGFR_dub_kcat', catalyze_CBL_EGFR_ub_kcat*catalyze_EGFR_dub_kcatr)
Expression('catalyze_PP2A_MEK_u_kcat', catalyze_PP2A_MEK_u_kcatr*catalyze_RAF_RAFrafiNone_MEK_p_kcat)
Expression('catalyze_DUSP_ERK_u_kcat', catalyze_DUSP_ERK_u_kcatr*catalyze_MEKmekiNone_phosphop_ERK_p_kcat)
Expression('catalyze_phosphatase_CRAF_uS642_kcat', catalyze_pERK_CRAF_pS642_kcat*catalyze_phosphatase_CRAF_uS642_kcatr)
Expression('catalyze_phosphatase_BRAF_uT753_kcat', catalyze_pERK_BRAF_pT753_kcat*catalyze_phosphatase_BRAF_uT753_kcatr)
Expression('catalyze_phosphatase_SOS1_uS1134_kcat', catalyze_pERK_SOS1_pS1134_kcat*catalyze_phosphatase_SOS1_uS1134_kcatr)
Expression('catalyze_NF1_RAS_gdp_kcat', catalyze_NF1_RAS_gdp_kcatr*catalyze_SOS1_RAS_gtp_kcat)
Expression('initBRAF', 1000000.0*BRAF_0/(N_Avogadro*volume))
Expression('initCRAF', 1000000.0*CRAF_0/(N_Avogadro*volume))
Expression('initRAS', 1000000.0*RAS_0/(N_Avogadro*volume))
Expression('initMEK', 1000000.0*MEK_0/(N_Avogadro*volume))
Expression('initERK', 1000000.0*ERK_0/(N_Avogadro*volume))
Expression('initEGF', 6.02214076208112e+23*EGF_0/(N_Avogadro*m_Da_EGF))
Expression('initGRB2', 1000000.0*GRB2_0/(N_Avogadro*volume))
Expression('initSOS1', 1000000.0*SOS1_0/(N_Avogadro*volume))
Expression('initCBL', 1000000.0*CBL_0/(N_Avogadro*volume))

Compartment(name='EC', parent=None, dimension=3, size=None)
Compartment(name='PM', parent=EC, dimension=2, size=None)
Compartment(name='CP', parent=PM, dimension=3, size=None)
Compartment(name='EM', parent=CP, dimension=2, size=None)

Observable('modulation_synthesize_pERK_DUSP_ERK', ERK(phospho='p'))
Observable('ubEGFR', EGFR(Tyr='p') ** PM)
Observable('modulation_synthesize_pERK_EGFR_ERK', ERK(phospho='p'))
Observable('modulation_synthesize_pERK_SPRY_ERK', ERK(phospho='p'))
Observable('tBRAF', BRAF())
Observable('tBRAF600E', BRAF(AA600='E'))
Observable('tCRAF', CRAF())
Observable('tRAS', RAS())
Observable('gdpRAS', RAS(state='gdp'))
Observable('gtpRAS', RAS(state='gtp'))
Observable('tRAFi', RAFi())
Observable('tPRAFi', PRAFi())
Observable('tMEK', MEK())
Observable('pMEK', MEK(phospho='p'))
Observable('tMEKi', MEKi())
Observable('tERK', ERK())
Observable('pERK', ERK(phospho='p'))
Observable('tDUSP', DUSP())
Observable('tmDUSP', mDUSP())
Observable('tEGF', EGF())
Observable('tEGFR', EGFR())
Observable('pEGFR', EGFR(Tyr='p'))
Observable('tmEGFR', mEGFR())
Observable('tGRB2', GRB2())
Observable('tSPRY', SPRY())
Observable('tmSPRY', mSPRY())
Observable('tSOS1', SOS1())
Observable('pS1134SOS1', SOS1(S1134='p'))
Observable('uS1134SOS1', SOS1(S1134='u'))
Observable('tCBL', CBL())

Expression('synthesize_pERK_DUSP_ERK_kmod', modulation_synthesize_pERK_DUSP_ERK*synthesize_pERK_DUSP_ERK_gexpslope/(modulation_synthesize_pERK_DUSP_ERK + synthesize_pERK_DUSP_ERK_kM) + 1)
Expression('msynthesize_pERK_DUSP_ksyn', 1.0*msynthesize_pERK_DUSP_kbase*synthesize_pERK_DUSP_ERK_kmod)
Expression('catalyze_EGFR_kendo', catalyze_EGFR_endo_kcat/(ubEGFR + catalyze_EGFR_deg_kM))
Expression('synthesize_pERK_EGFR_ERK_kmod', modulation_synthesize_pERK_EGFR_ERK*synthesize_pERK_EGFR_ERK_gexpslope/(modulation_synthesize_pERK_EGFR_ERK + synthesize_pERK_EGFR_ERK_kM) + 1)
Expression('msynthesize_pERK_EGFR_ksyn', msynthesize_pERK_EGFR_kbase*synthesize_pERK_EGFR_ERK_kmod*EGFR_crispr)
Expression('synthesize_pERK_SPRY_ERK_kmod', modulation_synthesize_pERK_SPRY_ERK*synthesize_pERK_SPRY_ERK_gexpslope/(modulation_synthesize_pERK_SPRY_ERK + synthesize_pERK_SPRY_ERK_kM) + 1)
Expression('msynthesize_pERK_SPRY_ksyn', 1.0*msynthesize_pERK_SPRY_kbase*synthesize_pERK_SPRY_ERK_kmod)
Expression('tBRAF_obs', log(1.0e-6*N_Avogadro*volume*tBRAF))
Expression('tCRAF_obs', log(1.0e-6*N_Avogadro*volume*tCRAF))
Expression('tRAS_obs', log(1.0e-6*N_Avogadro*volume*tRAS))
Expression('tMEK_obs', log(1.0e-6*N_Avogadro*volume*tMEK))
Expression('pMEK_obs', pMEK/tMEK)
Expression('pMEK_IF_obs', pMEK_obs*pMEK_IF_scale + pMEK_IF_offset)
Expression('tERK_obs', log(1.0e-6*N_Avogadro*volume*tERK))
Expression('pERK_obs', pERK/tERK)
Expression('pERK_IF_obs', pERK_obs*pERK_IF_scale + pERK_IF_offset)
Expression('tDUSP_obs', log(1.0e-6*N_Avogadro*volume*tDUSP))
Expression('tmDUSP_obs', log(tmDUSP))
Expression('tEGF_obs', log(1.0e-6*N_Avogadro*volume*tEGF))
Expression('tEGFR_obs', log(1.0e-6*N_Avogadro*volume*tEGFR))
Expression('pEGFR_obs', pEGFR/tEGFR)
Expression('tmEGFR_obs', log(tmEGFR))
Expression('tGRB2_obs', log(1.0e-6*N_Avogadro*volume*tGRB2))
Expression('tSPRY_obs', log(1.0e-6*N_Avogadro*volume*tSPRY))
Expression('tmSPRY_obs', log(tmSPRY))
Expression('tSOS1_obs', log(1.0e-6*N_Avogadro*volume*tSOS1))
Expression('pS1134SOS1_obs', pS1134SOS1/tSOS1)
Expression('tCBL_obs', log(1.0e-6*N_Avogadro*volume*tCBL))

Rule('RASgtp_and_BRAF_bind_and_dissociate', RAS(raf=None, state='gtp') + BRAF(RBD=None, raf=None) | RAS(raf=1, state='gtp') % BRAF(RBD=1, raf=None), bind_RASgtp_RAF_phi, Ea0_bind_RASgtp_BRAF, energy=True)
Rule('RASgtp_and_CRAF_bind_and_dissociate', RAS(raf=None, state='gtp') + CRAF(RBD=None, raf=None) | RAS(raf=1, state='gtp') % CRAF(RBD=1, raf=None), bind_RASgtp_RAF_phi, Ea0_bind_RASgtp_CRAF, energy=True)
Rule('BRAF_and_BRAF_bind_and_dissociate', BRAF(raf=None, RBD=ANY) + BRAF(raf=None, RBD=ANY) | BRAF(raf=1, RBD=ANY) % BRAF(raf=1, RBD=ANY), bind_RAF_RAF_phi, Ea0_bind_RAF_RAF, energy=True)
Rule('BRAF_and_CRAF_bind_and_dissociate', BRAF(raf=None, RBD=ANY) + CRAF(raf=None, RBD=ANY) | BRAF(raf=1, RBD=ANY) % CRAF(raf=1, RBD=ANY), bind_RAF_RAF_phi, Ea0_bind_RAF_RAF, energy=True)
Rule('CRAF_and_CRAF_bind_and_dissociate', CRAF(raf=None, RBD=ANY) + CRAF(raf=None, RBD=ANY) | CRAF(raf=1, RBD=ANY) % CRAF(raf=1, RBD=ANY), bind_RAF_RAF_phi, Ea0_bind_RAF_RAF, energy=True)
Rule('RAFi_and_BRAF_bind_and_dissociate', RAFi(raf=None) + BRAF(rafi=None) | RAFi(raf=1) % BRAF(rafi=1), bind_RAFi_RAF_phi, Ea0_bind_RAFi_RAF, energy=True)
Rule('RAFi_and_CRAF_bind_and_dissociate', RAFi(raf=None) + CRAF(rafi=None) | RAFi(raf=1) % CRAF(rafi=1), bind_RAFi_RAF_phi, Ea0_bind_RAFi_RAF, energy=True)
Rule('PRAFi_and_BRAF_bind_and_dissociate', PRAFi(raf=None) + BRAF(rafi=None) | PRAFi(raf=1) % BRAF(rafi=1), bind_PRAFi_RAF_phi, Ea0_bind_PRAFi_RAF, energy=True)
Rule('PRAFi_and_CRAF_bind_and_dissociate', PRAFi(raf=None) + CRAF(rafi=None) | PRAFi(raf=1) % CRAF(rafi=1), bind_PRAFi_RAF_phi, Ea0_bind_PRAFi_RAF, energy=True)
Rule('BRAF_and_uMEK_bind_and_dissociate', BRAF(mek=None) + MEK(phospho='u', raf=None) | BRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_uMEK_phi, Ea0_bind_BRAF_uMEK, energy=True)
Rule('CRAF_and_uMEK_bind_and_dissociate', CRAF(mek=None) + MEK(phospho='u', raf=None) | CRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_uMEK_phi, Ea0_bind_CRAF_uMEK, energy=True)
Rule('BRAF_BRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % BRAF(RBD=ANY, mek=1, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + BRAF(RBD=ANY, mek=None, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
Rule('BRAF_CRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % BRAF(RBD=ANY, mek=1, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + BRAF(RBD=ANY, mek=None, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
Rule('CRAF_BRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % CRAF(RBD=ANY, mek=1, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + CRAF(RBD=ANY, mek=None, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
Rule('CRAF_CRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % CRAF(RBD=ANY, mek=1, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + CRAF(RBD=ANY, mek=None, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
Rule('BRAFV600E_phosphorylates_MEK_bound', MEK(meki=ANY, phospho='u', raf=1) % BRAF(AA600='E', mek=1, raf=None, rafi=None) >> MEK(meki=ANY, phospho='p', raf=None) + BRAF(AA600='E', mek=None, raf=None, rafi=None), catalyze_RAFrafiNone_MEKmekiANY_p_kcat)
Rule('BRAFV600E_phosphorylates_MEK_unbound', MEK(meki=None, phospho='u', raf=1) % BRAF(AA600='E', mek=1, raf=None, rafi=None) >> MEK(meki=None, phospho='p', raf=None) + BRAF(AA600='E', mek=None, raf=None, rafi=None), catalyze_RAFrafiNone_MEKmekiNone_p_kcat)
Rule('MEK_is_dephosphorylated', MEK(phospho='p') >> MEK(phospho='u'), catalyze_PP2A_MEK_u_kcat)
Rule('MEKi_and_MEK_bind_and_dissociate', MEKi(mek=None) + MEK(meki=None) | MEKi(mek=1) % MEK(meki=1), bind_MEKi_MEK_phi, Ea0_bind_MEKi_MEK, energy=True)
Rule('bind_MEK_uERK', MEK(Dsite=None) + ERK(CD=None, phospho='u') | MEK(Dsite=1) % ERK(CD=1, phospho='u'), bind_MEK_uERK_phi, Ea0_bind_MEK_uERK, energy=True)
Rule('pMEK_phosphorylates_ERK', ERK(CD=1, phospho='u') % MEK(Dsite=1, meki=None, phospho='p') >> ERK(CD=None, phospho='p') + MEK(Dsite=None, meki=None, phospho='p'), catalyze_MEKmekiNone_phosphop_ERK_p_kcat)
Rule('bind_DUSP_pERK', DUSP(erk=None) + ERK(CD=None, phospho='p') | DUSP(erk=1) % ERK(CD=1, phospho='p'), bind_DUSP_pERK_phi, Ea0_bind_DUSP_pERK, energy=True)
Rule('DUSP_dephosphorylates_ERK', ERK(CD=1, phospho='p') % DUSP(erk=1) >> ERK(CD=None, phospho='u') + DUSP(erk=None), catalyze_DUSP_ERK_u_kcat)
Rule('synthesis_mDUSP', None >> mDUSP() ** CP, msynthesize_pERK_DUSP_ksyn)
Rule('basal_degradation_mDUSP', mDUSP() >> None, msynthesize_pERK_DUSP_kdeg, delete_molecules=True)
Rule('synthesis_pDUSP', mDUSP() ** CP >> mDUSP() ** CP + DUSP(erk=None) ** CP, psynthesize_pERK_DUSP_ksyn)
Rule('basal_degradation_pDUSP', DUSP() >> None, psynthesize_pERK_DUSP_kdeg, delete_molecules=True)
Rule('EGF_and_EGFR_bind_and_dissociate', EGF(rtk=None) + EGFR(rtkf=None) | EGF(rtk=1) % EGFR(rtkf=1), bind_EGF_EGFR_phi, Ea0_bind_EGF_EGFR, energy=True)
Rule('bind_EGFR_EGFRu', EGFR(KD=None, Tyr='u') + EGFR(KD=None, Tyr='u') | EGFR(KD=1, Tyr='u') % EGFR(KD=None, Tyr=('u', 1)), bind_EGFR_EGFR_phi, Ea0_bind_EGFR_EGFR, energy=True)
Rule('bind_EGFR_EGFRp', EGFR(KD=None, Tyr='p') + EGFR(KD=None, Tyr='u') | EGFR(KD=1, Tyr='p') % EGFR(KD=None, Tyr=('u', 1)), bind_EGFR_EGFR_phi, Ea0_bind_EGFR_EGFR, energy=True)
Rule('catalyze_EGFR_EGFR_p', EGFR(KD=1, rtkf=ANY) % EGFR(Tyr=('u', 1)) >> EGFR(KD=None, rtkf=ANY) + EGFR(Tyr='p'), catalyze_EGFR_EGFR_p_kcat)
Rule('catalyze_EGFR_u', EGFR(Tyr='p') >> EGFR(Tyr='u'), catalyze_EGFR_u_kcat)
Rule('pEGFR_is_endocytosed', EGFR(Tyr='p') ** PM >> EGFR(Tyr='p') ** EM, catalyze_EGFR_kendo, move_connected=True)
Rule('EGFR_is_recycled', EGFR() ** EM >> EGFR() ** PM, catalyze_EGFR_recycl_kcat, move_connected=True)
Rule('ubEGFR_is_degraded', EGFR(ub='true') ** EM >> None, catalyze_EGFR_deg_kcat, delete_molecules=True)
Rule('synthesis_mEGFR', None >> mEGFR() ** PM, msynthesize_pERK_EGFR_ksyn)
Rule('basal_degradation_mEGFR', mEGFR() >> None, msynthesize_pERK_EGFR_kdeg, delete_molecules=True)
Rule('synthesis_pEGFR', mEGFR() ** PM >> mEGFR() ** PM + EGFR(KD=None, Tyr='u', rtkf=None, ub='false') ** PM, psynthesize_pERK_EGFR_ksyn)
Rule('basal_degradation_pEGFR', EGFR() >> None, psynthesize_pERK_EGFR_kdeg, delete_molecules=True)
Rule('bind_pEGFR_GRB2', GRB2(SH2=None) + EGFR(Tyr='p') | GRB2(SH2=1) % EGFR(Tyr=('p', 1)), bind_pEGFR_GRB2_phi, Ea0_bind_pEGFR_GRB2, energy=True)
Rule('SPRY_and_GRB2_bind_and_dissociate', SPRY(SH3m=None) + GRB2(SH3=None) | SPRY(SH3m=1) % GRB2(SH3=1), bind_SPRY_GRB2_phi, Ea0_bind_SPRY_GRB2, energy=True)
Rule('synthesis_mSPRY', None >> mSPRY() ** CP, msynthesize_pERK_SPRY_ksyn)
Rule('basal_degradation_mSPRY', mSPRY() >> None, msynthesize_pERK_SPRY_kdeg, delete_molecules=True)
Rule('synthesis_pSPRY', mSPRY() ** CP >> mSPRY() ** CP + SPRY(SH3m=None) ** CP, psynthesize_pERK_SPRY_ksyn)
Rule('basal_degradation_pSPRY', SPRY() >> None, psynthesize_pERK_SPRY_kdeg, delete_molecules=True)
Rule('SOS1_is_dephosphorylated', SOS1(S1134='p') >> SOS1(S1134='u'), catalyze_phosphatase_SOS1_uS1134_kcat)
Rule('GRB2_and_SOS1_bind_and_dissociate', GRB2(SH3=None) + SOS1(SH3m=None) | GRB2(SH3=1) % SOS1(SH3m=1), bind_GRB2_SOS1_phi, Ea0_bind_GRB2_SOS1, energy=True)
Rule('SOS1_is_phosphorylated', SOS1(S1134='u') >> SOS1(S1134='p'), catalyze_pERK_SOS1_pS1134_kbase)
Rule('pERK_phosphorylates_SOS1', ERK(phospho='p') + SOS1(S1134='u') >> ERK(phospho='p') + SOS1(S1134='p'), catalyze_pERK_SOS1_pS1134_kcat)
Rule('pERK_phosphorylates_CRAF', ERK(phospho='p') + CRAF(S642='u', raf=None) >> ERK(phospho='p') + CRAF(S642='p', raf=None), catalyze_pERK_CRAF_pS642_kcat)
Rule('CRAF_is_dephosphorylated', CRAF(S642='p', raf=None) >> CRAF(S642='u', raf=None), catalyze_phosphatase_CRAF_uS642_kcat)
Rule('pERK_phosphorylates_BRAF', ERK(phospho='p') + BRAF(T753='u', raf=None) >> ERK(phospho='p') + BRAF(T753='p', raf=None), catalyze_pERK_BRAF_pT753_kcat)
Rule('BRAF_is_dephosphorylated', BRAF(T753='p', raf=None) >> BRAF(T753='u', raf=None), catalyze_phosphatase_BRAF_uT753_kcat)
Rule('RTK_and_GRB2_bound_SOS1_binds_RASgdp', GRB2(SH2=ANY, SH3=1) % SOS1(SH3m=1, ras=None) + RAS(sos1=None, state='gdp') >> GRB2(SH2=ANY, SH3=1) % SOS1(SH3m=1, ras=2) % RAS(sos1=2, state='gdp'), bind_SOS1_RAS_kf)
Rule('SOS1_dissociates_from_RAS', SOS1(ras=1) % RAS(sos1=1) >> SOS1(ras=None) + RAS(sos1=None), bind_SOS1_RAS_kr)
Rule('SOS1_catalyzes_RAS_guanosine_exchange', SOS1(ras=1) % RAS(sos1=1, state='gdp') >> SOS1(ras=None) + RAS(sos1=None, state='gtp'), catalyze_SOS1_RAS_gtp_kcat)
Rule('RAS_hydrolysis_GTP', RAS(raf=None, state='gtp') >> RAS(raf=None, state='gdp'), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_BRAF_from_RAS', RAS(raf=1, state='gtp') % BRAF(RBD=1, raf=None) >> RAS(raf=None, state='gdp') + BRAF(RBD=None, raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_BRAF_BRAF_from_RAS', RAS(raf=1, state='gtp') % BRAF(RBD=1, raf=2) % BRAF(raf=2) >> RAS(raf=None, state='gdp') + BRAF(RBD=None, raf=None) + BRAF(raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_BRAF_CRAF_from_RAS', RAS(raf=1, state='gtp') % BRAF(RBD=1, raf=2) % CRAF(raf=2) >> RAS(raf=None, state='gdp') + BRAF(RBD=None, raf=None) + CRAF(raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_CRAF_from_RAS', RAS(raf=1, state='gtp') % CRAF(RBD=1, raf=None) >> RAS(raf=None, state='gdp') + CRAF(RBD=None, raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_CRAF_BRAF_from_RAS', RAS(raf=1, state='gtp') % CRAF(RBD=1, raf=2) % BRAF(raf=2) >> RAS(raf=None, state='gdp') + CRAF(RBD=None, raf=None) + BRAF(raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('GTP_hydrolysis_dissociates_CRAF_CRAF_from_RAS', RAS(raf=1, state='gtp') % CRAF(RBD=1, raf=2) % CRAF(raf=2) >> RAS(raf=None, state='gdp') + CRAF(RBD=None, raf=None) + CRAF(raf=None), catalyze_NF1_RAS_gdp_kcat)
Rule('CBL_and_GRB2_bind_and_dissociate', CBL(SH3m=None) + GRB2(SH3=None) | CBL(SH3m=1) % GRB2(SH3=1), bind_CBL_GRB2_phi, Ea0_bind_CBL_GRB2, energy=True)
Rule('catalyze_CBL_EGFR_ub', EGFR(ub='false') % CBL() >> EGFR(ub='true') % CBL(), catalyze_CBL_EGFR_ub_kcat)
Rule('catalyze_EGFR_dub', EGFR(ub='true') >> EGFR(ub='false'), catalyze_EGFR_dub_kcat)

EnergyPattern('ep_BRAF_RAF_mod_pRAF', as_complex_pattern(BRAF(raf=ANY, T753='p')), ep_RAF_RAF_mod_pRAF_ddG)
EnergyPattern('ep_CRAF_RAF_mod_pRAF', as_complex_pattern(CRAF(raf=ANY, S642='p')), ep_RAF_RAF_mod_pRAF_ddG)
EnergyPattern('ep_BRAF_RAS_mod_pRAF', as_complex_pattern(BRAF(RBD=ANY, T753='p')), ep_RAF_RAS_mod_pRAF_ddG)
EnergyPattern('ep_CRAF_RAS_mod_pRAF', as_complex_pattern(CRAF(RBD=ANY, S642='p')), ep_RAF_RAS_mod_pRAF_ddG)
EnergyPattern('ep_bind_RASgtp_BRAF', RAS(raf=1, state='gtp') % BRAF(RBD=1), bind_RASgtp_RAF_dG)
EnergyPattern('ep_bind_RASgtp_CRAF', RAS(raf=1, state='gtp') % CRAF(RBD=1), bind_RASgtp_RAF_dG)
EnergyPattern('ep_bind_BRAF_BRAF', BRAF(raf=1) % BRAF(raf=1), bind_RAF_RAF_dG)
EnergyPattern('ep_bind_BRAF_CRAF', BRAF(raf=1) % CRAF(raf=1), bind_RAF_RAF_dG)
EnergyPattern('ep_bind_CRAF_CRAF', CRAF(raf=1) % CRAF(raf=1), bind_RAF_RAF_dG)
EnergyPattern('ep_bind_RAFi_BRAF', RAFi(raf=1) % BRAF(rafi=1), bind_RAFi_RAF_dG)
EnergyPattern('ep_bind_RAFi_CRAF', RAFi(raf=1) % CRAF(rafi=1), bind_RAFi_RAF_dG)
EnergyPattern('ep_bind_PRAFi_BRAF', PRAFi(raf=1) % BRAF(rafi=1), bind_PRAFi_RAF_dG)
EnergyPattern('ep_bind_PRAFi_CRAF', PRAFi(raf=1) % CRAF(rafi=1), bind_PRAFi_RAF_dG)
EnergyPattern('ep_BRAF_BRAF_mod_RAFi_single', BRAF(raf=1) % BRAF(raf=1, rafi=2) % RAFi(raf=2), ep_RAF_RAF_mod_RAFi_single_ddG)
EnergyPattern('ep_BRAF_CRAF_mod_RAFi_single', BRAF(raf=1) % CRAF(raf=1, rafi=2) % RAFi(raf=2), ep_RAF_RAF_mod_RAFi_single_ddG)
EnergyPattern('ep_CRAF_BRAF_mod_RAFi_single', CRAF(raf=1) % BRAF(raf=1, rafi=2) % RAFi(raf=2), ep_RAF_RAF_mod_RAFi_single_ddG)
EnergyPattern('ep_CRAF_CRAF_mod_RAFi_single', CRAF(raf=1) % CRAF(raf=1, rafi=2) % RAFi(raf=2), ep_RAF_RAF_mod_RAFi_single_ddG)
EnergyPattern('ep_BRAF_BRAF_mod_RAFi_double', RAFi(raf=2) % BRAF(raf=1, rafi=2) % BRAF(raf=1, rafi=3) % RAFi(raf=3), ep_RAF_RAF_mod_RAFi_double_Gf)
EnergyPattern('ep_BRAF_CRAF_mod_RAFi_double', RAFi(raf=2) % BRAF(raf=1, rafi=2) % CRAF(raf=1, rafi=3) % RAFi(raf=3), ep_RAF_RAF_mod_RAFi_double_Gf)
EnergyPattern('ep_CRAF_CRAF_mod_RAFi_double', RAFi(raf=2) % CRAF(raf=1, rafi=2) % CRAF(raf=1, rafi=3) % RAFi(raf=3), ep_RAF_RAF_mod_RAFi_double_Gf)
EnergyPattern('ep_bind_BRAF_uMEK', BRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_uMEK_dG)
EnergyPattern('ep_bind_CRAF_uMEK', CRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_uMEK_dG)
EnergyPattern('ep_bind_MEKi_MEK', MEKi(mek=1) % MEK(meki=1), bind_MEKi_MEK_dG)
EnergyPattern('ep_pMEK_MEKi_single', MEK(meki=1, phospho='p') % MEKi(mek=1), ep_pMEK_MEKi_ddG)
EnergyPattern('ep_bind_EGF_EGFR', EGF(rtk=1) % EGFR(rtkf=1), bind_EGF_EGFR_dG)
EnergyPattern('ep_bind_EGFR_EGFR', EGFR(KD=1) % EGFR(Tyr=1), bind_EGFR_EGFR_dG)
EnergyPattern('ep_EGFR_EGFR_EGF', EGF(rtk=1) % EGFR(KD=ANY, rtkf=1), ep_EGFR_EGFR_mod_EGF_single_ddG)
EnergyPattern('ep_bind_pEGFR_GRB2', EGFR(Tyr=1) % GRB2(SH2=1), bind_pEGFR_GRB2_dG)
EnergyPattern('ep_bind_SPRY_GRB2', SPRY(SH3m=1) % GRB2(SH3=1), bind_SPRY_GRB2_dG)
EnergyPattern('ep_bind_GRB2_SOS1', GRB2(SH3=1) % SOS1(SH3m=1), bind_GRB2_SOS1_dG)
EnergyPattern('ep_SOS1S1134p_GRB2_single', SOS1(S1134='p', SH3m=1) % GRB2(SH3=1), ep_SOS1S1134p_GRB2_ddG)
EnergyPattern('ep_bind_CBL_GRB2', CBL(SH3m=1) % GRB2(SH3=1), bind_CBL_GRB2_dG)

Initial(BRAF(AA600='E', RBD=None, mek=None, raf=None, rafi=None, T753='u') ** CP, initBRAF)
Initial(CRAF(RBD=None, mek=None, raf=None, rafi=None, S642='u') ** CP, initCRAF)
Initial(RAS(raf=None, sos1=None, state='gdp') ** CP, initRAS)
Initial(RAFi(raf=None) ** CP, RAFi_0, fixed=True)
Initial(PRAFi(raf=None) ** CP, PRAFi_0, fixed=True)
Initial(MEK(Dsite=None, meki=None, phospho='u', raf=None) ** CP, initMEK)
Initial(MEKi(mek=None) ** CP, MEKi_0, fixed=True)
Initial(ERK(CD=None, phospho='u') ** CP, initERK)
Initial(EGF(rtk=None) ** CP, initEGF, fixed=True)
Initial(GRB2(SH2=None, SH3=None) ** CP, initGRB2)
Initial(SOS1(S1134='u', SH3m=None, ras=None) ** CP, initSOS1)
Initial(CBL(SH3m=None) ** CP, initCBL)

Annotation(RASgtp_and_BRAF_bind_and_dissociate, 'http://identifiers.org/pubmed/7969158', 'isDescribedBy')
Annotation(RASgtp_and_CRAF_bind_and_dissociate, 'http://identifiers.org/pubmed/7969158', 'isDescribedBy')
Annotation(BRAF_and_uMEK_bind_and_dissociate, 'http://identifiers.org/pubmed/25155755', 'isDescribedBy')
Annotation(CRAF_and_uMEK_bind_and_dissociate, 'http://identifiers.org/pubmed/25155755', 'isDescribedBy')
Annotation(bind_MEK_uERK, 'http://identifiers.org/pubmed/10655591', 'isDescribedBy')
Annotation(bind_MEK_uERK, 'http://identifiers.org/pubmed/11157753', 'isDescribedBy')
Annotation(bind_MEK_uERK, 'http://identifiers.org/pubmed/10567369', 'isDescribedBy')
Annotation(bind_MEK_uERK, 'http://identifiers.org/pubmed/15979847', 'isDescribedBy')
Annotation(pMEK_phosphorylates_ERK, 'http://identifiers.org/pubmed/19406201', 'isDescribedBy')
Annotation(bind_DUSP_pERK, 'http://identifiers.org/pubmed/10655591', 'isDescribedBy')
Annotation(bind_DUSP_pERK, 'http://identifiers.org/pubmed/11157753', 'isDescribedBy')
Annotation(EGF_and_EGFR_bind_and_dissociate, 'http://identifiers.org/pubmed/16946702', 'isDescribedBy')
Annotation(GRB2_and_SOS1_bind_and_dissociate, 'http://identifiers.org/pubmed/7893993', 'isDescribedBy')
Annotation(RTK_and_GRB2_bound_SOS1_binds_RASgdp, 'http://identifiers.org/pubmed/26565026', 'isDescribedBy')
Annotation(BRAF, 'http://identifiers.org/uniprot/P15056', 'is')
Annotation(CRAF, 'http://identifiers.org/uniprot/P04049', 'is')
Annotation(RAS, 'http://identifiers.org/uniprot/P01111', 'is')
Annotation(RAS, 'http://identifiers.org/uniprot/P01116', 'is')
Annotation(RAS, 'http://identifiers.org/uniprot/P01112', 'is')
Annotation(RAFi, 'http://identifiers.org/chebi/63637', 'is')
Annotation(RAFi, 'http://identifiers.org/chebi/75045', 'is')
Annotation(MEK, 'http://identifiers.org/uniprot/Q02750', 'is')
Annotation(MEK, 'http://identifiers.org/uniprot/P36507', 'is')
Annotation(MEKi, 'http://identifiers.org/chebi/90851', 'is')
Annotation(MEKi, 'http://identifiers.org/chebi/75998', 'is')
Annotation(MEKi, 'http://identifiers.org/chebi/90227', 'is')
Annotation(MEKi, 'http://identifiers.org/chebi/145371', 'is')
Annotation(MEKi, 'http://identifiers.org/chebi/88249', 'is')
Annotation(ERK, 'http://identifiers.org/uniprot/P27361', 'is')
Annotation(ERK, 'http://identifiers.org/uniprot/P28482', 'is')
Annotation(DUSP, 'http://identifiers.org/uniprot/Q13115', 'is')
Annotation(DUSP, 'http://identifiers.org/uniprot/Q16828', 'is')
Annotation(mDUSP, 'http://identifiers.org/hgnc/3070', 'is')
Annotation(mDUSP, 'http://identifiers.org/hgnc/3072', 'is')
Annotation(EGF, 'http://identifiers.org/uniprot/P01133', 'is')
Annotation(EGFR, 'http://identifiers.org/uniprot/P00533', 'is')
Annotation(mEGFR, 'http://identifiers.org/hgnc/3236', 'is')
Annotation(GRB2, 'http://identifiers.org/uniprot/P62993', 'is')
Annotation(SPRY, 'http://identifiers.org/uniprot/O43597', 'is')
Annotation(SPRY, 'http://identifiers.org/uniprot/Q9C004', 'is')
Annotation(mSPRY, 'http://identifiers.org/hgnc/11270', 'is')
Annotation(mSPRY, 'http://identifiers.org/hgnc/15533', 'is')
Annotation(SOS1, 'http://identifiers.org/uniprot/Q07889', 'is')
Annotation(CBL, 'http://identifiers.org/uniprot/P22681', 'is')
