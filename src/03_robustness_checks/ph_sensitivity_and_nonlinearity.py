# ================================================================
# ph_sensitivity_and_nonlinearity.py
# ----------------------------------------------------------------
# Two supplementary checks reported in Appendix A:
#
#   [1] A piecewise Cox specification for the grant-timing model,
#       splitting the risk period at three years from the
#       examination-request date, addressing the detected marginal
#       proportional-hazards violation for decayed stock
#       (test statistic 4.70, p=0.030). Reported results:
#         0-3 years:  coef=0.019 (p=0.360)
#         3+ years:   coef=0.034 (p=0.003)
#         formal difference test: z=0.67, p=0.50 (not significant)
#
#   [2] A quadratic relatedness term in Model 1A's preferred
#       specification, testing for nonlinearity (relatedness must
#       be mean-centered first, since uncentered relatedness and
#       its square correlate at 0.96 in this sample). Reported
#       results: linear=1.114 (p=0.002), quadratic=-1.621 (p=0.102),
#       implied turning point rho ~= 0.62.
#
# Requires: df_application_v3_clean.pkl, panel_v3_clean.pkl
#
# Usage:
#   python ph_sensitivity_and_nonlinearity.py
# ================================================================

import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from lifelines import CoxPHFitter
import pyfixest as pf

CACHE_DIR = '.'

t0 = time.time()

# ================================================================
# [1] Piecewise Cox: proportional-hazards sensitivity for stock
# ================================================================
print("=" * 60)
print("[1] Piecewise Cox for grant timing (0-3yr vs 3yr+)")
print("=" * 60)

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')
ap['log_decayed_stock'] = np.log1p(ap['decayed_stock_d10'])
adate, rdate = ap['adate_dt'], ap['rdate_dt']
censoring_time = ap['censoring_time']
ap['event_grant_simple'] = ap['granted'].astype(int)
ap['duration_grant_simple'] = np.where(
    ap['event_grant_simple'] == 1, (rdate - adate).dt.days, (censoring_time - adate).dt.days)

cols = ['log_decayed_stock', 'relatedness_dynamic', 'own_portfolio_hhi',
        'corporate_flag', 'post_2001_reform', 'field_id', 'applicant_id']
sub = ap[['duration_grant_simple', 'event_grant_simple'] + cols].dropna()
sub = sub[sub['duration_grant_simple'] > 0].copy()
sub['corporate_flag'] = sub['corporate_flag'].astype(int)

CUTOFF = 1095  # three years, in days

rows_early = sub.copy()
rows_early['duration_grant_simple'] = np.minimum(rows_early['duration_grant_simple'], CUTOFF)
rows_early['event_grant_simple'] = np.where(sub['duration_grant_simple'] <= CUTOFF,
                                             sub['event_grant_simple'], 0)

rows_late = sub[sub['duration_grant_simple'] > CUTOFF].copy()
rows_late['duration_grant_simple'] = rows_late['duration_grant_simple'] - CUTOFF

ts = time.time()
cph_early = CoxPHFitter()
cph_early.fit(rows_early, duration_col='duration_grant_simple', event_col='event_grant_simple',
              strata=['field_id', 'post_2001_reform', 'corporate_flag'], cluster_col='applicant_id')
early_coef = cph_early.summary.loc['log_decayed_stock']
print(f"0-3 years (n={len(rows_early):,}, event rate={rows_early['event_grant_simple'].mean():.1%}, "
      f"{time.time()-ts:.1f}s)")
print(early_coef[['coef', 'se(coef)', 'p']])

ts = time.time()
cph_late = CoxPHFitter()
cph_late.fit(rows_late, duration_col='duration_grant_simple', event_col='event_grant_simple',
             strata=['field_id', 'post_2001_reform', 'corporate_flag'], cluster_col='applicant_id')
late_coef = cph_late.summary.loc['log_decayed_stock']
print(f"\n3+ years (n={len(rows_late):,}, event rate={rows_late['event_grant_simple'].mean():.1%}, "
      f"{time.time()-ts:.1f}s)")
print(late_coef[['coef', 'se(coef)', 'p']])

# Formal (approximate, independent-samples) test of coefficient equality
b1, se1 = early_coef['coef'], early_coef['se(coef)']
b2, se2 = late_coef['coef'], late_coef['se(coef)']
diff = b2 - b1
se_diff = np.sqrt(se1**2 + se2**2)
z = diff / se_diff
p = 2 * (1 - norm.cdf(abs(z)))
print(f"\nDifference test: diff={diff:.4f}, SE={se_diff:.4f}, z={z:.3f}, p={p:.4f}")

# ================================================================
# [2] Relatedness nonlinearity (quadratic term, mean-centered)
# ================================================================
print("\n" + "=" * 60)
print("[2] Relatedness nonlinearity in Model 1A preferred spec")
print("=" * 60)

panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3_clean.pkl')
first_entry = (panel[panel['n_filed'] > 0].groupby(['applicant_id', 'field_id'])['app_year']
               .min().reset_index().rename(columns={'app_year': 'first_entry_year'}))
p2 = panel.merge(first_entry, on=['applicant_id', 'field_id'], how='left')
p2 = p2[p2['app_year'] > p2['first_entry_year']].copy()
p2['log_decayed_stock'] = np.log1p(p2['decayed_stock_d10'])
p2['log_grant_stock'] = np.log1p(p2['grant_stock'])

mean_rho = p2['relatedness_dynamic'].mean()
p2['rho_centered'] = p2['relatedness_dynamic'] - mean_rho
p2['rho_centered_sq'] = p2['rho_centered'] ** 2
print(f"Sample mean relatedness: {mean_rho:.4f}")
print("Correlation, uncentered rho vs rho^2:",
      p2['relatedness_dynamic'].corr(p2['relatedness_dynamic'] ** 2))
print("Correlation, centered rho vs rho^2:",
      p2['rho_centered'].corr(p2['rho_centered_sq']))

ts = time.time()
m_nonlin = pf.fepois(
    'n_filed ~ log_decayed_stock + log_grant_stock + rho_centered + rho_centered_sq '
    '| applicant_id^field_id + app_year',
    data=p2, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
print(f"\nFitted ({time.time()-ts:.1f}s)")
print(m_nonlin.summary())

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
