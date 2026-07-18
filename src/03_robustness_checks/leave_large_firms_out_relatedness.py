# ================================================================
# leave_large_firms_out_relatedness.py
# ----------------------------------------------------------------
# Reconstructs the field-pair similarity matrix R_{gh,t-1} excluding
# the top 1% of applicant codes by total filing volume from its
# construction, recomputes relatedness for ALL firms (including the
# excluded ones) using this large-firm-excluded matrix, and
# re-estimates Table 1's preferred RQ1 specification with the
# reconstructed relatedness measure in place of the baseline one.
#
# This addresses the concern that a firm's own prior filings
# contribute to the shared field-pair similarity matrix used to
# compute that same firm's relatedness score (self-referential
# construction). See paper Appendix A ("Relatedness construction")
# and Table A2 for the reported results:
#   baseline coefficient:            0.904 (p<0.001)
#   large-firm-excluded coefficient: 0.871 (p=0.013)
#   correlation between measures:    0.70
#
# Requires: panel_v3_clean.pkl, df_application_v3_clean.pkl
#   (see src/01_data_construction for how these are built)
#
# Usage:
#   python leave_large_firms_out_relatedness.py
# ================================================================

import pandas as pd
import numpy as np
import time
import pyfixest as pf

CACHE_DIR = '.'  # set to the directory containing the cached .pkl files

t0 = time.time()

# ----------------------------------------------------------------
# [1] Reconstruct R_{gh,t-1} excluding the top 1% of firms
# ----------------------------------------------------------------
panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3_clean.pkl')
ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')

counts = ap.groupby('applicant_id').size()
top1pct_ids = set(counts.sort_values(ascending=False).head(int(len(counts) * 0.01)).index)
print(f"Top 1% firms: {len(top1pct_ids)} ({time.time()-t0:.1f}s)")

fields = sorted(panel['field_id'].unique())
years = sorted(panel['app_year'].unique())


def cosine_sim_matrix(M):
    """M: firms x fields (patent_stock) -> fields x fields cosine similarity."""
    norm = np.linalg.norm(M, axis=0)
    norm[norm == 0] = 1e-12
    Mn = M / norm
    return Mn.T @ Mn


R_excl_by_year = {}
for y in years:
    sub = panel[(panel['app_year'] == y) & (~panel['applicant_id'].isin(top1pct_ids))]
    if len(sub) == 0:
        continue
    piv = sub.pivot_table(index='applicant_id', columns='field_id', values='patent_stock', fill_value=0)
    piv = piv.reindex(columns=fields, fill_value=0)
    R = cosine_sim_matrix(piv.values)
    R_excl_by_year[y] = pd.DataFrame(R, index=fields, columns=fields)
print(f"R matrices built for {len(R_excl_by_year)} years ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------------
# [2] Recompute relatedness for ALL firms using the large-firm-
#     excluded matrix (including the excluded top firms themselves)
# ----------------------------------------------------------------
results = []
for y in years:
    if y not in R_excl_by_year:
        continue
    R_year = R_excl_by_year[y]
    sub = panel[panel['app_year'] == y][['applicant_id', 'field_id', 'patent_stock']].copy()
    piv_all = sub.pivot_table(index='applicant_id', columns='field_id', values='patent_stock', fill_value=0)
    piv_all = piv_all.reindex(columns=fields, fill_value=0)
    for g in fields:
        others = [f for f in fields if f != g]
        w = piv_all[others].values
        denom = w.sum(axis=1)
        denom_safe = np.where(denom == 0, np.nan, denom)
        w_norm = w / denom_safe[:, None]
        Rg = R_year.loc[g, others].values
        rho = (w_norm * Rg[None, :]).sum(axis=1)
        tmp = pd.DataFrame({'applicant_id': piv_all.index, 'field_id': g, 'app_year': y,
                             'relatedness_excl_large': rho})
        results.append(tmp)
    if y % 5 == 0:
        print(f"  year {y} done ({time.time()-t0:.1f}s)")

rel_excl = pd.concat(results, ignore_index=True)
rel_excl.to_pickle(f'{CACHE_DIR}/relatedness_excl_large_firms.pkl')
print(f"Saved relatedness_excl_large_firms.pkl ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------------
# [3] Merge and re-estimate Table 1's preferred RQ1 specification
# ----------------------------------------------------------------
panel = panel.merge(rel_excl, on=['applicant_id', 'field_id', 'app_year'], how='left')
corr = panel[['relatedness_dynamic', 'relatedness_excl_large']].corr().iloc[0, 1]
print(f"Correlation (baseline vs. large-firm-excluded relatedness): {corr:.3f}")

first_entry = (panel[panel['n_filed'] > 0].groupby(['applicant_id', 'field_id'])['app_year']
               .min().reset_index().rename(columns={'app_year': 'first_entry_year'}))
p2 = panel.merge(first_entry, on=['applicant_id', 'field_id'], how='left')
p2 = p2[p2['app_year'] > p2['first_entry_year']].copy()

p2['log_decayed_stock'] = np.log1p(p2['decayed_stock_d10'])
p2['log_grant_stock'] = np.log1p(p2['grant_stock'])

m_excl = pf.fepois(
    'n_filed ~ log_decayed_stock + log_grant_stock + relatedness_excl_large | applicant_id^field_id + app_year',
    data=p2, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
print(f"\n=== RQ1 preferred spec, large-firm-excluded relatedness ({time.time()-t0:.1f}s) ===")
print(m_excl.summary())
