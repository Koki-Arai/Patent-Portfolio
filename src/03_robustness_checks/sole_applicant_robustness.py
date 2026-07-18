# ================================================================
# sole_applicant_robustness.py
# 単独出願人限定サンプルでのRQ1・RQ2頑健性チェック（Google Colab）
# ----------------------------------------------------------------
# これまでの全分析はfirst-listed applicantで出願人を識別しており、
# 共同出願・大学産学連携等では、第2出願人以降を通じた技術的連続性が
# 欠落する可能性があった。本スクリプトは、applicant.txtのseq列から
# 出願人数を数え、単独出願（=first-listed applicantが唯一の出願人）
# のみに限定してRQ1(Model1A)・RQ2(Model2)を再推定する。
#
# ローカル検証(2020年代データのみ)では、単独出願の割合は92.65%
# （636,729件中）——大半が単独出願人であることを確認済み。
#
# 実行方法：
#   1. df_application_v3_clean.pkl, panel_v3_clean.pkl,
#      applicant_1990s.txt〜applicant_2020s.txt を用意
#   2. pip install pyfixest
#   3. exec(open('/content/sole_applicant_robustness.py').read())
# ================================================================

import pandas as pd
import numpy as np
import time
import pyfixest as pf

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
IIP_DIR = '/content'
DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'

t0 = time.time()
print("=" * 70)
print("単独出願人限定サンプルでのRQ1・RQ2頑健性チェック — 開始")
print("=" * 70)

# ----------------------------------------------------------------
# [1] 出願人数のカウント（applicant.txtのseq列から）
# ----------------------------------------------------------------
print(f"\n[1/3] Counting applicants per application ...", flush=True)
appl_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/applicant_{dec}.txt'
    df = pd.read_csv(path, sep=SEP, encoding=ENCODING, usecols=['ida', 'seq'], dtype=str)
    appl_frames.append(df)
appl_all = pd.concat(appl_frames, ignore_index=True)
n_applicants = appl_all.groupby('ida')['seq'].nunique().rename('n_applicants')
print(f"  applications counted: {len(n_applicants):,}")
print(f"  sole-applicant share: {(n_applicants==1).mean():.1%}")

sole_ids = set(n_applicants[n_applicants == 1].index)

# ----------------------------------------------------------------
# [2] RQ1 (Model 1A) — 単独出願人限定
# ----------------------------------------------------------------
print(f"\n[2/3] RQ1 sole-applicant restriction ...", flush=True)
ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')
ap['is_sole_applicant'] = ap['app_id'].isin(sole_ids)
print(f"  matched sole-applicant flag rate in ap: {ap['is_sole_applicant'].mean():.1%}")

panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3_clean.pkl')
# 出願人単位のpanelには、当該firm-fieldのすべての出願が単独出願人か
# どうかを示すフラグを付与(全出願が単独出願人の場合のみ残す、
# 保守的な定義)
sole_by_firm_field_year = (ap.groupby(['applicant_id', 'field_id', 'app_year'])['is_sole_applicant']
                            .mean().reset_index().rename(columns={'is_sole_applicant': 'sole_share'}))
panel = panel.merge(sole_by_firm_field_year, on=['applicant_id', 'field_id', 'app_year'], how='left')
panel['log_decayed_stock'] = np.log1p(panel['decayed_stock_d10'])
panel['log_grant_stock'] = np.log1p(panel['grant_stock'])

first_entry = (panel[panel['n_filed'] > 0].groupby(['applicant_id', 'field_id'])['app_year']
               .min().reset_index().rename(columns={'app_year': 'first_entry_year'}))
p2 = panel.merge(first_entry, on=['applicant_id', 'field_id'], how='left')
p2 = p2[p2['app_year'] > p2['first_entry_year']].copy()

# 単独出願人比率が閾値以上のfirm-field-year観測のみに限定(sole_share>=0.9)
p2_sole = p2[(p2['sole_share'].isna()) | (p2['sole_share'] >= 0.9)].copy()
print(f"  restricted sample: {len(p2_sole):,} / {len(p2):,} rows")

m_sole = pf.fepois('n_filed ~ log_decayed_stock + log_grant_stock + relatedness_dynamic | applicant_id^field_id + app_year',
                    data=p2_sole, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
print(f"\n=== RQ1 preferred spec, sole-applicant-dominant sample ===")
print(m_sole.summary())

# ----------------------------------------------------------------
# [3] RQ2 (Model 2, Part 2) — 単独出願のみ
# ----------------------------------------------------------------
print(f"\n[3/3] RQ2 sole-applicant restriction ...", flush=True)
ap['log_decayed_stock'] = np.log1p(ap['decayed_stock_d10'])
sub = ap[ap['is_sole_applicant']].dropna(subset=['relatedness_dynamic', 'own_portfolio_hhi', 'n_active_fields'])
fml_vars = 'log_decayed_stock + relatedness_dynamic + own_portfolio_hhi + n_active_fields'

for ctx in ['rejection_ctx', 'grant_ctx']:
    sub2 = sub[sub[f'{ctx}_context_occurred'] == 1]
    m = pf.feols(f'{ctx}_any_self_within_context ~ {fml_vars} | applicant_id + field_id + app_year',
                 data=sub2, vcov={'CRV1': 'applicant_id'})
    print(f"\n--- {ctx}, own-patent citation | context recorded, sole-applicant only (n={len(sub2):,}) ---")
    print(m.summary())

print(f"\n完了 ({time.time()-t0:.1f}s total)")
print("\n解釈：単独出願人限定でも係数の符号・有意性のパターンが維持されれば、")
print("共同出願・大学産学連携によるfirst-listed applicant限定の測定誤差が")
print("RQ1・RQ2の結論を大きく歪めていないことが確認できる。")
