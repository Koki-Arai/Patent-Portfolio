# ================================================================
# divisional_family_approximation.py
# 分割出願・ファミリー近似と頑健性チェック（Google Colab）
# ----------------------------------------------------------------
# IIPデータには特許ファミリー識別子が直接存在しないため、以下の
# ヒューリスティックで「分割出願・手続的な同時大量出願」を近似する：
#
#   同一applicant_id × 同一field_id × 同一出願日(adate)に
#   3件以上の出願が集中している場合、それらを
#   "bulk_filing_flag=1" として近似的にフラグする。
#
# この定義は真のfamily/divisional識別ではなく、「単一の発明群が
# 手続的に分割・多重出願された結果、出願日が完全に一致する」という
# 観察可能な代理指標に基づく、保守的な近似である。
#
# 頑健性チェックは2通り：
#   [A] bulk filing (flag=1)を除外してModel1A主仕様を再推定
#   [B] intensive margin (n_filed件数)ではなく、extensive margin
#       (その年に1件でも出願したか、二値)で再推定し、同じ結論が
#       出るか確認する。件数の水増しが結果を駆動しているなら、
#       二値アウトカムでは効果が弱まるはずである。
#
# 実データでの検証結果(ローカル、フルサンプル)：
#   bulk-filing flagged applications: 10.7%
#   [A] bulk除外後: decayed_stock 0.489*** (元0.507), grant_stock
#       -0.116*** (元-0.120), relatedness 0.660*** (元0.904、
#       p=0.003、約27%の減衰だが依然として有意)
#   [B] extensive margin: decayed_stock 0.019*** , grant_stock
#       -0.059***, relatedness 0.062* — 符号・有意性は維持
#   → bulk filingの寄与を除いても、また出願件数ではなく出願の
#     有無という二値アウトカムでも、RQ1の結論(3変数すべて有意)は
#     頑健に維持される。relatednessの減衰(0.904→0.660)は本文に
#     報告すべき、無視できない大きさである。
#
# 実行方法：
#   1. df_application_v3_clean.pkl, panel_v3_clean.pkl を用意
#   2. pip install pyfixest
#   3. exec(open('/content/divisional_family_approximation.py').read())
# ================================================================

import pandas as pd
import numpy as np
import time
import pyfixest as pf

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
BULK_FILING_THRESHOLD = 3  # 同一applicant×field×adateでの出願数がこれ以上ならbulk filingとみなす

t0 = time.time()
print("=" * 70)
print("分割出願・ファミリー近似と頑健性チェック — 開始")
print("=" * 70)

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')

# ----------------------------------------------------------------
# [1] Bulk filing (分割出願近似) のフラグ付け
# ----------------------------------------------------------------
print(f"\n[1/3] Flagging bulk (same-day, same-firm-field) filings ...", flush=True)
same_day_counts = (ap.groupby(['applicant_id', 'field_id', 'adate'])
                     .size().rename('n_same_day').reset_index())
ap = ap.merge(same_day_counts, on=['applicant_id', 'field_id', 'adate'], how='left')
ap['bulk_filing_flag'] = (ap['n_same_day'] >= BULK_FILING_THRESHOLD).astype(int)
print(f"  bulk-filing flagged applications: {ap['bulk_filing_flag'].sum():,} "
      f"({ap['bulk_filing_flag'].mean():.1%} of sample)")

# ----------------------------------------------------------------
# [2] Panelレベルでbulk filing出願を除いたn_filedを再構築
# ----------------------------------------------------------------
print(f"\n[2/3] Rebuilding panel excluding bulk filings, re-estimating RQ1 ...", flush=True)
panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3_clean.pkl')

n_filed_excl_bulk = (ap[ap['bulk_filing_flag'] == 0]
                      .groupby(['applicant_id', 'field_id', 'app_year'])
                      .size().rename('n_filed_excl_bulk').reset_index())
panel = panel.merge(n_filed_excl_bulk, on=['applicant_id', 'field_id', 'app_year'], how='left')
panel['n_filed_excl_bulk'] = panel['n_filed_excl_bulk'].fillna(0)
panel['extensive_margin'] = (panel['n_filed'] > 0).astype(int)

panel['log_decayed_stock'] = np.log1p(panel['decayed_stock_d10'])
panel['log_grant_stock'] = np.log1p(panel['grant_stock'])

first_entry = (panel[panel['n_filed'] > 0].groupby(['applicant_id', 'field_id'])['app_year']
               .min().reset_index().rename(columns={'app_year': 'first_entry_year'}))
p2 = panel.merge(first_entry, on=['applicant_id', 'field_id'], how='left')
p2 = p2[p2['app_year'] > p2['first_entry_year']].copy()

m_excl_bulk = pf.fepois('n_filed_excl_bulk ~ log_decayed_stock + log_grant_stock + relatedness_dynamic | applicant_id^field_id + app_year',
                         data=p2, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
print(f"\n=== [A] Bulk filing除外後のn_filed ({time.time()-t0:.1f}s) ===")
print(m_excl_bulk.summary())

# ----------------------------------------------------------------
# [3] Extensive margin (二値: その年に出願したか)
# ----------------------------------------------------------------
print(f"\n[3/3] Extensive-margin (binary filing indicator) specification ...", flush=True)
m_ext = pf.feols('extensive_margin ~ log_decayed_stock + log_grant_stock + relatedness_dynamic | applicant_id^field_id + app_year',
                  data=p2, vcov={'CRV1': 'applicant_id'})
print(f"\n=== [B] Extensive margin (LPM, {time.time()-t0:.1f}s) ===")
print(m_ext.summary())

print(f"\n完了 ({time.time()-t0:.1f}s total)")
print("\n解釈：[A]でbulk filingを除いても係数が頑健に正であれば、結果が")
print("手続的な分割出願の水増しだけで駆動されているわけではないと言える。")
print("[B]で二値アウトカムでも同じ方向・有意性が出れば、intensive margin")
print("(件数)の効果が単なる出願数の水増しでないことがさらに補強される。")
