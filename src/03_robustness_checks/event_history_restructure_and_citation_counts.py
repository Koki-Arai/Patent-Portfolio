# ================================================================
# model3_restructure_and_model2_counts.py
# Model3の再構成（独立イベント史モデル）とModel2の引用数分解
# （Google Colab）
# ----------------------------------------------------------------
# 【Model3の再構成】
#   reason22記録のある出願の14.9%が後にgrantされているため、
#   grantとreason22を"どちらか一方しか起こらない終端競合リスク"
#   として扱うのは不正確である（外部査読の指摘）。
#   以下の3モデルに再構成する：
#     (a) Grant hazard（reason22の有無と無関係に、grant自体をイベントとする）
#     (b) Reason22 hazard（grantの有無と無関係に、reason22自体をイベントとする）
#     (c) reason22発生後のgrantへの移行（reason22を経験した出願に限定し、
#         その後のgrant移行をイベントとする——新規分析）
#
# 【Model2の引用数分解】
#   自己引用の有無（二値）だけでなく、self_count／external_count／
#   total_countを別々に分析し、relatednessが「自己引用を減らす」のか
#   「外部引用を増やす」のか（＝prior-art密度の増加）を区別する。
#
# 実行時間の目安：Cox系はfirm-clustered SEで各モデル4〜6分程度。
#
# 実行方法：
#   1. df_application_v3_clean.pkl, cc_1990s.txt〜cc_2020s.txt を用意
#   2. pip install lifelines pyfixest
#   3. exec(open('/content/model3_restructure_and_model2_counts.py').read())
# ================================================================

import pandas as pd
import numpy as np
import time
from lifelines import CoxPHFitter
import pyfixest as pf

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
IIP_DIR = '/content'
DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000
REJECT_REASON_CTX = [19, 89]

t0 = time.time()
print("=" * 70)
print("Model3再構成 + Model2引用数分解 — 開始")
print("=" * 70)

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')
ap['log_decayed_stock'] = np.log1p(ap['decayed_stock_d10'])

# ----------------------------------------------------------------
# PART A: Model3の再構成
# ----------------------------------------------------------------
print(f"\n[A] Model3: separate event-history models ...", flush=True)

adate, rdate, r22date = ap['adate_dt'], ap['rdate_dt'], ap['reason22_first_date']
censoring_time = ap['censoring_time']

ap['event_grant_simple'] = ap['granted'].astype(int)
ap['duration_grant_simple'] = np.where(ap['event_grant_simple'] == 1,
    (rdate - adate).dt.days, (censoring_time - adate).dt.days)

ap['event_r22_simple'] = r22date.notna().astype(int)
ap['duration_r22_simple'] = np.where(ap['event_r22_simple'] == 1,
    (r22date - adate).dt.days, (censoring_time - adate).dt.days)

cols = ['log_decayed_stock', 'relatedness_dynamic', 'own_portfolio_hhi',
        'corporate_flag', 'post_2001_reform', 'field_id', 'applicant_id']

results3 = {}
for label, dur, ev in [('grant_simple', 'duration_grant_simple', 'event_grant_simple'),
                        ('reason22_simple', 'duration_r22_simple', 'event_r22_simple')]:
    sub = ap[[dur, ev] + cols].dropna()
    sub = sub[sub[dur] > 0].copy()
    sub['corporate_flag'] = sub['corporate_flag'].astype(int)
    ts = time.time()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=dur, event_col=ev,
            strata=['field_id', 'post_2001_reform', 'corporate_flag'], cluster_col='applicant_id')
    print(f"\n--- {label} (n={len(sub):,}, event_rate={sub[ev].mean():.1%}, {time.time()-ts:.1f}s) ---")
    print(cph.summary[['coef', 'se(coef)', 'p', 'exp(coef)']])
    results3[label] = cph
    cph.summary.to_csv(f'{CACHE_DIR}/model3_{label}_results.csv')

# --- (c) reason22発生後のgrantへの移行 ---
print(f"\n--- (c) reason22-to-grant transition ---", flush=True)
sub_r22 = ap[ap['reason22_first_date'].notna()].copy()
sub_r22['event_subsequent_grant'] = (sub_r22['granted'] &
    (sub_r22['rdate_dt'] > sub_r22['reason22_first_date'])).astype(int)
sub_r22['duration_from_r22'] = np.where(sub_r22['event_subsequent_grant'] == 1,
    (sub_r22['rdate_dt'] - sub_r22['reason22_first_date']).dt.days,
    (sub_r22['censoring_time'] - sub_r22['reason22_first_date']).dt.days)
sub_r22 = sub_r22[sub_r22['duration_from_r22'] > 0]
sub2 = sub_r22[['duration_from_r22', 'event_subsequent_grant'] + cols].dropna()
sub2['corporate_flag'] = sub2['corporate_flag'].astype(int)
ts = time.time()
cph3 = CoxPHFitter()
cph3.fit(sub2, duration_col='duration_from_r22', event_col='event_subsequent_grant',
         strata=['field_id', 'post_2001_reform', 'corporate_flag'], cluster_col='applicant_id')
print(f"(n={len(sub2):,}, event_rate={sub2['event_subsequent_grant'].mean():.1%}, {time.time()-ts:.1f}s)")
print(cph3.summary[['coef', 'se(coef)', 'p', 'exp(coef)']])
cph3.summary.to_csv(f'{CACHE_DIR}/model3_reason22_to_grant_results.csv')

# ----------------------------------------------------------------
# PART B: Model2 self/external citation count decomposition
# ----------------------------------------------------------------
print(f"\n[B] Model2: self- vs external-citation count decomposition ...", flush=True)

pharma_ids = set(ap['app_id'])
cc_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/cc_{dec}.txt'
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['citing', 'cited', 'reason'], dtype=str,
                              chunksize=CHUNKSIZE, low_memory=False):
        chunk['reason'] = pd.to_numeric(chunk['reason'], errors='coerce')
        mask = chunk['reason'].isin(REJECT_REASON_CTX) & chunk['citing'].isin(pharma_ids)
        cc_frames.append(chunk[mask])
cc = pd.concat(cc_frames, ignore_index=True)

ap_id_appl = ap[['app_id', 'applicant_id']].drop_duplicates('app_id')
cc = (cc.merge(ap_id_appl.rename(columns={'app_id': 'cited', 'applicant_id': 'cited_appl'}), on='cited', how='left')
        .merge(ap_id_appl.rename(columns={'app_id': 'citing', 'applicant_id': 'citing_appl'}), on='citing', how='left'))
cc['is_self'] = (cc['cited_appl'] == cc['citing_appl']).astype(int)

counts = (cc.groupby('citing')
            .agg(self_count=('is_self', 'sum'), total_count=('is_self', 'count'))
            .reset_index().rename(columns={'citing': 'app_id'}))
counts['external_count'] = counts['total_count'] - counts['self_count']

ap2 = ap.merge(counts, on='app_id', how='left')
sub_ctx = ap2.dropna(subset=['self_count', 'relatedness_dynamic', 'own_portfolio_hhi', 'n_active_fields'])
print(f"  rejection-context sample: {len(sub_ctx):,}")

fml_vars = 'log_decayed_stock + relatedness_dynamic + own_portfolio_hhi + n_active_fields'
for outcome in ['self_count', 'external_count', 'total_count']:
    m = pf.fepois(f'{outcome} ~ {fml_vars} | applicant_id + field_id + app_year',
                  data=sub_ctx, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
    print(f"\n--- {outcome} ---")
    print(m.summary())

print(f"\n完了 ({time.time()-t0:.1f}s total)")
print("\n解釈：relatednessがexternal_countを増やし、self_countを増やさない")
print("（または減らす）なら、『関連性の高い出願は外部prior artの密度が")
print("高い領域に位置する』という解釈が定量的に裏付けられる。")
