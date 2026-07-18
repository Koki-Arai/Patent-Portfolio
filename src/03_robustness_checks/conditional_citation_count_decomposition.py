# ================================================================
# conditional_citation_count_decomposition.py
# RQ2条件付き引用数分解（rejection-context発生サンプルに限定）
# ----------------------------------------------------------------
# Table 2 Part 2と厳密に同じ条件付け(rejection_ctx_context_occurred==1)
# に限定した上で、self/external/total citation countを分解する。
# 全サンプル版(unconditional)では既にself_count負・external_count
# 非有意・total_count境界的、という結果を得ているが、RQ2の主張は
# "citation contextが記録された条件下"であるため、この条件付き版が
# 本文の主張により直接対応する。
#
# 実行方法：
#   1. df_application_v3_clean.pkl, cc_1990s.txt〜cc_2020s.txt を用意
#   2. pip install pyfixest
#   3. exec(open('/content/conditional_citation_count_decomposition.py').read())
# ================================================================

import pandas as pd
import numpy as np
import time
import pyfixest as pf

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
IIP_DIR = '/content'
DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000
REJECT_REASON_CTX = [19, 89]

t0 = time.time()
print("=" * 70)
print("条件付き引用数分解（rejection-context発生サンプル限定）— 開始")
print("=" * 70)

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')
ap['log_decayed_stock'] = np.log1p(ap['decayed_stock_d10'])

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

# --- 重要な変更点：Table 2 Part 2と同じ条件付けに限定 ---
# rejection_ctx_context_occurred==1 のサンプルのみを対象とする。
# context未発生の出願はcitation countが構造的に0/欠損であり、
# unconditional版ではこれらが分析に混入していた。
sub_cond = ap2[ap2['rejection_ctx_context_occurred'] == 1].copy()
sub_cond[['self_count', 'external_count', 'total_count']] = sub_cond[
    ['self_count', 'external_count', 'total_count']].fillna(0)

sub_ctx = sub_cond.dropna(subset=['relatedness_dynamic', 'own_portfolio_hhi', 'n_active_fields'])
print(f"  conditional (context-recorded) sample: {len(sub_ctx):,}")
print(f"  self_count>0 の割合: {(sub_ctx['self_count']>0).mean():.1%}")

fml_vars = 'log_decayed_stock + relatedness_dynamic + own_portfolio_hhi + n_active_fields'
for outcome in ['self_count', 'external_count', 'total_count']:
    m = pf.fepois(f'{outcome} ~ {fml_vars} | applicant_id + field_id + app_year',
                  data=sub_ctx, vcov={'CRV1': 'applicant_id'}, fixef_maxiter=3000)
    print(f"\n--- {outcome} (conditional, n={len(sub_ctx):,}) ---")
    print(m.summary())

print(f"\n完了 ({time.time()-t0:.1f}s total)")
print("\n解釈：unconditional版(self=-0.449, external=-0.035 n.s., total=-0.082境界的)と")
print("この条件付き版の符号・有意性のパターンが一致すれば、RQ2の主張が")
print("『citation context発生』と『context内引用数』の混同ではなかったことが")
print("より直接的に確認できる。")
