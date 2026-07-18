# ================================================================
# diag05_idname_integrity_full.py
# 診断用スクリプト（Google Colab）
# ----------------------------------------------------------------
# 目的：
#   diag04で判明した「applicant_id（idname）が無関係な複数企業に
#   再利用されている」問題が、スパイク検出企業（797件）に限らず
#   医薬品出願全体（applicant_id 数万件規模）でどの程度の規模で
#   発生しているかを検証する。
#
#   単純な「name種類数>1」だけでは、
#     - 正当な社名変更・合併（例：ゼネカ→アストラゼネカ）
#     - 表記揺れ（全角/半角、読点の違い）
#   と、
#     - 無関係な別企業によるidname使い回し
#       （例：信越化学工業 vs 宇部興産——両社とも現存する別会社）
#   を区別できないため、文字列類似度によるヒューリスティックで
#   自動判定を行う。
#
# 判定ロジック：
#   同一applicant_idに紐づく全nameペアについて、
#   difflib.SequenceMatcherによる類似度比を計算し、
#   最小類似度が閾値未満の場合を「likely_unrelated_reuse」と
#   フラグする（要目視確認。あくまでスクリーニング用）。
#
# 実行方法：
#   1. df_application_v2.pkl（Drive上）と
#      applicant_1990s.txt〜applicant_2020s.txt を用意
#   2. exec(open('/content/diag05_idname_integrity_full.py').read())
#
# 注意：
#   全applicant_id×全name履歴を読み込むため、diag04より重い処理です。
#   Colabのメモリに応じてCHUNKSIZEを調整してください。
# ================================================================

import pandas as pd
import numpy as np
import os, gc, warnings
from difflib import SequenceMatcher
warnings.filterwarnings('ignore')

IIP_DIR   = '/content'
CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v2'
OUT_DIR   = '/content/diag05_results'
os.makedirs(OUT_DIR, exist_ok=True)

DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000

SIMILARITY_THRESHOLD = 0.5   # ★要調整：これ未満を「無関係企業の疑い」とする
TOP_N_INSPECT = 40

# ----------------------------------------------------------------
# CELL 1: 対象となる全applicant_idを特定（医薬品出願全体）
# ----------------------------------------------------------------
print("[1/3] Loading df_application_v2.pkl and identifying all applicant_ids ...")

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v2.pkl')
pharma_ids = set(ap['app_id'].unique())
all_applicant_ids = set(ap['applicant_id'].unique())
print(f"  医薬品出願総数: {len(pharma_ids):,}")
print(f"  対象applicant_id総数: {len(all_applicant_ids):,}")

# ----------------------------------------------------------------
# CELL 2: applicant.txt から全件の name を読み込み
# ----------------------------------------------------------------
print("\n[2/3] Loading full applicant name history (all pharma applicant_ids) ...")

name_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/applicant_{dec}.txt'
    if not os.path.exists(path):
        print(f"  ! {path} not found — skipping"); continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida', 'seq', 'idname', 'name'],
                              dtype=str, chunksize=CHUNKSIZE, low_memory=False):
        chunk['seq'] = pd.to_numeric(chunk['seq'], errors='coerce')
        mask = (chunk['seq'] == 1) & chunk['ida'].isin(pharma_ids)
        filtered = chunk[mask].copy()
        n += len(filtered)
        name_frames.append(filtered)
    print(f"  ✓ applicant_{dec}.txt  → {n:,} matched rows")

names = pd.concat(name_frames, ignore_index=True)
del name_frames; gc.collect()
names = names.rename(columns={'ida': 'app_id', 'idname': 'applicant_id'})
names = names.merge(ap[['app_id', 'app_year']].drop_duplicates('app_id'), on='app_id', how='left')
names['applicant_id'] = names['applicant_id'].fillna(names['app_id'])

# ----------------------------------------------------------------
# CELL 3: name多様性の集計 + 類似度ヒューリスティック
# ----------------------------------------------------------------
print("\n[3/3] Computing name diversity and similarity heuristic for ALL applicant_ids ...")

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

records = []
for aid, g in names.groupby('applicant_id'):
    distinct_names = sorted(set(g['name'].dropna()))
    n_names = len(distinct_names)
    if n_names <= 1:
        continue
    # 全ペアの最小類似度
    min_sim = 1.0
    for i in range(len(distinct_names)):
        for j in range(i + 1, len(distinct_names)):
            s = sim(distinct_names[i], distinct_names[j])
            min_sim = min(min_sim, s)
    records.append({
        'applicant_id': aid,
        'n_distinct_names': n_names,
        'min_pairwise_similarity': min_sim,
        'names_list': distinct_names,
        'n_apps': len(g),
        'year_min': g['app_year'].min(),
        'year_max': g['app_year'].max(),
        'likely_unrelated_reuse': min_sim < SIMILARITY_THRESHOLD,
    })

if not records:
    print("\n複数nameを持つapplicant_idは見つかりませんでした"
          "（対象decadeが限定的な場合はこの結果になり得ます。全decadeで再実行してください）。")
    multi = pd.DataFrame(columns=['applicant_id', 'n_distinct_names', 'min_pairwise_similarity',
                                   'names_list', 'n_apps', 'year_min', 'year_max',
                                   'likely_unrelated_reuse'])
else:
    multi = pd.DataFrame(records).sort_values('min_pairwise_similarity')

print(f"\n複数nameを持つapplicant_id（全体）: {len(multi):,} / {len(all_applicant_ids):,} "
      f"({len(multi)/len(all_applicant_ids):.2%})")

n_reuse = multi['likely_unrelated_reuse'].sum()
n_reuse_apps = multi.loc[multi['likely_unrelated_reuse'], 'n_apps'].sum()
print(f"\n類似度{SIMILARITY_THRESHOLD}未満（無関係企業の疑い）: {n_reuse:,} applicant_id "
      f"（該当する出願件数合計: {n_reuse_apps:,}件、"
      f"全体の{n_reuse_apps/len(pharma_ids):.2%}）")

legit = multi[~multi['likely_unrelated_reuse']]
print(f"類似度{SIMILARITY_THRESHOLD}以上（正当な社名変更・表記揺れの可能性が高い）: "
      f"{len(legit):,} applicant_id")

print(f"\n=== 「無関係企業の疑い」上位{TOP_N_INSPECT}件（類似度が低い順） ===")
suspect = multi[multi['likely_unrelated_reuse']].head(TOP_N_INSPECT)
for _, row in suspect.iterrows():
    print(f"\napplicant_id={row['applicant_id']}  "
          f"min_sim={row['min_pairwise_similarity']:.2f}  "
          f"n_apps={row['n_apps']}  year_range={row['year_min']}-{row['year_max']}")
    for nm in row['names_list']:
        print(f"    - {nm}")

# ----------------------------------------------------------------
# 保存
# ----------------------------------------------------------------
multi.drop(columns=['names_list']).to_csv(f'{OUT_DIR}/diag_idname_integrity_summary.csv', index=False)
multi.to_pickle(f'{OUT_DIR}/diag_idname_integrity_full.pkl')  # names_list含む完全版

print(f"\n全診断結果を {OUT_DIR} に保存しました。")
print("次のステップ: likely_unrelated_reuse=Trueのapplicant_idについて、")
print("             出願年で分割して別々のapplicant_idとして再割当てするか、")
print("             主分析から除外するかを決定してください。")
