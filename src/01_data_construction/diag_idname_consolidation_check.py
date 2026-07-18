# ================================================================
# diag04_idname_consolidation_check.py
# 診断用スクリプト（Google Colab）
# ----------------------------------------------------------------
# 目的：
#   01_build_variables_v2.py で検出されたM&Aスパイクの異常集中
#   （1991年・2000年・2010年に突出）が、真の企業合併・買収ではなく、
#   JPOによるapplicant_id（idname）の遡及的な名寄せ・統合処理に
#   よるものかどうかを検証する。
#
#   applicant.txt の 'name'（出願人名称）は今回のパイプラインでは
#   未抽出だったため、本スクリプトで別途取得し、スパイク企業年の
#   applicant_idに、過去〜現在にかけて複数の異なるnameが
#   紐づいていないかを確認する。
#
# 判定ロジック：
#   あるapplicant_id（idname）に対応するnameが、時系列で
#   2種類以上存在する場合 → 名寄せ・統合の可能性が高い
#   （表記揺れの範囲を超える大きな変化かどうかは目視確認が必要）
#
# 実行方法：
#   1. df_application_v2.pkl（前回保存済み）と
#      applicant_1990s.txt〜applicant_2020s.txt を /content/ に用意
#      （df_application_v2.pklは /content/drive/MyDrive/
#       patent_analysis_cache_v2/ に保存済みのはず）
#   2. exec(open('/content/diag04_idname_consolidation_check.py').read())
# ================================================================

import pandas as pd
import numpy as np
import os, gc, warnings
warnings.filterwarnings('ignore')

IIP_DIR   = '/content'
CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v2'
OUT_DIR   = '/content/diag04_results'
os.makedirs(OUT_DIR, exist_ok=True)

DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000

SPIKE_MIN_BASE = 3
SPIKE_RATIO    = 3.0
TOP_N_INSPECT  = 30   # 詳細表示する上位スパイク企業数

# ----------------------------------------------------------------
# CELL 1: 前回保存済みの df_application_v2.pkl を読み込み、
#         スパイク対象の applicant_id × app_year を再特定
# ----------------------------------------------------------------
print("[1/3] Loading df_application_v2.pkl and identifying spike applicant_ids ...")

ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v2.pkl')
pharma_ids = set(ap['app_id'].unique())

spike_events = ap.loc[ap['ma_spike_flag'] == True, ['applicant_id', 'app_year']].drop_duplicates()
spike_applicant_ids = set(spike_events['applicant_id'].unique())
print(f"  スパイク企業年: {len(spike_events):,} 件 / 対象applicant_id: {len(spike_applicant_ids):,} 件")

print("\n  スパイク年の分布（再確認）:")
print(spike_events['app_year'].value_counts().sort_index().to_string())

# ----------------------------------------------------------------
# CELL 2: applicant.txt から name 列を含めて再読み込み
#         （スパイク対象のapplicant_idに限定して軽量化）
# ----------------------------------------------------------------
print("\n[2/3] Loading applicant name history for spike applicant_ids ...")

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
        mask = (chunk['seq'] == 1) & chunk['ida'].isin(pharma_ids) & chunk['idname'].isin(spike_applicant_ids)
        filtered = chunk[mask].copy()
        n += len(filtered)
        name_frames.append(filtered)
    print(f"  ✓ applicant_{dec}.txt  → {n:,} matched rows")

names = pd.concat(name_frames, ignore_index=True)
del name_frames; gc.collect()
names = names.rename(columns={'ida': 'app_id', 'idname': 'applicant_id'})
names = names.merge(ap[['app_id', 'app_year']].drop_duplicates('app_id'), on='app_id', how='left')

# ----------------------------------------------------------------
# CELL 3: applicant_id ごとの name 種類数を集計
# ----------------------------------------------------------------
print("\n[3/3] Checking name diversity per applicant_id ...")

name_diversity = (names.groupby('applicant_id')
                        .agg(n_distinct_names=('name', 'nunique'),
                             names_list=('name', lambda s: sorted(set(s.dropna()))),
                             year_min=('app_year', 'min'),
                             year_max=('app_year', 'max'),
                             n_apps=('app_id', 'count'))
                        .reset_index()
                        .sort_values('n_distinct_names', ascending=False))

print(f"\n複数nameを持つapplicant_id: {(name_diversity['n_distinct_names'] > 1).sum():,} / "
      f"{len(name_diversity):,} "
      f"({(name_diversity['n_distinct_names'] > 1).mean():.1%})")

print(f"\n=== name種類数トップ{TOP_N_INSPECT}（名寄せ疑いが強い順） ===")
for _, row in name_diversity.head(TOP_N_INSPECT).iterrows():
    print(f"\napplicant_id={row['applicant_id']}  "
          f"n_distinct_names={row['n_distinct_names']}  "
          f"n_apps={row['n_apps']}  year_range={row['year_min']}-{row['year_max']}")
    for nm in row['names_list'][:5]:
        print(f"    - {nm}")
    if len(row['names_list']) > 5:
        print(f"    ... ほか{len(row['names_list'])-5}件")

# スパイク年ごとに、その年時点でのname多様性を突合
spike_check = spike_events.merge(
    name_diversity[['applicant_id', 'n_distinct_names']], on='applicant_id', how='left')
spike_with_multiname = spike_check['n_distinct_names'] > 1
print(f"\n\nスパイク企業年のうち、applicant_idに複数nameが紐づくもの: "
      f"{spike_with_multiname.sum():,} / {len(spike_check):,} "
      f"({spike_with_multiname.mean():.1%})")
print("→ この割合が高ければ、スパイクの正体は名寄せ・統合である可能性が高い。")
print("  低ければ、真の急増出願（またはIPC再分類など別の要因）を疑う必要がある。")

# ----------------------------------------------------------------
# 保存
# ----------------------------------------------------------------
name_diversity.to_csv(f'{OUT_DIR}/diag_name_diversity_by_applicant.csv', index=False)
spike_check.to_csv(f'{OUT_DIR}/diag_spike_vs_namediversity.csv', index=False)
print(f"\n全診断結果を {OUT_DIR} に保存しました。")
