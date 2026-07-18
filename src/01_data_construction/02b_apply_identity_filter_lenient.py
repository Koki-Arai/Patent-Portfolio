# ================================================================
# 02b_apply_identity_filter_lenient.py
# 出願人ID整合性フィルタ — 緩和版（Google Colab）
# ----------------------------------------------------------------
# 従来版(02_apply_identity_filter_v3.py)は、名称類似度(difflib)が
# 閾値0.5未満の41 applicant_idを一律除外する保守的な方式だった。
# この方式は、ゼネカ→アストラゼネカ、スミスクライン・ビーチャム→
# グラクソスミスクラインのような正当な合併・改称ケースも
# 巻き添えで除外してしまうという限界があった。
#
# 本スクリプトは、法人形態語（株式会社・コーポレーション等）を
# 除去したうえで正規化した名称同士の「最長共通部分文字列」の
# 一致率を計算し、一致率が高い（=旧社名の一部が新社名に
# 引き継がれている）ケースを「合理的に同一企業とみなせる」として
# 復元する。
#
# 【原理的な限界】
#   社名が完全に変わる買収（例：ザイモジェネティクス→ノボ
#   ノルディスク、ターゲット・セラピューティクス→ボストン・
#   サイエンティフィック）は文字列としては一致率ゼロになるため、
#   この手法では正当な合併でも捉えられない。したがって、
#   一致率が閾値未満のケースは、実際には正当な合併である
#   可能性を残したまま、引き続き保守的に除外する。
#
# 実行方法：
#   1. df_application_v3.pkl（除外前のフル版）, panel_v3.pkl,
#      diag_idname_integrity_full.pkl を用意
#   2. exec(open('/content/02b_apply_identity_filter_lenient.py').read())
#
# 出力：
#   df_application_v3_clean_lenient.pkl / panel_v3_clean_lenient.pkl
#   （保守版 df_application_v3_clean.pkl とは別ファイルとして保存。
#    両方の結果を比較するロバストネスチェックとして使うことを推奨）
# ================================================================

import pandas as pd
import numpy as np
import os

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
DIAG_PATH = '/content/diag_idname_integrity_full.pkl'
LCS_RATIO_THRESHOLD = 0.30  # ★要検討：この値以上を「同一企業」とみなす

CORP_SUFFIXES = [
    '株式会社', '有限会社', '合同会社', 'コーポレーション', 'コーポレイション',
    'インコーポレイテッド', 'インコーポレーテッド', 'インコーポレイティド', 'インク',
    'カンパニー', 'リミテッド', 'アクチエンゲゼルシャフト', 'アクチエンゲゼルシヤフト',
    'アクチボラグ', 'ゲゼルシャフト・ミット・ベシュレンクテル・ハフツング',
    'ゲゼルシヤフト・ミツト・ベシユレンクテル・ハフツング', 'ゲーエムベーハー',
    'エルエルシー', 'エル・エル・シー', 'ピーティワイ', 'ソシエタ・ペル・アツィオーニ',
    'エイ／エス',
]
PUNCT = ['\u3000', ' ', '・', '，', ',', '、', '．', '.', '－', '-', '（', '）', '(', ')']


def normalize(name):
    s = name
    for p in PUNCT:
        s = s.replace(p, '')
    for suf in CORP_SUFFIXES:
        s = s.replace(suf.replace('・', ''), '')
    return s


def lcs_len(a, b):
    """最長共通部分文字列（連続）の長さ"""
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    best = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                best = max(best, m[i][j])
    return best


print("[1/3] Loading data and computing name-similarity reclassification ...")
multi = pd.read_pickle(DIAG_PATH)
flagged_orig = multi[multi['likely_unrelated_reuse']].copy()

records = []
for _, row in flagged_orig.iterrows():
    names = row['names_list']
    if len(names) != 2:
        records.append({'applicant_id': row['applicant_id'], 'lcs_ratio': np.nan,
                         'reclassified_as_same_firm': False})
        continue
    n1, n2 = normalize(names[0]), normalize(names[1])
    L = lcs_len(n1, n2)
    ratio = L / min(len(n1), len(n2)) if min(len(n1), len(n2)) > 0 else 0
    records.append({'applicant_id': row['applicant_id'], 'name1': names[0], 'name2': names[1],
                     'lcs_ratio': round(ratio, 3),
                     'reclassified_as_same_firm': ratio >= LCS_RATIO_THRESHOLD})

reclass = pd.DataFrame(records)
still_excluded = set(reclass.loc[~reclass['reclassified_as_same_firm'], 'applicant_id'])
rescued = reclass.loc[reclass['reclassified_as_same_firm']]

print(f"  元の除外対象: {len(flagged_orig)} applicant_id")
print(f"  同一企業として復元: {len(rescued)} applicant_id")
print(f"  引き続き除外: {len(still_excluded)} applicant_id")
print("\n  復元されたapplicant_id:")
for _, r in rescued.iterrows():
    print(f"    {r['applicant_id']}  (lcs_ratio={r['lcs_ratio']:.3f})  "
          f"{r.get('name1','')} <-> {r.get('name2','')}")

reclass.to_csv(f'{CACHE_DIR}/idname_reclassification_lenient.csv', index=False)

print("\n[2/3] Applying lenient exclusion (only still_excluded ids removed) ...")
ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3.pkl')
panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3.pkl')

n_before_ap, n_before_panel = len(ap), len(panel)
ap['excluded_idname_conflict_lenient'] = ap['applicant_id'].isin(still_excluded)
ap_lenient = ap[~ap['excluded_idname_conflict_lenient']].copy()
panel['excluded_idname_conflict_lenient'] = panel['applicant_id'].isin(still_excluded)
panel_lenient = panel[~panel['excluded_idname_conflict_lenient']].copy()

print(f"  ap:    {n_before_ap:,} -> {len(ap_lenient):,} "
      f"({n_before_ap - len(ap_lenient):,}件除外, {(n_before_ap-len(ap_lenient))/n_before_ap:.2%})")
print(f"  panel: {n_before_panel:,} -> {len(panel_lenient):,}")

print("\n[3/3] Saving ...")
ap_lenient.to_pickle(f'{CACHE_DIR}/df_application_v3_clean_lenient.pkl')
panel_lenient.to_pickle(f'{CACHE_DIR}/panel_v3_clean_lenient.pkl')
print(f"  -> {CACHE_DIR}/df_application_v3_clean_lenient.pkl")
print(f"  -> {CACHE_DIR}/panel_v3_clean_lenient.pkl")
print("\n注意: own_portfolio_hhi・relatedness_dynamic・patent_stock等は、")
print("復元された7社について、01_build_variables_v3.py実行時点の")
print("(保守的除外込みの)定義のままです。真に厳密には、これらの企業を")
print("復元した状態で変数構築からやり直す必要がありますが、影響は")
print("7/221,375社と極めて限定的なため、まずはサンプル差し替えの")
print("影響のみを頑健性チェックとして確認することを推奨します。")
