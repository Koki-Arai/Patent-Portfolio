# ================================================================
# 02_apply_identity_filter_v3.py
# 出願人ID整合性フィルタの事後適用（Google Colab）
# ----------------------------------------------------------------
# 01_build_variables_v3.py実行時にdiag_idname_integrity_full.pklが
# 見つからず除外がスキップされた場合、958秒かけてパイプライン全体を
# 再実行する必要はない。ストック・relatedness等の変数は企業ごとに
# 独立して計算されているため、対象41 applicant_idの行を事後的に
# 取り除くだけで、他企業の値には影響を与えずに正しい除外サンプルが
# 得られる（relatedness matrixへの影響も41/約22万社と軽微）。
#
# 実行方法：
#   1. diag_idname_integrity_full.pkl を /content/ にアップロード
#   2. exec(open('/content/02_apply_identity_filter_v3.py').read())
# ================================================================

import pandas as pd
import os

CACHE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
DIAG_PATH = '/content/diag_idname_integrity_full.pkl'

print("[1/2] Loading v3 outputs and flagged applicant_id list ...")
ap = pd.read_pickle(f'{CACHE_DIR}/df_application_v3.pkl')
panel = pd.read_pickle(f'{CACHE_DIR}/panel_v3.pkl')

if not os.path.exists(DIAG_PATH):
    raise FileNotFoundError(
        "diag_idname_integrity_full.pkl が見つかりません。"
        "前回の診断(diag05)出力をDriveから/content/にコピーしてください。")

multi = pd.read_pickle(DIAG_PATH)
flagged = set(multi.loc[multi['likely_unrelated_reuse'], 'applicant_id'])
print(f"  除外対象applicant_id: {len(flagged):,} 件")

print("\n[2/2] Applying exclusion ...")
n_before_ap, n_before_panel = len(ap), len(panel)

ap['excluded_idname_conflict'] = ap['applicant_id'].isin(flagged)
ap_clean = ap[~ap['excluded_idname_conflict']].copy()

panel['excluded_idname_conflict'] = panel['applicant_id'].isin(flagged)
panel_clean = panel[~panel['excluded_idname_conflict']].copy()

print(f"  ap:    {n_before_ap:,} -> {len(ap_clean):,} "
      f"({n_before_ap - len(ap_clean):,}件除外, {(n_before_ap-len(ap_clean))/n_before_ap:.2%})")
print(f"  panel: {n_before_panel:,} -> {len(panel_clean):,} "
      f"({n_before_panel - len(panel_clean):,}行除外)")

ap_clean.to_pickle(f'{CACHE_DIR}/df_application_v3_clean.pkl')
panel_clean.to_pickle(f'{CACHE_DIR}/panel_v3_clean.pkl')

print(f"\n保存完了 -> {CACHE_DIR}/df_application_v3_clean.pkl")
print(f"        -> {CACHE_DIR}/panel_v3_clean.pkl")
print("\n以降のBlock1A〜3推定は、_clean版のファイルを使用してください。")
