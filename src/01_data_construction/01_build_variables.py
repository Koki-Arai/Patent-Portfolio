# ================================================================
# 01_build_variables_v3.py
# Patent Portfolio Dynamics — 変数構築 完全版（バグ修正済み）
# ----------------------------------------------------------------
# 2件の外部査読（index 10, 11相当）で指摘された全バグを修正した
# マスタースクリプト。v2からの主な変更点：
#
#   [バグ#1・#2 修正] patent_stock / grant_stock / decayed_stock を
#     group-safeな transform() で、かつ完全ゼロ埋め済みの
#     firm×field×year グリッド上で直接計算する。
#     旧コードは groupby().cumsum().shift(1) のshift(1)が
#     データフレーム全体に適用され、企業×分野グループの先頭行に
#     直前グループの累積値が漏れ込んでいた（実データ検証で
#     全体の61.8%の行に影響、グループ先頭行の99.9996%が汚染）。
#
#   [バグ#3 修正] own_portfolio_hhi / portfolio_entropy を、
#     「当年に出願した分野」だけでなく、企業が過去に活動した
#     全分野（完全ゼロ埋めグリッド）から計算する。
#
#   [バグ#4 修正] relatedness matrixを全期間一括ではなく、
#     各年ごとに「その年より前のデータだけ」を使って構築する
#     （year-expanding window、未来情報の混入を排除）。
#
#   [バグ#8 修正] Block2の自己引用フラグを二段階に分離：
#     (a) その reason 文脈自体が発生したか
#     (b) 文脈が発生した出願に限定した場合の自己引用有無
#     旧コードは「文脈なし」と「文脈ありだが自己引用なし」を
#     同じ0として扱っていた。
#
#   [バグ#9・#10 修正] cc.txtからreason_dateを読み込み、
#     reason=22（拒絶査定関連）の実際の発生日を取得。
#     grant と reason22 を「competing risks」として明示的に分離し、
#     cause-specific hazard用のduration/eventペアを別々に構築。
#
#   [その他] sdate起点のduration列も保持（起点の選択問題への対応）。
#     出願人ID整合性フィルタ（既知の41件除外）を early stage で適用。
#
# 実行方法：
#   1. ap_*.txt, applicant_*.txt, cc_*.txt を /content/ にアップロード
#   2. diag_idname_integrity_full.pkl を /content/ に用意
#      （前回の診断で作成済み。Driveから取得）
#   3. exec(open('/content/01_build_variables_v3.py').read())
# ================================================================

import pandas as pd
import numpy as np
import os, gc, warnings, time
warnings.filterwarnings('ignore')

IIP_DIR = '/content'
SAVE_DIR = '/content/drive/MyDrive/patent_analysis_cache_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

DECADES = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000
PHARMA_IPC = ['A61', 'C07', 'C12']

REJECT_REASON_CTX = [19, 89]
REJECT_DECISION_CTX = [22]
GRANT_CTX = [31]
AMENDMENT_CTX = [75]
PREAPPEAL_CTX = [93]
VALID_REASONS = REJECT_REASON_CTX + REJECT_DECISION_CTX + GRANT_CTX + AMENDMENT_CTX + PREAPPEAL_CTX

SAMPLE_END_YEAR = 2021
DATA_CUTOFF = pd.Timestamp('2023-07-10')
REFORM_DATE = pd.Timestamp('2001-10-01')
CENSOR_CAP_YEARS = 15
DECAY_DELTA = 0.10
MA_SPIKE_MIN_BASE = 3
MA_SPIKE_RATIO = 3.0

t0 = time.time()
print("=" * 70)
print("01_build_variables_v3.py — 開始（バグ修正版）")
print("=" * 70)

# ----------------------------------------------------------------
# STEP 1: ap（医薬品IPC絞り込み）+ applicant（kohokan含む）
# ----------------------------------------------------------------
print(f"\n[1/8] Loading ap and applicant ...", flush=True)

ap_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/ap_{dec}.txt'
    if not os.path.exists(path):
        print(f"  ! {path} not found — skipping"); continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida', 'adate', 'sdate', 'idr', 'rdate',
                                       'class1', 'group1', 'claim1', 'claim2', 'claim3'],
                              dtype=str, chunksize=CHUNKSIZE, low_memory=False):
        mask = chunk['class1'].str[:3].isin(PHARMA_IPC)
        ap_frames.append(chunk[mask])
        n += mask.sum()
    print(f"  ✓ ap_{dec}.txt -> {n:,} rows")

ap = pd.concat(ap_frames, ignore_index=True); del ap_frames; gc.collect()
ap = ap.rename(columns={'ida': 'app_id'})
ap['adate_dt'] = pd.to_datetime(ap['adate'], errors='coerce')
ap['sdate_dt'] = pd.to_datetime(ap['sdate'], errors='coerce')
ap['rdate_dt'] = pd.to_datetime(ap['rdate'], errors='coerce')
ap['app_year'] = ap['adate_dt'].dt.year
ap = ap.dropna(subset=['app_year']); ap['app_year'] = ap['app_year'].astype(int)
ap['field_id'] = ap['class1']
ap['granted'] = ap['idr'].notna() & (ap['idr'].astype(str).str.strip() != '')
for c in ['claim1', 'claim2', 'claim3']:
    ap[c] = pd.to_numeric(ap[c], errors='coerce')
ap = ap[(ap['claim1'].isna() | (ap['claim1'] >= 0)) & (ap['claim3'].isna() | (ap['claim3'] >= 0))]

pharma_ids = set(ap['app_id'])
appl_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/applicant_{dec}.txt'
    if not os.path.exists(path):
        continue
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida', 'seq', 'idname', 'kohokan'], dtype=str,
                              chunksize=CHUNKSIZE, low_memory=False):
        chunk['seq'] = pd.to_numeric(chunk['seq'], errors='coerce')
        mask = (chunk['seq'] == 1) & chunk['ida'].isin(pharma_ids)
        appl_frames.append(chunk[mask][['ida', 'idname', 'kohokan']])
appl = pd.concat(appl_frames, ignore_index=True); del appl_frames; gc.collect()
appl = appl.rename(columns={'ida': 'app_id', 'idname': 'applicant_id'}).drop_duplicates('app_id')
appl['applicant_id'] = appl['applicant_id'].fillna(appl['app_id'])
appl['kohokan'] = pd.to_numeric(appl['kohokan'], errors='coerce')
appl['corporate_flag'] = (appl['kohokan'] == 2)
ap = ap.merge(appl, on='app_id', how='left')
ap['applicant_id'] = ap['applicant_id'].fillna(ap['app_id'])
ap['corporate_flag'] = ap['corporate_flag'].fillna(False)
print(f"  ap total: {len(ap):,} rows  ({time.time()-t0:.1f}s)", flush=True)

# ----------------------------------------------------------------
# STEP 2: applicant ID整合性フィルタ（既知の除外リストを適用）
# ----------------------------------------------------------------
print(f"\n[2/8] Applying applicant-identity filter ...", flush=True)

idname_path = f'{IIP_DIR}/diag_idname_integrity_full.pkl'
if os.path.exists(idname_path):
    multi = pd.read_pickle(idname_path)
    flagged = set(multi.loc[multi['likely_unrelated_reuse'], 'applicant_id'])
    n_before = len(ap)
    ap['excluded_idname_conflict'] = ap['applicant_id'].isin(flagged)
    ap = ap[~ap['excluded_idname_conflict']].copy()
    print(f"  excluded {n_before - len(ap):,} rows ({len(flagged)} applicant_ids)")
else:
    print("  ! diag_idname_integrity_full.pkl not found — skipping exclusion "
          "(re-run diag05 first if this is unexpected)")
    ap['excluded_idname_conflict'] = False

pharma_ids = set(ap['app_id'])

# ----------------------------------------------------------------
# STEP 3: cc（reason_date含む）— Block2二段階変数 + reason22実イベント日
# ----------------------------------------------------------------
print(f"\n[3/8] Loading cc with reason_date ...", flush=True)

cc_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/cc_{dec}.txt'
    if not os.path.exists(path):
        continue
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['citing', 'cited', 'reason', 'reason_date'], dtype=str,
                              chunksize=CHUNKSIZE, low_memory=False):
        chunk['reason'] = pd.to_numeric(chunk['reason'], errors='coerce')
        mask = chunk['reason'].isin(VALID_REASONS) & chunk['citing'].isin(pharma_ids)
        cc_frames.append(chunk[mask])
cc = pd.concat(cc_frames, ignore_index=True); del cc_frames; gc.collect()
cc['reason_date_dt'] = pd.to_datetime(cc['reason_date'], errors='coerce')
print(f"  cc: {len(cc):,} rows  ({time.time()-t0:.1f}s)", flush=True)

ap_id_appl = ap[['app_id', 'applicant_id']].drop_duplicates('app_id')
cc = (cc.merge(ap_id_appl.rename(columns={'app_id': 'cited', 'applicant_id': 'cited_appl'}), on='cited', how='left')
        .merge(ap_id_appl.rename(columns={'app_id': 'citing', 'applicant_id': 'citing_appl'}), on='citing', how='left'))
cc['is_self'] = (cc['cited_appl'] == cc['citing_appl']).astype(int)

r22 = (cc[cc['reason'].isin(REJECT_DECISION_CTX)].groupby('citing')['reason_date_dt'].min()
       .reset_index().rename(columns={'citing': 'app_id', 'reason_date_dt': 'reason22_first_date'}))


def context_two_stage(reason_list, label):
    sub = cc[cc['reason'].isin(reason_list)]
    g = (sub.groupby('citing')
            .agg(**{f'{label}_total': ('is_self', 'count'), f'{label}_self': ('is_self', 'sum')})
            .reset_index().rename(columns={'citing': 'app_id'}))
    g[f'{label}_context_occurred'] = 1
    g[f'{label}_any_self_within_context'] = (g[f'{label}_self'] > 0).astype(int)
    g[f'{label}_self_ratio_within_context'] = g[f'{label}_self'] / g[f'{label}_total'].clip(lower=1)
    return g[['app_id', f'{label}_context_occurred', f'{label}_any_self_within_context',
              f'{label}_self_ratio_within_context']]


ctx_rejection = context_two_stage(REJECT_REASON_CTX, 'rejection_ctx')
ctx_grant = context_two_stage(GRANT_CTX, 'grant_ctx')
del cc; gc.collect()
print(f"  Block2 two-stage contexts + reason22 dates built  ({time.time()-t0:.1f}s)", flush=True)

# ----------------------------------------------------------------
# STEP 4: 完全ゼロ埋めグリッド + group-safe stock（バグ#1,#2修正）
# ----------------------------------------------------------------
print(f"\n[4/8] Building zero-filled grid and group-safe stocks ...", flush=True)

firm_field_counts = ap.groupby(['applicant_id', 'field_id', 'app_year']).size().reset_index(name='n_filed')
fy = firm_field_counts.groupby('applicant_id')['app_year'].agg(['min', 'max']).reset_index()
fy['max'] = fy['max'].clip(upper=SAMPLE_END_YEAR)
fy = fy[fy['max'] >= fy['min']]
fields_by_firm = firm_field_counts.groupby('applicant_id')['field_id'].unique().reset_index()
grid = fy.merge(fields_by_firm, on='applicant_id')
grid['years'] = grid.apply(lambda r: list(range(int(r['min']), int(r['max']) + 1)), axis=1)
grid = grid[['applicant_id', 'years', 'field_id']].explode('years').explode('field_id')
grid = grid.rename(columns={'years': 'app_year'}); grid['app_year'] = grid['app_year'].astype(int)
grid = grid.merge(firm_field_counts, on=['applicant_id', 'field_id', 'app_year'], how='left')
grid['n_filed'] = grid['n_filed'].fillna(0)
grid = grid.sort_values(['applicant_id', 'field_id', 'app_year']).reset_index(drop=True)

grid['patent_stock'] = (grid.groupby(['applicant_id', 'field_id'])['n_filed']
                             .transform(lambda s: s.cumsum().shift(1).fillna(0)))

ap['grant_year'] = ap['rdate_dt'].dt.year
granted = ap.dropna(subset=['grant_year'])
grant_counts = granted.groupby(['applicant_id', 'field_id', 'grant_year']).size().reset_index(name='n_granted')
grant_counts['grant_year'] = grant_counts['grant_year'].astype(int)
grant_counts = grant_counts.rename(columns={'grant_year': 'app_year'})
grid = grid.merge(grant_counts, on=['applicant_id', 'field_id', 'app_year'], how='left')
grid['n_granted'] = grid['n_granted'].fillna(0)
grid['grant_stock'] = (grid.groupby(['applicant_id', 'field_id'])['n_granted']
                            .transform(lambda s: s.cumsum().shift(1).fillna(0)))


def decay_transform(s, delta=DECAY_DELTA):
    vals = s.values.astype(float); out = np.zeros(len(vals)); running = 0.0
    for i in range(len(vals)):
        out[i] = running; running = running * (1 - delta) + vals[i]
    return pd.Series(out, index=s.index)


grid['decayed_stock_d10'] = (grid.groupby(['applicant_id', 'field_id'], group_keys=False)['n_filed']
                                  .apply(decay_transform))
print(f"  zero-filled grid: {len(grid):,} rows, stocks built  ({time.time()-t0:.1f}s)", flush=True)

# ----------------------------------------------------------------
# STEP 5: own_portfolio_hhi / entropy（完全な過去ポートフォリオから、バグ#3修正）
# ----------------------------------------------------------------
print(f"\n[5/8] Building own_portfolio_hhi / entropy from full historical grid ...", flush=True)

firm_year_total = grid.groupby(['applicant_id', 'app_year'])['patent_stock'].sum().reset_index(name='total_stock')
tmp = grid.merge(firm_year_total, on=['applicant_id', 'app_year'])
tmp['share'] = np.where(tmp['total_stock'] > 0, tmp['patent_stock'] / tmp['total_stock'], 0)
tmp['share_sq'] = tmp['share'] ** 2
with np.errstate(divide='ignore'):
    tmp['entropy_term'] = np.where(tmp['share'] > 0, -tmp['share'] * np.log(tmp['share']), 0)
own_portfolio = (tmp.groupby(['applicant_id', 'app_year'])
                     .agg(own_portfolio_hhi=('share_sq', 'sum'),
                          portfolio_entropy=('entropy_term', 'sum'),
                          n_active_fields=('patent_stock', lambda s: (s > 0).sum()),
                          total_stock=('patent_stock', 'sum'))
                     .reset_index())
grid = grid.merge(own_portfolio[['applicant_id', 'app_year', 'own_portfolio_hhi', 'portfolio_entropy', 'n_active_fields']],
                   on=['applicant_id', 'app_year'], how='left')
print(f"  done  ({time.time()-t0:.1f}s)", flush=True)

# ----------------------------------------------------------------
# STEP 6: 年別relatedness matrix（未来情報排除、バグ#4修正）
# ----------------------------------------------------------------
print(f"\n[6/8] Building year-expanding relatedness matrices ...", flush=True)

all_fields = sorted(ap['field_id'].unique())
years = sorted(grid['app_year'].unique())
matrices = {}
for y in years:
    prior = ap[ap['app_year'] < y]
    if len(prior) == 0:
        matrices[y] = None
        continue
    ffm = prior.groupby(['field_id', 'applicant_id']).size().unstack(fill_value=0)
    ffm = ffm.reindex(index=all_fields, fill_value=0)
    X = ffm.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
    Xn = X / norms
    matrices[y] = pd.DataFrame(Xn @ Xn.T, index=all_fields, columns=all_fields)

port_dict = (grid[grid['patent_stock'] > 0].groupby(['applicant_id', 'app_year'])[['field_id', 'patent_stock']]
             .apply(lambda g: dict(zip(g['field_id'], g['patent_stock']))).to_dict())


def compute_relatedness(fid, yr, portfolio):
    R = matrices.get(yr)
    if R is None:
        return np.nan
    other = {h: v for h, v in portfolio.items() if h != fid and v > 0}
    if not other:
        return np.nan
    tot = sum(other.values())
    s = 0.0
    for h, v in other.items():
        r = R.at[fid, h] if (fid in R.index and h in R.columns) else 0.0
        s += (v / tot) * r
    return s


grid['relatedness_dynamic'] = [
    compute_relatedness(fid, yr, port_dict.get((aid, yr), {}))
    for fid, yr, aid in zip(grid['field_id'], grid['app_year'], grid['applicant_id'])
]
print(f"  relatedness_dynamic non-missing: {grid['relatedness_dynamic'].notna().mean():.1%}"
      f"  ({time.time()-t0:.1f}s)", flush=True)

grid.to_pickle(f'{SAVE_DIR}/panel_v3.pkl')

# ----------------------------------------------------------------
# STEP 7: M&Aスパイク + Block3（competing risks、バグ#9,#10,#12修正）
# ----------------------------------------------------------------
print(f"\n[7/8] Building M&A flag and Block3 competing-risk variables ...", flush=True)

firm_year_all = (ap.groupby(['applicant_id', 'app_year']).size().reset_index(name='n_filed_total')
                    .sort_values(['applicant_id', 'app_year']))
firm_year_all['n_filed_prev'] = firm_year_all.groupby('applicant_id')['n_filed_total'].shift(1)
firm_year_all['spike_ratio'] = np.where(
    firm_year_all['n_filed_prev'].fillna(0) >= MA_SPIKE_MIN_BASE,
    firm_year_all['n_filed_total'] / firm_year_all['n_filed_prev'].replace(0, np.nan), np.nan)
firm_year_all['ma_spike_flag'] = firm_year_all['spike_ratio'] >= MA_SPIKE_RATIO
ap = ap.merge(firm_year_all[['applicant_id', 'app_year', 'ma_spike_flag']], on=['applicant_id', 'app_year'], how='left')
ap['ma_spike_flag'] = ap['ma_spike_flag'].fillna(False)

ap = ap.merge(r22, on='app_id', how='left')
ap['post_2001_reform'] = (ap['adate_dt'] >= REFORM_DATE).astype(int)

sdate_cap = ap['sdate_dt'] + pd.to_timedelta(CENSOR_CAP_YEARS * 365.25, unit='D')
fallback_cap = ap['adate_dt'] + pd.to_timedelta(CENSOR_CAP_YEARS * 365.25, unit='D')
ap['censoring_time'] = sdate_cap.fillna(fallback_cap).clip(upper=DATA_CUTOFF)

grant_date = ap['rdate_dt']
reason22_date = ap['reason22_first_date']
grant_cmp = grant_date.fillna(pd.Timestamp.max)
reason22_cmp = reason22_date.fillna(pd.Timestamp.max)
grant_is_first = grant_date.notna() & (grant_cmp <= reason22_cmp)
reason22_is_first = reason22_date.notna() & (reason22_cmp < grant_cmp)

ap['event_grant'] = grant_is_first.astype(int)
ap['duration_to_grant'] = np.where(
    ap['event_grant'] == 1, (grant_date - ap['adate_dt']).dt.days,
    (np.minimum(reason22_cmp.where(reason22_is_first, ap['censoring_time']), ap['censoring_time']) - ap['adate_dt']).dt.days)

ap['event_reason22'] = reason22_is_first.astype(int)
ap['duration_to_reason22'] = np.where(
    ap['event_reason22'] == 1, (reason22_date - ap['adate_dt']).dt.days,
    (np.minimum(grant_cmp.where(grant_is_first, ap['censoring_time']), ap['censoring_time']) - ap['adate_dt']).dt.days)

ap['duration_to_grant_from_sdate'] = np.where(
    ap['event_grant'] == 1, (grant_date - ap['sdate_dt']).dt.days,
    (np.minimum(reason22_cmp.where(reason22_is_first, ap['censoring_time']), ap['censoring_time']) - ap['sdate_dt']).dt.days)

ap = ap.merge(ctx_rejection, on='app_id', how='left')
ap = ap.merge(ctx_grant, on='app_id', how='left')
for c in ['rejection_ctx_context_occurred', 'rejection_ctx_any_self_within_context', 'rejection_ctx_self_ratio_within_context',
          'grant_ctx_context_occurred', 'grant_ctx_any_self_within_context', 'grant_ctx_self_ratio_within_context']:
    ap[c] = ap[c].fillna(0)

ap['claim_reduction'] = (ap['claim1'] - ap['claim3']).clip(lower=0)
ap['claim_expansion'] = (ap['claim3'] - ap['claim1']).clip(lower=0)
ap['any_claim_change'] = (ap['claim1'] != ap['claim3']).astype('Int64')
print(f"  event_grant rate: {ap['event_grant'].mean():.1%}, "
      f"event_reason22 rate: {ap['event_reason22'].mean():.1%}  ({time.time()-t0:.1f}s)", flush=True)

# ----------------------------------------------------------------
# STEP 8: 統合・保存
# ----------------------------------------------------------------
print(f"\n[8/8] Merging panel-level vars into application-level table and saving ...", flush=True)

ap = ap.merge(grid[['applicant_id', 'field_id', 'app_year', 'patent_stock', 'grant_stock',
                     'decayed_stock_d10', 'relatedness_dynamic', 'own_portfolio_hhi',
                     'portfolio_entropy', 'n_active_fields']],
              on=['applicant_id', 'field_id', 'app_year'], how='left')

neg_check = (ap['duration_to_grant'] < 0).sum() + (ap['duration_to_reason22'] < 0).sum()
print(f"  negative-duration sanity check (should be 0): {neg_check}")

ap.to_pickle(f'{SAVE_DIR}/df_application_v3.pkl')
print(f"\n完了 ({time.time()-t0:.1f}s total)")
print(f"Saved -> {SAVE_DIR}/df_application_v3.pkl, panel_v3.pkl")
print("\n次のステップ: この修正版データで Block1A〜3 を再推定してください。")
