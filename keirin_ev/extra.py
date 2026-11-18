"""手元の結果から作る追加の視点（2026-11-18 追加）。どれも「そのレースの日より前」の結果だけで計算する。

  対戦成績     : 今回いっしょに走る相手と過去に同じレースを走ったとき、何回先着したか（相手ごとに縮めて平均）
  対戦・同ライン外: 上と同じだが、相手が別ラインのときだけ（ライン内の着順は役割で決まりやすいので分ける）
  場のクセ     : その場で 先頭／番手／3番手／単騎 が1着になる割合の、全場平均との差（回数が少ないと0に寄せる）
                 × 今回の位置（先頭なら先頭のクセ…）
  場の決まり手  : その場の 逃げ・捲り・差し の決まりやすさ（全場との差）× 選手の自力型・差し型の割合
  脚質の変化    : 直近5走の B（バック先頭）率 − それ以前30走の B 率（自力に寄ってきたか）
  調子の変化    : 直近5走の 3着内率 − それ以前30走の 3着内率
  B率（直近）  : 直近10走の B 率
"""
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

EXTRA_FEATURES = ["x_h2h", "x_h2h_out", "x_venue_pos", "x_venue_kima", "x_b_trend", "x_form_trend", "x_b_recent"]
EXTRA_LABEL = dict(x_h2h="対戦成績", x_h2h_out="対戦成績（別ライン）", x_venue_pos="場のクセ（位置）",
                   x_venue_kima="場のクセ（決まり手）", x_b_trend="脚質の変化（B率）",
                   x_form_trend="調子の変化（3着内率）", x_b_recent="直近10走のB率")
H2H_K = 3.0       # 対戦成績の縮め具合（この回数ぶん五分の実績を足したのと同じ）
VENUE_K = 200.0   # 場のクセの縮め具合（レース数）
POS = ("head", "2nd", "3rd", "single")
KIMA = {"逃": "nige", "捲": "makuri", "差": "sashi", "ク": "sashi"}


def _rkey(df):
    term = pd.to_numeric(df["term"], errors="coerce").fillna(0).astype(int).astype(str)
    return df["name"].astype(str) + "|" + df["pref"].astype(str) + "|" + term


def _pos(line_pos, line_size):
    if line_size <= 1:
        return "single"
    return {1: "head", 2: "2nd"}.get(int(line_pos), "3rd")


def load_finish(root="data/daily"):
    L = []
    for f in sorted(Path(root).glob("*/*/*/finish.csv.gz")):
        try:
            L.append(pd.read_csv(f, dtype={"race_id": str}, usecols=["race_id", "car", "rank", "kimarite", "sb"]))
        except Exception:
            pass
    return pd.concat(L, ignore_index=True) if L else pd.DataFrame(columns=["race_id", "car", "rank", "kimarite", "sb"])


def build(entries: pd.DataFrame, races: pd.DataFrame, finish: pd.DataFrame = None, targets=None):
    """entries（過去分も含む全出走）・races・finish（着順）から、targets（race_id の集合。None なら全部）の
    各出走に追加の視点を付ける。戻り値: DataFrame[race_id, car, *EXTRA_FEATURES]"""
    if finish is None:
        finish = load_finish()
    e = entries.copy()
    e["rk"] = _rkey(e)
    e = e.merge(races[["race_id", "date", "venue"]], on="race_id", how="inner")
    fin = finish.drop_duplicates(["race_id", "car"])
    e = e.merge(fin, on=["race_id", "car"], how="left")
    e["pos"] = [_pos(p, s) for p, s in zip(e["line_pos"].fillna(1), e["line_size"].fillna(1))]
    e["date"] = e["date"].astype(str)
    e = e.sort_values(["date", "race_id", "car"])
    tot = e[["nige", "makuri", "sashi", "mark"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    s = tot.sum(1).replace(0, np.nan)
    e["jiriki"] = ((tot["nige"] + tot["makuri"]) / s).fillna(0)
    e["sasi_t"] = ((tot["sashi"] + tot["mark"]) / s).fillna(0)

    h2h = defaultdict(lambda: [0.0, 0])                      # (i, j) → [i が先着した回数, 対戦数]
    hist = defaultdict(lambda: deque(maxlen=35))             # 選手 → 直近の (B取得, 3着内)
    v_pos = defaultdict(lambda: defaultdict(float)); v_n = defaultdict(int)     # 場 → 位置ごとの1着数, レース数
    v_kim = defaultdict(lambda: defaultdict(float))
    g_pos = defaultdict(float); g_kim = defaultdict(float); g_n = [0]
    out = []
    want = None if targets is None else set(targets)
    for date, day in e.groupby("date", sort=True):
        # ---- その日のレースの視点（前日までの集計だけを使う） ----
        for rid, g in day.groupby("race_id", sort=False):
            if want is not None and rid not in want:
                continue
            ven = g["venue"].iloc[0]
            n_v = v_n[ven]
            rows = g[["car", "rk", "line_id", "pos", "jiriki", "sasi_t"]].values
            for car, rk, lid, pos, jr, st in rows:
                a = b = 0.0; na = nb = 0
                for car2, rk2, lid2, *_ in rows:
                    if rk2 == rk:
                        continue
                    w, n = h2h[(rk, rk2)]
                    if n:
                        d = (w - n / 2) / (n + H2H_K)
                        a += d; na += 1
                        if lid2 != lid:
                            b += d; nb += 1
                vp = 0.0
                if g_n[0]:
                    vr = v_pos[ven][pos] / n_v if n_v else 0.0
                    gr = g_pos[pos] / g_n[0]
                    vp = (vr - gr) * n_v / (n_v + VENUE_K)
                vk = 0.0
                if g_n[0] and n_v:
                    sh = n_v / (n_v + VENUE_K)
                    dj = (v_kim[ven]["nige"] + v_kim[ven]["makuri"]) / n_v - (g_kim["nige"] + g_kim["makuri"]) / g_n[0]
                    ds = v_kim[ven]["sashi"] / n_v - g_kim["sashi"] / g_n[0]
                    vk = sh * (jr * dj + st * ds)
                h = list(hist[rk])
                rec, old = h[-5:], h[:-5]
                bt = ft = 0.0
                if len(rec) >= 3 and len(old) >= 5:
                    bt = np.mean([x[0] for x in rec]) - np.mean([x[0] for x in old])
                    ft = np.mean([x[1] for x in rec]) - np.mean([x[1] for x in old])
                br = np.mean([x[0] for x in h[-10:]]) if len(h) >= 3 else 0.0
                out.append((rid, int(car), a / na if na else 0.0, b / nb if nb else 0.0, vp, vk, bt, ft, br))
        # ---- その日の結果で集計を更新 ----
        for rid, g in day.groupby("race_id", sort=False):
            g = g[g["rank"].notna()]
            if g.empty:
                continue
            ven = g["venue"].iloc[0]
            rk_ = dict(zip(g["rk"], g["rank"].astype(float)))
            ks = list(rk_)
            for i in ks:
                for j in ks:
                    if i != j:
                        c = h2h[(i, j)]; c[0] += float(rk_[i] < rk_[j]); c[1] += 1
            for rk, rank, sb in zip(g["rk"], g["rank"], g["sb"].fillna("")):
                hist[rk].append((float("B" in str(sb).upper()), float(rank <= 3)))
            w = g[g["rank"] == 1]
            if len(w):
                p = w["pos"].iloc[0]; k = KIMA.get(str(w["kimarite"].iloc[0]), None)
                v_pos[ven][p] += 1; g_pos[p] += 1
                if k:
                    v_kim[ven][k] += 1; g_kim[k] += 1
                v_n[ven] += 1; g_n[0] += 1
    return pd.DataFrame(out, columns=["race_id", "car", *EXTRA_FEATURES])
