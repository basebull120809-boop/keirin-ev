"""競輪 期待値投資システム  コマンド一覧

  python main.py init                         DB作成とCSVひな型出力
  python main.py import <table> <csv>         CSV取込 (races/entries/results/odds_trifecta)
  python main.py probe                        Kドリームスで何年前まで取れるか確認
  python main.py collect <開始日> <終了日>      Kドリームスから取得し data/daily に保存
  python main.py backfill <日数> <最古日>       取得済みの最古日からさらに遡って取得
  python main.py builddb                      data/daily を SQLite に読み込む
  python main.py train                        全データで学習し models/latest.json に保存
  python main.py live [分]                     当日運用：締切前オッズで判定してIssueに通知
  python main.py market                       市場の歪みレポート（人気別・オッズ帯別回収率）
  python main.py backtest [--ev 1.2]          ウォークフォワード検証
  python main.py predict <race_id> [--bank 300000]   当日レースの推奨買い目
  python main.py demo [--bias 0.25]           合成データで全工程を試す
"""
import sys
import argparse
import numpy as np
import pandas as pd
from keirin_ev.db import connect
from keirin_ev import importer, collector
from keirin_ev.model import TrifectaModel, build_tensors, COMBOS
from keirin_ev.strategy import Rules, pick_bets
from keirin_ev.backtest import walk_forward, summarize, market_report

pd.set_option("display.width", 160)


def cmd_backtest(con, a):
    rules = Rules(ev_min=a.ev, kelly_frac=a.kelly)
    log, coefs = walk_forward(con, rules, bankroll0=a.bank,
                              min_train_months=a.train_months)
    print("\n■ 市場ブレンド係数の推移（a=モデル, b=市場。aが0に近ければ優位性なし）")
    print(coefs.round(3).to_string(index=False))
    print("\n■ 検証結果")
    for k, v in summarize(log, a.bank).items():
        print(f"  {k}: {v}")
    if not log.empty:
        log.to_csv(a.out, index=False, encoding="utf-8-sig")
        m = log.groupby("month").agg(投資=("stake", "sum"), 払戻=("payout", "sum"))
        m["回収率"] = (m["払戻"] / m["投資"]).round(3)
        print("\n■ 月別"); print(m.to_string())
        print(f"\n購入ログ: {a.out}")


def cmd_predict(con, a):
    D = build_tensors(con)                       # 学習: 結果がある全レース
    mdl = TrifectaModel().fit(D["T"], D["first"], D["second"], D["third"],
                              D["Q"], D["win_combo"])
    X = build_tensors(con, race_ids=[a.race_id], need_result=False)
    if not X["T"].race_ids:
        sys.exit("出走表かオッズが未登録です")
    P = mdl.predict(X["T"], X["Q"])[0]
    O = X["O"][0]
    bets = pick_bets(P, O, a.bank, Rules(ev_min=a.ev, kelly_frac=a.kelly))
    cars = X["T"].cars[0]
    print(f"■ {a.race_id}  （オッズは締切直前に再取得して再判定すること）")
    if not bets:
        print("  見送り：期待値の基準を満たす買い目なし")
        return
    print(f"  {'買い目':<8}{'金額':>7}{'確率':>8}{'オッズ':>8}{'期待値':>8}")
    for n, st, p, o, ev in bets:
        i, j, k = COMBOS[n]
        print(f"  {cars[i]}-{cars[j]}-{cars[k]:<4}{st:>7,}{p:>8.2%}{o:>8.1f}{ev:>8.2f}")


def cmd_report(con, a):
    from pathlib import Path
    from keirin_ev import store
    days = store.load_into_db(con)
    nr = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    L = [f"# 競輪 期待値システム レポート",
         f"更新: {pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}",
         "", f"- 取得済み: **{days}日分 / 結果あり {nr}レース**"]
    dates = store.collected_dates()
    if dates:
        L.append(f"- 期間: {dates[0]} 〜 {dates[-1]}")
    if nr >= 200:
        r1, r2 = market_report(con)
        L += ["", "## 人気帯別の回収率（100円均等買い）",
              "控除率25%なので、市場が正しければどこも約75%になる。偏りが歪み。", "",
              r1.to_markdown(), "", "## オッズ帯別の回収率", "", r2.to_markdown()]
    months = con.execute("SELECT COUNT(DISTINCT substr(date,1,7)) FROM races").fetchone()[0]
    if months >= a.train_months + 1:
        log, coefs = walk_forward(con, Rules(ev_min=a.ev, kelly_frac=a.kelly),
                                  bankroll0=a.bank, min_train_months=a.train_months)
        sm = summarize(log, a.bank)
        L += ["", f"## ウォークフォワード検証（期待値{a.ev}以上・1/4ケリー）", ""]
        L += [f"- {k}: **{v}**" for k, v in sm.items()]
        L += ["", "判定基準: 回収率95%区間の下限が1.0を超えるまで実弾は入れない。", "",
              "### 市場ブレンド係数（a=モデル b=市場）", "", coefs.round(3).to_markdown(index=False)]
        if not log.empty:
            m = log.groupby("month").agg(投資=("stake", "sum"), 払戻=("payout", "sum"))
            m["回収率"] = (m["払戻"] / m["投資"]).round(3)
            L += ["", "### 月別", "", m.to_markdown()]
    else:
        L += ["", f"検証には{a.train_months + 1}ヶ月分以上のデータが必要です（現在{months}ヶ月）。"]
    from keirin_ev.live import paper_results
    pb = paper_results()
    if pb is not None:
        done = pb.dropna(subset=["hit"])
        L += ["", "## 当日運用（締切前オッズで判定した買い目の実績）", ""]
        if len(done):
            st, py = done["stake"].sum(), done["payout"].sum()
            L += [f"- 判定レース {done['race_id'].nunique()} / {len(done)}点 / 的中 {int(done['hit'].sum())}",
                  f"- 投資 {int(st):,}円 → 払戻 {int(py):,}円  回収率 **{py / st:.1%}**", ""]
            d = done.groupby("date").agg(点数=("combo", "size"), 投資=("stake", "sum"),
                                         払戻=("payout", "sum"))
            d["回収率"] = (d["払戻"] / d["投資"]).round(3)
            L += [d.tail(14).to_markdown()]
        pending = pb["hit"].isna().sum()
        if pending:
            L += ["", f"結果待ち {pending}点"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def cmd_demo(con, a):
    from keirin_ev.synth import generate
    print(f"合成データ生成中（市場の偏り bias={a.bias}）…")
    for name, df in zip(["races", "entries", "results", "odds_trifecta"],
                        generate(a.races, bias=a.bias, seed=a.seed)):
        collector.store(con, {name: df})
    a.out = f"data/demo_bets_bias{a.bias}.csv"
    rep_rank, rep_odds = market_report(con)
    print("\n■ 人気帯別 回収率"); print(rep_rank.to_string())
    print("\n■ オッズ帯別 回収率"); print(rep_odds.to_string())
    cmd_backtest(con, a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("args", nargs="*")
    p.add_argument("--db", default="data/keirin.db")
    p.add_argument("--ev", type=float, default=1.2)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--bank", type=int, default=300_000)
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--out", default="data/backtest_bets.csv")
    p.add_argument("--folder", default="downloads")
    p.add_argument("--source", default="kdreams")
    p.add_argument("--bias", type=float, default=0.25)
    p.add_argument("--races", type=int, default=6000)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "demo":
        a.db = f"data/demo_bias{a.bias}.db"
    con = connect(a.db)

    if a.cmd == "init":
        importer.write_templates("templates")
        print("DB作成済み / templates/ にCSVひな型を出力しました")
    elif a.cmd == "import":
        print(importer.import_csv(con, a.args[0], a.args[1]), "行取込")
    elif a.cmd == "collect":                     # 指定期間を日別CSVに保存
        from keirin_ev.kdreams import KdreamsCollector
        from keirin_ev import store
        c = KdreamsCollector()
        for d in pd.date_range(a.args[0], a.args[1]):
            ds = d.strftime("%Y-%m-%d")
            if store.has_day(ds):
                print(ds, "取得済み"); continue
            print(ds, flush=True)
            print(f"  → {store.save_day(ds, c.fetch_day(ds))} 行保存", flush=True)
    elif a.cmd == "backfill":                    # 古い方へ N日ずつ遡って取得
        from keirin_ev.kdreams import KdreamsCollector
        from keirin_ev import store
        n_days, stop = int(a.args[0]), a.args[1]
        done = store.collected_dates()
        d = pd.Timestamp(done[0]) if done else pd.Timestamp.today().normalize()
        c, cnt = KdreamsCollector(), 0
        while cnt < n_days and d > pd.Timestamp(stop):
            d -= pd.Timedelta(days=1)
            ds = d.strftime("%Y-%m-%d")
            if store.has_day(ds):
                continue
            print(ds, flush=True)
            print(f"  → {store.save_day(ds, c.fetch_day(ds))} 行保存", flush=True)
            cnt += 1
    elif a.cmd == "builddb":                     # 日別CSV → SQLite
        from keirin_ev import store
        print(store.load_into_db(con), "日分を読み込みました")
    elif a.cmd == "probe":
        from keirin_ev.kdreams import probe
        today = pd.Timestamp.today().normalize()
        probe([(today - pd.DateOffset(**k)).strftime("%Y-%m-%d") for k in
               [dict(days=1), dict(months=1), dict(years=1), dict(years=2),
                dict(years=3), dict(years=5)]])
    elif a.cmd == "train":                       # 全データで学習 → models/latest.json
        from keirin_ev import store
        store.load_into_db(con)
        D = build_tensors(con)
        n = len(D["T"].race_ids)
        if n < 100:
            sys.exit(f"学習データが少なすぎます（{n}レース）")
        m = TrifectaModel().fit(D["T"], D["first"], D["second"], D["third"], D["Q"], D["win_combo"])
        dates = D["races"]["date"]
        m.to_json("models/latest.json", n_races=n, period=f"{dates.min()}〜{dates.max()}",
                  trained_at=f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}")
        print(f"学習完了 {n}レース  ブレンド a={m.blend[0]:.3f} b={m.blend[1]:.3f}")
    elif a.cmd == "live":                        # 当日運用（締切前判定と通知）
        from keirin_ev.live import run
        run(max_minutes=float(a.args[0]) if a.args else 340)
    elif a.cmd == "report":                      # スマホで見るレポート(Markdown)
        cmd_report(con, a)
    elif a.cmd == "market":
        r1, r2 = market_report(con)
        print(r1.to_string()); print(); print(r2.to_string())
    elif a.cmd == "backtest":
        cmd_backtest(con, a)
    elif a.cmd == "predict":
        a.race_id = a.args[0]; cmd_predict(con, a)
    elif a.cmd == "demo":
        cmd_demo(con, a)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
