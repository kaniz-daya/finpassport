"""
FinPassport - Model 1: probabilistic SAR->BDT rate forecast + "send now or wait" policy.

Data
  Default: a simulated 3-year daily SAR->BDT series (seeded) that mimics a managed
  floating taka: slow crawling depreciation, mean reversion around the trend,
  weekday and month-end remittance effects, volatility clustering and rare
  policy step moves. Replace it with a real series by passing --csv path
  (columns: date,rate).

Model
  For each horizon h = 1..7 days and quantile q in {0.1, 0.5, 0.9}, a linear
  quantile regression predicts the log-return r(t+h)/r(t) - 1 from features known
  at day t: recent returns, deviation from the 30-day mean, 10-day volatility,
  day-of-week and month-end indicators. 21 small linear models, exported to JSON
  so the browser runs them with no dependencies.

Evaluation (out-of-sample, last 12 months)
  * pinball loss vs. a random-walk baseline with empirical quantiles
  * coverage of the 80% interval (should be close to 80%)
  * decision backtest: a student must send 1,000 SAR within 7 days.
      - always send today
      - model policy: wait for the day with the best predicted median, only if the
        expected gain beats a threshold and the 10% quantile is not much worse
      - oracle (perfect hindsight) as the upper bound
"""
import argparse, json, math, os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
QUANTILES = [0.1, 0.5, 0.9]
HORIZONS = list(range(1, 8))
TODAY = pd.Timestamp("2026-10-20")
FUTURE_DAYS = 14


def simulate(seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(TODAY - pd.Timedelta(days=3 * 365), TODAY + pd.Timedelta(days=FUTURE_DAYS), freq="D")
    n = len(dates)
    trend = np.log(31.2) + np.arange(n) * (np.log(32.45 / 31.2) / n)  # slow taka depreciation
    dev = 0.0
    sigma = 0.0011
    x = np.zeros(n)
    for i, d in enumerate(dates):
        # volatility clustering (GARCH-like)
        shock = rng.normal()
        sigma = math.sqrt(0.0000002 + 0.08 * (sigma * shock) ** 2 + 0.88 * sigma ** 2)
        sigma = min(max(sigma, 0.0006), 0.004)
        dow = d.dayofweek  # Mon=0 ... Sun=6
        weekday_eff = {6: 0.0007, 0: 0.0005, 3: -0.0004, 4: -0.0006}.get(dow, 0.0)
        month_end = 1 if d.days_in_month - d.day < 3 else 0
        month_eff = -0.0009 * month_end  # remittance inflows peak at month end -> taka firmer
        jump = rng.choice([0, 1], p=[0.997, 0.003]) * rng.uniform(0.006, 0.015)
        dev = 0.88 * dev + sigma * shock + jump
        x[i] = trend[i] + dev + weekday_eff + month_eff
    return pd.DataFrame({"date": dates, "rate": np.round(np.exp(x), 4)})


def features(df):
    r = df["rate"].values
    lr = np.log(r)
    out = []
    for t in range(len(df)):
        if t < 30:
            out.append(None)
            continue
        ret = lambda k: lr[t] - lr[t - k]
        w = lr[t - 29:t + 1]
        d = df["date"].iloc[t]
        vol = np.std(np.diff(lr[t - 10:t + 1]))
        f = [ret(1), ret(3), ret(7), lr[t] - w.mean(), vol]
        f += [1.0 if d.dayofweek == k else 0.0 for k in range(7)]
        f += [1.0 if d.days_in_month - d.day < 3 else 0.0, 1.0 if d.day <= 3 else 0.0]
        out.append(f)
    return out


FEATURE_NAMES = ["ret_1d", "ret_3d", "ret_7d", "dev_from_30d_mean", "vol_10d",
                 "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
                 "month_end", "month_start"]


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def run(df, train_end, test_end, verbose=True):
    F = features(df)
    lr = np.log(df["rate"].values)
    dates = df["date"]
    models, report = {}, {"horizons": {}}
    idx_train = [t for t in range(len(df)) if F[t] is not None and dates.iloc[t] + pd.Timedelta(days=7) <= train_end]
    idx_test = [t for t in range(len(df)) if F[t] is not None and dates.iloc[t] > train_end and dates.iloc[t] + pd.Timedelta(days=7) <= test_end]
    Xtr = np.array([F[t] for t in idx_train]); Xte = np.array([F[t] for t in idx_test])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    sd[5:] = 1.0; mu[5:] = 0.0  # keep indicators as 0/1
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    preds = {}
    for h in HORIZONS:
        ytr = np.array([lr[t + h] - lr[t] for t in idx_train])
        yte = np.array([lr[t + h] - lr[t] for t in idx_test])
        models[h] = {}
        rep = {}
        for tau in QUANTILES:
            m = QuantileRegressor(quantile=tau, alpha=0.0005, solver="highs").fit(Ztr, ytr)
            models[h][tau] = (m.intercept_, m.coef_)
            p = m.predict(Zte)
            preds[(h, tau)] = p
            base = np.quantile(ytr, tau)  # random walk with empirical quantiles
            rep[f"pinball_q{int(tau*100)}"] = {"model": pinball(yte, p, tau), "random_walk": pinball(yte, np.full_like(yte, base), tau)}
        lo, hi = preds[(h, 0.1)], preds[(h, 0.9)]
        rep["coverage_80"] = float(np.mean((yte >= lo) & (yte <= hi)))
        rep["mae_median_bps"] = float(np.mean(np.abs(yte - preds[(h, 0.5)])) * 1e4)
        rep["mae_random_walk_bps"] = float(np.mean(np.abs(yte - np.median(ytr))) * 1e4)
        report["horizons"][h] = rep

    # Decision backtest: send 1,000 SAR within 7 days
    amount = 1000
    now_bdt, policy_bdt, oracle_bdt, waited, lost = [], [], [], 0, 0
    for j, t in enumerate(idx_test):
        today = df["rate"].iloc[t]
        med = np.array([preds[(h, 0.5)][j] for h in HORIZONS])
        p10 = np.array([preds[(h, 0.1)][j] for h in HORIZONS])
        k = int(np.argmax(med))
        choose = 0
        if med[k] > 0.0008 and p10[k] > -0.0025:  # expected gain > 0.08% and limited downside
            choose = HORIZONS[k]
        got = df["rate"].iloc[t + choose] * amount
        now_bdt.append(today * amount)
        policy_bdt.append(got)
        oracle_bdt.append(max(df["rate"].iloc[t:t + 8]) * amount)
        if choose:
            waited += 1
            if got < today * amount:
                lost += 1
    now_bdt, policy_bdt, oracle_bdt = map(np.array, (now_bdt, policy_bdt, oracle_bdt))
    report["backtest"] = {
        "days_tested": len(idx_test),
        "avg_gain_vs_send_now_bdt_per_1000_sar": float(np.mean(policy_bdt - now_bdt)),
        "oracle_gain_bdt_per_1000_sar": float(np.mean(oracle_bdt - now_bdt)),
        "share_of_oracle_captured": float(np.mean(policy_bdt - now_bdt) / max(1e-9, np.mean(oracle_bdt - now_bdt))),
        "wait_recommended_share": waited / len(idx_test),
        "wait_turned_out_worse_share": lost / max(1, waited),
    }
    export = {
        "feature_names": FEATURE_NAMES, "mu": mu.tolist(), "sd": sd.tolist(),
        "quantiles": QUANTILES, "horizons": HORIZONS,
        "models": {str(h): {str(q): {"b": float(models[h][q][0]), "w": [float(v) for v in models[h][q][1]]} for q in QUANTILES} for h in HORIZONS},
        "policy": {"min_gain": 0.0008, "max_downside_q10": -0.0025},
    }
    return export, report


def predict_one(export, df, t):
    F = features(df)[t]
    z = (np.array(F) - np.array(export["mu"])) / np.array(export["sd"])
    out = {}
    for h in HORIZONS:
        out[h] = {q: export["models"][str(h)][str(q)]["b"] + float(np.dot(export["models"][str(h)][str(q)]["w"], z)) for q in QUANTILES}
    return out


def demo_ok(df, export):
    """The app's demo needs: today below its 30-day median, a reachable alert target
    within 4 days, and a model that recommends waiting."""
    t = int(np.where(df["date"] == TODAY)[0][0])
    w = df["rate"].iloc[t - 29:t + 1].values
    today = w[-1]
    pct = np.mean(w[:-1] < today)
    target = max(round(float(np.quantile(w, 0.75)), 2), round(today + 0.05, 2))
    fut = df["rate"].iloc[t + 1:t + 5].values
    p = predict_one(export, df, t)
    hb = max(HORIZONS, key=lambda h: p[h][0.5])
    wait = p[hb][0.5] > export["policy"]["min_gain"] and p[hb][0.1] > export["policy"]["max_downside_q10"]
    return pct < 0.4 and fut.max() >= target and fut[0] < target and wait, dict(pct=pct, target=target, today=today, h=hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="real series with columns date,rate")
    args = ap.parse_args()
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        seed, source = None, "csv:" + os.path.basename(args.csv)
        train_end = df["date"].iloc[-1] - pd.Timedelta(days=365); test_end = df["date"].iloc[-1]
        export, report = run(df, train_end, test_end)
    else:
        source = "simulated"
        for seed in range(1, 500):
            df = simulate(seed)
            export, report = run(df, TODAY - pd.Timedelta(days=365), TODAY - pd.Timedelta(days=1))
            ok, info = demo_ok(df, export)
            if ok:
                break
        print("seed", seed, info)
    report["data"] = {"source": source, "seed": seed, "days": len(df)}
    t = int(np.where(df["date"] == TODAY)[0][0]) if not args.csv else len(df) - 1
    hist = df.iloc[t - 89:t + 1]
    fut = df.iloc[t + 1:t + 1 + FUTURE_DAYS]
    export["series"] = {"start": str(hist["date"].iloc[0].date()), "history": hist["rate"].round(4).tolist(), "future": fut["rate"].round(4).tolist()}
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "out", "sar_bdt_series.csv"), index=False)
    json.dump(export, open(os.path.join(HERE, "out", "forecast_model.json"), "w"))
    json.dump(report, open(os.path.join(HERE, "out", "forecast_report.json"), "w"), indent=2, default=float)
    print(json.dumps({"backtest": report["backtest"], "h1": report["horizons"][1], "h7": report["horizons"][7]}, indent=2, default=float))


if __name__ == "__main__":
    main()
