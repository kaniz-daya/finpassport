"""
FinPassport - Model 2: federated fraud detection across remittance providers.

Problem
  Each remittance provider sees only its own customers. A scam pattern that is
  rare at every single provider is common across all of them, but providers
  cannot pool customer data (privacy law, competition, security).

Data (synthetic, seeded)
  Three providers with different customer bases and different fraud mixes
  (non-IID):
    A (large, students):   mostly "tuition-deposit" scams
    B (medium, workers):   mostly account takeovers
    C (small, wallets):    mostly money-mule transfers
  10 features per transfer, all computable by the agent at send time.

Models (same 10-16-1 neural network everywhere)
  * Local only   - each provider trains on its own data
  * FedAvg       - providers train locally, a coordinator averages weights
                   (McMahan et al., 2017); raw data never leaves a provider
  * FedAvg + DP  - each update is clipped and Gaussian noise is added before
                   averaging (DP-style protection of individual updates)
  * Pooled       - all data in one place (privacy-violating upper bound)

All models are evaluated on the same held-out test set that contains every
fraud pattern. The FedAvg model is exported to JSON for the browser, with
feature statistics for explanation (reason codes by occlusion).
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["amount_vs_usual", "new_recipient", "recipient_account_new", "night_time",
            "transfers_last_24h", "new_device", "risky_memo_words", "mobile_wallet_payout",
            "first_time_customer", "amount_round_thousand"]
REASONS = {
    "amount_vs_usual": "amount much higher than usual",
    "new_recipient": "first transfer to this recipient",
    "recipient_account_new": "recipient account opened recently",
    "night_time": "sent late at night",
    "transfers_last_24h": "many transfers in 24 hours",
    "new_device": "new or unknown device",
    "risky_memo_words": "words used in known scams (deposit, agent fee, visa)",
    "mobile_wallet_payout": "cash-out to a mobile wallet",
    "first_time_customer": "new customer",
    "amount_round_thousand": "round amount often used in scams",
}

rng = np.random.default_rng(7)


def legit(n, profile):
    X = np.zeros((n, 10))
    X[:, 0] = np.clip(rng.lognormal(0.05, 0.55, n), 0.15, 8)       # amount / usual
    X[:, 1] = rng.random(n) < profile["new_rec"]                      # legit users add new people sometimes
    X[:, 2] = (X[:, 1] == 1) & (rng.random(n) < 0.35)
    X[:, 3] = rng.random(n) < 0.12
    X[:, 4] = rng.poisson(profile.get("burst", 0.6), n)
    X[:, 5] = rng.random(n) < 0.08
    X[:, 6] = rng.random(n) < profile["memo"]                         # "tuition deposit" is also legit
    X[:, 7] = rng.random(n) < profile["wallet"]
    X[:, 8] = rng.random(n) < 0.12
    X[:, 9] = rng.random(n) < 0.3
    return X


def fraud(n, kind):
    X = np.zeros((n, 10))
    if kind == "tuition_scam":       # fake "university agent" asks for an admission deposit
        X[:, 0] = rng.uniform(1.0, 3.0, n); X[:, 1] = rng.random(n) < 0.85; X[:, 2] = rng.random(n) < 0.5
        X[:, 3] = rng.random(n) < 0.12; X[:, 4] = rng.poisson(0.7, n); X[:, 5] = rng.random(n) < 0.08
        X[:, 6] = rng.random(n) < 0.7; X[:, 7] = rng.random(n) < 0.3; X[:, 8] = rng.random(n) < 0.15
        X[:, 9] = rng.random(n) < 0.75
    elif kind == "takeover":         # stolen login: new device, at night, big amount, often to existing payees
        X[:, 0] = rng.uniform(1.5, 6.0, n); X[:, 1] = rng.random(n) < 0.3; X[:, 2] = rng.random(n) < 0.15
        X[:, 3] = rng.random(n) < 0.6; X[:, 4] = rng.poisson(1.2, n); X[:, 5] = rng.random(n) < 0.75
        X[:, 6] = rng.random(n) < 0.05; X[:, 7] = rng.random(n) < 0.3; X[:, 8] = rng.random(n) < 0.05
        X[:, 9] = rng.random(n) < 0.35
    else:                            # money mule: bursts of small transfers to brand-new wallets
        X[:, 0] = rng.uniform(0.3, 1.3, n); X[:, 1] = rng.random(n) < 0.6; X[:, 2] = rng.random(n) < 0.6
        X[:, 3] = rng.random(n) < 0.2; X[:, 4] = rng.poisson(3.0, n); X[:, 5] = rng.random(n) < 0.15
        X[:, 6] = rng.random(n) < 0.05; X[:, 7] = rng.random(n) < 0.9; X[:, 8] = rng.random(n) < 0.5
        X[:, 9] = rng.random(n) < 0.3
    return X


PROVIDERS = {
    "A": dict(n=6000, profile=dict(new_rec=0.15, memo=0.10, wallet=0.25, burst=0.5), mix={"tuition_scam": 60, "takeover": 0, "mule": 2}),
    "B": dict(n=4000, profile=dict(new_rec=0.10, memo=0.03, wallet=0.35, burst=0.6), mix={"tuition_scam": 2, "takeover": 45, "mule": 0}),
    "C": dict(n=3000, profile=dict(new_rec=0.25, memo=0.03, wallet=0.8, burst=1.2), mix={"tuition_scam": 0, "takeover": 2, "mule": 40}),
}


def make(p):
    X = [legit(p["n"], p["profile"])]
    y = [np.zeros(p["n"])]
    for k, m in p["mix"].items():
        if m:
            X.append(fraud(m, k)); y.append(np.ones(m))
    X = np.vstack(X); y = np.concatenate(y)
    i = rng.permutation(len(y))
    return X[i], y[i]


def transform(X):
    Z = X.copy().astype(float)
    Z[:, 0] = np.log(Z[:, 0])
    Z[:, 4] = np.log1p(Z[:, 4])
    return Z


# ------------------------------------------------------------ tiny neural net
def init(seed=0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 0.3, (10, 16)), "b1": np.zeros(16), "W2": r.normal(0, 0.3, (16, 1)), "b2": np.zeros(1)}


def forward(p, Z):
    h = np.tanh(Z @ p["W1"] + p["b1"])
    return 1 / (1 + np.exp(-(h @ p["W2"] + p["b2"]))).ravel(), h


def train(p, Z, y, epochs=3, lr=0.05, batch=64, pos_weight=10.0):
    p = {k: v.copy() for k, v in p.items()}
    n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            out, h = forward(p, Z[b])
            w = np.where(y[b] == 1, pos_weight, 1.0)
            g = ((out - y[b]) * w)[:, None] / len(b)
            gW2 = h.T @ g; gb2 = g.sum(0)
            gh = (g @ p["W2"].T) * (1 - h ** 2)
            gW1 = Z[b].T @ gh; gb1 = gh.sum(0)
            for k, gk in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)):
                p[k] -= lr * gk
    return p


def flat(p):
    return np.concatenate([p[k].ravel() for k in ("W1", "b1", "W2", "b2")])


def unflat(v, like):
    out, i = {}, 0
    for k in ("W1", "b1", "W2", "b2"):
        n = like[k].size
        out[k] = v[i:i + n].reshape(like[k].shape); i += n
    return out


def fedavg(parts, rounds=40, local_epochs=2, dp=None):
    g = init(1)
    sizes = np.array([len(y) for _, y in parts], float)
    for _ in range(rounds):
        deltas = []
        for Z, y in parts:
            local = train(g, Z, y, epochs=local_epochs)
            d = flat(local) - flat(g)
            if dp:
                norm = np.linalg.norm(d)
                d = d * min(1.0, dp["clip"] / (norm + 1e-12))
                d = d + rng.normal(0, dp["noise"] * dp["clip"], d.shape)
            deltas.append(d)
        avg = np.average(np.vstack(deltas), axis=0, weights=sizes)
        g = unflat(flat(g) + avg, g)
    return g


def metrics(p, Z, y):
    s, _ = forward(p, Z)
    neg = np.sort(s[y == 0])
    thr = neg[int(0.99 * len(neg))]  # threshold at 1% false-positive rate
    rec = {}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "recall_at_1pct_fpr": float(np.mean(s[y == 1] > thr))}


def recall_by_pattern(p, mu, sd):
    out = {}
    for k in ("tuition_scam", "takeover", "mule"):
        Xf = fraud(400, k); Xl = legit(4000, dict(new_rec=0.15, memo=0.05, wallet=0.4, burst=0.7))
        Zf = (transform(Xf) - mu) / sd; Zl = (transform(Xl) - mu) / sd
        sl, _ = forward(p, Zl); sf, _ = forward(p, Zf)
        thr = np.sort(sl)[int(0.99 * len(sl))]
        out[k] = float(np.mean(sf > thr))
    return out


def main():
    data = {k: make(v) for k, v in PROVIDERS.items()}
    # common held-out test set with every pattern
    test_parts = [make(dict(n=v["n"] // 2, profile=v["profile"], mix={k2: 40 for k2 in v["mix"]})) for v in PROVIDERS.values()]
    Xte = np.vstack([x for x, _ in test_parts]); yte = np.concatenate([y for _, y in test_parts])
    allX = np.vstack([x for x, _ in data.values()])
    mu = transform(allX).mean(0); sd = transform(allX).std(0) + 1e-9  # in practice: agreed public scaling or secure aggregation of stats
    Z = {k: ((transform(x) - mu) / sd, y) for k, (x, y) in data.items()}
    Zte = (transform(Xte) - mu) / sd

    results = {}
    for k, (Zk, yk) in Z.items():
        m = train(init(1), Zk, yk, epochs=80)
        results[f"local_{k}"] = metrics(m, Zte, yte) | {"by_pattern": recall_by_pattern(m, mu, sd)}
    fed = fedavg(list(Z.values()), rounds=40, local_epochs=2)
    results["fedavg"] = metrics(fed, Zte, yte) | {"by_pattern": recall_by_pattern(fed, mu, sd)}
    fed_dp = fedavg(list(Z.values()), rounds=40, local_epochs=2, dp={"clip": 1.0, "noise": 0.05})
    results["fedavg_dp"] = metrics(fed_dp, Zte, yte) | {"by_pattern": recall_by_pattern(fed_dp, mu, sd)}
    Zall = np.vstack([z for z, _ in Z.values()]); yall = np.concatenate([y for _, y in Z.values()])
    pooled = train(init(1), Zall, yall, epochs=80)
    results["pooled"] = metrics(pooled, Zte, yte) | {"by_pattern": recall_by_pattern(pooled, mu, sd)}

    for k, v in results.items():
        print(f"{k:10s} AUC {v['roc_auc']:.3f}  PR-AUC {v['pr_auc']:.3f}  recall@1%FPR {v['recall_at_1pct_fpr']:.2f}  by pattern {json.dumps({a: round(b, 2) for a, b in v['by_pattern'].items()})}")

    # thresholds for the agent, chosen on the test set legit distribution
    s, _ = forward(fed, Zte)
    neg = np.sort(s[yte == 0])
    t_review = float(neg[int(0.98 * len(neg))]); t_block = float(max(0.9, neg[int(0.999 * len(neg))]))
    baseline = ((transform(legit(5000, dict(new_rec=0.15, memo=0.05, wallet=0.3, burst=0.6))) - mu) / sd).mean(0)
    export = {"features": FEATURES, "reasons": REASONS, "mu": mu.tolist(), "sd": sd.tolist(),
              "W1": fed["W1"].tolist(), "b1": fed["b1"].tolist(), "W2": fed["W2"].ravel().tolist(), "b2": float(fed["b2"][0]),
              "baseline_z": baseline.tolist(), "thresholds": {"review": round(max(0.5, t_review), 3), "block": round(t_block, 3)},
              "training": {"providers": {k: {"transfers": int(len(v[1])), "fraud": int(v[1].sum())} for k, v in data.items()},
                           "rounds": 40, "local_epochs": 2}}
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(export, open(os.path.join(HERE, "out", "fraud_model.json"), "w"))
    json.dump(results, open(os.path.join(HERE, "out", "fraud_report.json"), "w"), indent=2)
    print("thresholds", export["thresholds"])


if __name__ == "__main__":
    main()
