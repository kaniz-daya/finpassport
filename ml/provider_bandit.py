"""
FinPassport - Model 3: learning which provider really delivers the most.

Problem
  A quote is not what the family receives. Some providers lose money on the way:
  a cash-out fee on mobile wallets, a weekend intermediary rate, a compliance
  charge on large amounts, a flat fee taken by the receiving bank. These hidden
  deductions depend on the context of the transfer and are not in the quote.

Approach: contextual Thompson sampling
  For each provider, a Bayesian linear regression predicts the hidden deduction
  (share of the quoted amount that never arrives) from the context:
      x = [1, mobile_wallet_payout, weekend, log(amount / 1000 SAR), amount > 2000 SAR]
  After each transfer, the delivered amount confirmed by the family updates the
  posterior of the provider that was used. To choose, the agent samples
  deductions from each posterior and picks the provider with the highest
  sampled delivered amount, which balances trying providers and using the best.

Simulation (seeded)
  2,000 transfers by a community of students, contexts drawn from realistic
  ranges, true hidden deductions as below (unknown to the agent).
  Baselines: best quote (ignores hidden deductions), random, oracle.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MID = 32.46
PROVIDERS = [
    {"id": "gulfx", "name": "GulfXpress", "fee": 10, "margin": 0.004},
    {"id": "quick", "name": "QuickRemit", "fee": 15, "margin": 0.006},
    {"id": "desh", "name": "DeshLink", "fee": 0, "margin": 0.015},
    {"id": "bank", "name": "Bank wire", "fee": 25, "margin": 0.012},
]
HIDDEN = {  # true hidden deduction, share of quoted BDT; unknown to the agent
    "gulfx": lambda c: 0.0 + 0.009 * c["big"] + 0.004 * c["wallet"],
    "quick": lambda c: 0.001 + 0.011 * c["wallet"],
    "desh": lambda c: 0.0 + 0.007 * c["weekend"],
    "bank": lambda c: 350.0 / (c["amount"] * MID),  # receiving bank takes ~BDT 350
}
NOISE = 0.0015


def ctx_vec(c):
    return np.array([1.0, c["wallet"], c["weekend"], np.log(c["amount"] / 1000.0), c["big"]])


def quote(p, amount):
    return max(0.0, amount - p["fee"]) * MID * (1 - p["margin"])


def sample_ctx(rng):
    amount = float(np.clip(rng.lognormal(np.log(1000), 0.6), 150, 6000))
    return {"amount": amount, "wallet": float(rng.random() < 0.3), "weekend": float(rng.random() < 2 / 7), "big": float(amount > 2000)}


class BLR:
    def __init__(self, d=5, prior_sd=0.01, noise=NOISE):
        self.A = np.eye(d) / prior_sd ** 2
        self.b = np.zeros(d)
        self.noise = noise

    def update(self, x, y):
        self.A += np.outer(x, x) / self.noise ** 2
        self.b += x * y / self.noise ** 2

    def mean(self):
        return np.linalg.solve(self.A, self.b)

    def sample(self, rng):
        cov = np.linalg.inv(self.A)
        return rng.multivariate_normal(self.mean(), cov)


def simulate(policy, n, seed):
    rng = np.random.default_rng(seed)
    models = {p["id"]: BLR() for p in PROVIDERS}
    regret, got, best, counts = [], 0.0, 0.0, {p["id"]: 0 for p in PROVIDERS}
    for i in range(n):
        c = sample_ctx(rng); x = ctx_vec(c)
        true = {p["id"]: quote(p, c["amount"]) * (1 - HIDDEN[p["id"]](c)) for p in PROVIDERS}
        if policy == "thompson":
            scores = {p["id"]: quote(p, c["amount"]) * (1 - float(models[p["id"]].sample(rng) @ x)) for p in PROVIDERS}
        elif policy == "best_quote":
            scores = {p["id"]: quote(p, c["amount"]) for p in PROVIDERS}
        elif policy == "random":
            scores = {p["id"]: rng.random() for p in PROVIDERS}
        else:  # oracle
            scores = true
        k = max(scores, key=scores.get)
        counts[k] += 1
        observed_cut = HIDDEN[k](c) + rng.normal(0, NOISE)
        models[k].update(x, observed_cut)
        delivered = quote(next(p for p in PROVIDERS if p["id"] == k), c["amount"]) * (1 - observed_cut)
        got += delivered; best += max(true.values())
        regret.append((max(true.values()) - true[k]) / c["amount"] * 1000)  # BDT lost per 1,000 SAR
    return np.array(regret), models, counts


def main():
    n = 2000
    out = {}
    curves = {}
    for policy in ("best_quote", "random", "thompson"):
        runs = [simulate(policy, n, s)[0] for s in range(5)]
        R = np.vstack(runs)
        out[policy] = {
            "avg_loss_vs_oracle_bdt_per_1000_sar_all": float(R.mean()),
            "avg_loss_first_100": float(R[:, :100].mean()),
            "avg_loss_last_500": float(R[:, -500:].mean()),
            "std_last_500_across_seeds": float(R[:, -500:].mean(1).std()),
        }
        cum = R.cumsum(1).mean(0)
        curves[policy] = [float(cum[i]) for i in range(49, n, 50)]
        print(policy, json.dumps({k: round(v, 2) for k, v in out[policy].items()}))
    # learned state after the community history (exported for the app)
    _, models, counts = simulate("thompson", n, 0)
    export = {"context": ["intercept", "mobile_wallet_payout", "weekend", "log_amount_per_1000", "amount_over_2000"],
              "noise": NOISE, "prior_sd": 0.01, "receipts": n, "counts": counts,
              "providers": {k: {"A": m.A.tolist(), "b": m.b.tolist(), "mean": m.mean().tolist()} for k, m in models.items()}}
    for k, m in models.items():
        print(k, "learned weights", np.round(m.mean(), 4).tolist(), "used", counts[k])
    json.dump(export, open(os.path.join(HERE, "out", "provider_model.json"), "w"))
    json.dump({"policies": out, "cumulative_regret_every_50": curves, "transfers": n, "seeds": 5},
              open(os.path.join(HERE, "out", "provider_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
