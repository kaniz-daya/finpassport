"""Repeat the federated fraud experiment over 5 seeds and report mean and std."""
import json, os, io, contextlib
import numpy as np
import federated_fraud as ff

rows = {}
for seed in [7, 11, 23, 42, 99]:
    ff.rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        ff.main()
    rep = json.load(open(os.path.join(ff.HERE, "out", "fraud_report.json")))
    for k, v in rep.items():
        rows.setdefault(k, []).append([v["roc_auc"], v["pr_auc"], v["recall_at_1pct_fpr"]])
summary = {}
for k, v in rows.items():
    a = np.array(v)
    summary[k] = {"roc_auc": [a[:, 0].mean(), a[:, 0].std()], "pr_auc": [a[:, 1].mean(), a[:, 1].std()], "recall_at_1pct_fpr": [a[:, 2].mean(), a[:, 2].std()]}
    print(f"{k:10s} AUC {a[:,0].mean():.3f}±{a[:,0].std():.3f}  PR-AUC {a[:,1].mean():.3f}±{a[:,1].std():.3f}  recall@1%FPR {a[:,2].mean():.2f}±{a[:,2].std():.2f}")
json.dump(summary, open(os.path.join(ff.HERE, "out", "fraud_report_5seeds.json"), "w"), indent=2)
# leave the exported model from the default seed
ff.rng = np.random.default_rng(7)
with contextlib.redirect_stdout(io.StringIO()):
    ff.main()
