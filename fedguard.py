"""
FedGuard-IoT demo: privacy-preserving federated intrusion detection with explanations.

Written in pure NumPy so it runs anywhere (no PyTorch / Flower / Opacus / SHAP needed).
It follows the same pipeline as the project slides:

    edge clients train locally on private data
      -> clip the update + add Gaussian noise (optional differential privacy)
      -> server aggregates with FedAvg or FedProx
      -> global model detects attacks
      -> every alert gets a "top features" explanation

NOTE: the traffic data is SYNTHETIC, shaped like Edge-IIoTset (Normal + 14 attacks).
It is for demonstration only. Replace make_dataset() with the real Edge-IIoTset CSV
for actual results.

Run:  python fedguard.py
"""
import json
import os

import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "tcp.flags.syn", "tcp.connection.rst", "tcp.len", "tcp.dstport.unique",
    "udp.packet_rate", "icmp.packet_rate", "http.request.len", "http.uri.special_chars",
    "http.upload.size", "dns.qry.name.len", "arp.reply_rate", "mqtt.msg_rate",
    "modbus.rare_func_code", "payload.entropy", "login.failed_attempts",
    "outbound.conn_rate", "file.write_rate", "ip.ttl.variance", "sql.keyword_count",
    "flow.duration",
]

# Each attack pushes a few features away from normal traffic (feature index -> shift).
SIGNATURES = {
    "Normal": {},
    "DDoS_TCP_SYN": {0: 4.0, 1: 2.5},
    "DDoS_UDP": {4: 4.0, 19: -2.0},
    "DDoS_HTTP": {6: 2.5, 19: -2.5, 2: 2.0},
    "DDoS_ICMP": {5: 4.0},
    "Port_Scanning": {3: 4.0, 1: 2.0},
    "OS_Fingerprinting": {17: 4.0, 3: 2.0},
    "Vulnerability_Scanner": {7: 2.5, 3: 2.5, 6: 2.0},
    "MITM": {10: 4.0, 9: 3.0},
    "XSS": {7: 4.0, 6: 2.0},
    "SQL_Injection": {18: 4.0, 7: 2.0},
    "Uploading": {8: 4.0, 2: 2.0},
    "Backdoor": {15: 3.5, 13: 2.5},
    "Password_Cracking": {14: 4.0},
    "Ransomware": {16: 4.0, 13: 2.5},
}
CLASSES = list(SIGNATURES)


# ---------------------------------------------------------------- data

def make_dataset(n_per_attack=500, n_normal=3000, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, name in enumerate(CLASSES):
        n = n_normal if name == "Normal" else n_per_attack
        mean = np.zeros(len(FEATURES))
        for f, shift in SIGNATURES[name].items():
            mean[f] = shift
        X.append(rng.normal(mean, 1.0, size=(n, len(FEATURES))))
        y.append(np.full(n, c))
    X, y = np.vstack(X), np.concatenate(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def train_test_split(X, y, test_frac=0.2):
    n_test = int(len(y) * test_frac)
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]


def minmax_fit(X):
    return X.min(0), X.max(0)


def minmax_apply(X, lo, hi):
    return (X - lo) / (hi - lo + 1e-9)


def partition_dirichlet(y, n_clients, alpha, rng):
    """Non-IID split: each class is spread over clients by Dirichlet(alpha).
    Small alpha => each site sees only a few attack types."""
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx, cuts)):
            parts[i].extend(chunk.tolist())
    return [np.array(p, dtype=int) for p in parts]


# ---------------------------------------------------------------- model (small neural net)

def init_model(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def forward(m, X):
    h = np.maximum(0, X @ m["W1"] + m["b1"])
    logits = h @ m["W2"] + m["b2"]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return h, p / p.sum(1, keepdims=True)


def predict_proba(m, X):
    return forward(m, X)[1]


def local_train(model, X, y, epochs, lr, batch=32, mu=0.0, rng=None):
    """SGD on one client's private data. mu > 0 adds the FedProx term
    (mu/2)*||w - w_global||^2, which stops the client drifting too far."""
    m = {k: v.copy() for k, v in model.items()}
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = order[s:s + batch]
            h, p = forward(m, X[b])
            d = p.copy()
            d[np.arange(len(b)), y[b]] -= 1
            d /= len(b)
            dh = d @ m["W2"].T
            dh[h <= 0] = 0
            grads = {"W2": h.T @ d, "b2": d.sum(0), "W1": X[b].T @ dh, "b1": dh.sum(0)}
            for k in m:
                g = grads[k] + mu * (m[k] - model[k]) if mu > 0 else grads[k]
                m[k] -= lr * g
    return m


def privatize(update, clip, sigma, rng):
    """Differential privacy step: clip the update norm to `clip`, then add Gaussian noise."""
    norm = np.sqrt(sum((v ** 2).sum() for v in update.values()))
    scale = min(1.0, clip / (norm + 1e-12))
    return {k: v * scale + rng.normal(0, sigma * clip, v.shape) for k, v in update.items()}


# ---------------------------------------------------------------- evaluation & explanation

def evaluate(m, X, y):
    pred = predict_proba(m, X).argmax(1)
    f1s = []
    for c in range(len(CLASSES)):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f1s))}


def explain(m, x, baseline, top_k=5):
    """SHAP-style attribution (occlusion approximation): replace one feature at a time
    with its normal-traffic value and measure how much the alert confidence drops."""
    p = predict_proba(m, x[None])[0]
    c = int(p.argmax())
    occluded = np.repeat(x[None], len(x), 0)
    occluded[np.arange(len(x)), np.arange(len(x))] = baseline
    drops = p[c] - predict_proba(m, occluded)[:, c]
    top = np.argsort(-drops)[:top_k]
    return {
        "prediction": CLASSES[c],
        "confidence": float(p[c]),
        "top_features": [{"feature": FEATURES[i], "impact": float(drops[i])} for i in top],
    }


# ---------------------------------------------------------------- training runs

def run_centralized(Xtr, ytr, Xte, yte, rounds, lr, seed):
    """Baseline: all raw data collected on one server (what FedGuard avoids)."""
    rng = np.random.default_rng(seed)
    m = init_model(Xtr.shape[1], 64, len(CLASSES), rng)
    history = []
    for _ in range(rounds):
        m = local_train(m, Xtr, ytr, 1, lr, rng=rng)
        history.append(evaluate(m, Xte, yte))
    return m, history


def run_federated(clients, Xte, yte, rounds, local_epochs, lr, mu=0.0, dp=None, seed=0):
    rng = np.random.default_rng(seed)
    g = init_model(Xte.shape[1], 64, len(CLASSES), rng)
    history = []
    for _ in range(rounds):
        updates, sizes = [], []
        for Xc, yc in clients:                      # runs on each edge node
            local = local_train(g, Xc, yc, local_epochs, lr, mu=mu, rng=rng)
            upd = {k: local[k] - g[k] for k in g}   # only weights leave the node
            if dp:
                upd = privatize(upd, dp["clip"], dp["sigma"], rng)
            updates.append(upd)
            sizes.append(len(yc))
        w = np.array(sizes) / sum(sizes)            # server: weighted average
        g = {k: g[k] + sum(wi * u[k] for wi, u in zip(w, updates)) for k in g}
        history.append(evaluate(g, Xte, yte))
    return g, history


def save_chart(histories, client_counts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, hist in histories.items():
        a1.plot(range(1, len(hist) + 1), [h["macro_f1"] for h in hist], label=name, lw=2)
    a1.set(title="Detection quality per round", xlabel="Round", ylabel="Macro F1", ylim=(0, 1))
    a1.grid(alpha=0.3)
    a1.legend()

    bottom = np.zeros(len(client_counts))
    cmap = plt.get_cmap("tab20")
    for c, name in enumerate(CLASSES):
        vals = np.array([cc[c] for cc in client_counts])
        a2.bar([f"Client {i + 1}" for i in range(len(client_counts))], vals, bottom=bottom,
               color=cmap(c), label=name)
        bottom += vals
    a2.set(title="Non-IID data: what each edge site sees", ylabel="Flows")
    a2.legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    rounds, local_epochs, lr, n_clients, alpha, seed = 30, 2, 0.1, 5, 0.3, 42
    rng = np.random.default_rng(seed)

    print("[1] Building synthetic Edge-IIoTset-style traffic ...")
    X, y = make_dataset(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y)
    lo, hi = minmax_fit(Xtr)                         # fit scaler on training split only
    Xtr, Xte = minmax_apply(Xtr, lo, hi), minmax_apply(Xte, lo, hi)
    print(f"    {len(ytr)} training flows, {len(yte)} test flows, "
          f"{len(FEATURES)} features, {len(CLASSES)} classes")

    print(f"[2] Splitting training data over {n_clients} edge clients (non-IID, alpha={alpha}) ...")
    parts = partition_dirichlet(ytr, n_clients, alpha, rng)
    clients = [(Xtr[p], ytr[p]) for p in parts if len(p)]
    client_counts = [np.bincount(ytr[p], minlength=len(CLASSES)).tolist() for p in parts]
    for i, cc in enumerate(client_counts):
        seen = sum(1 for v in cc if v > 20)
        print(f"    Client {i + 1}: {sum(cc):5d} flows, sees {seen:2d} of {len(CLASSES)} classes")

    print("[3] Training ...")
    dp_cfg = {"clip": 1.0, "sigma": 0.01}
    runs = {
        "Centralized (no privacy)": lambda: run_centralized(Xtr, ytr, Xte, yte, rounds, lr, seed),
        "FedAvg": lambda: run_federated(clients, Xte, yte, rounds, local_epochs, lr, seed=seed),
        "FedProx": lambda: run_federated(clients, Xte, yte, rounds, local_epochs, lr, mu=0.01, seed=seed),
        "FedProx + DP": lambda: run_federated(clients, Xte, yte, rounds, local_epochs, lr, mu=0.01,
                                              dp=dp_cfg, seed=seed),
    }
    histories, models = {}, {}
    for name, fn in runs.items():
        models[name], histories[name] = fn()
        print(f"    {name:26s} done")

    print("\n    Final results on held-out test traffic")
    print(f"    {'Setting':26s} {'Accuracy':>9s} {'Macro F1':>9s}")
    for name, hist in histories.items():
        print(f"    {name:26s} {hist[-1]['accuracy']:9.3f} {hist[-1]['macro_f1']:9.3f}")

    final = models["FedProx + DP"]
    baseline = Xtr[ytr == 0].mean(0)                 # "normal traffic" reference point
    print("\n[4] Sample alerts from the FedGuard global model (FedProx + DP) with explanations:")
    alerts, shown = [], set()
    for i in range(len(yte)):
        e = explain(final, Xte[i], baseline)
        if e["prediction"] == "Normal" or e["prediction"] in shown:
            continue
        shown.add(e["prediction"])
        e["true_label"] = CLASSES[yte[i]]
        alerts.append(e)
        if len(alerts) <= 4:
            feats = ", ".join(f["feature"] for f in e["top_features"][:3])
            print(f"    ALERT {e['prediction']:22s} conf={e['confidence']:.2f} "
                  f"(true: {e['true_label']})  because of: {feats}")

    save_chart(histories, client_counts, os.path.join(OUT_DIR, "results.png"))
    np.savez(os.path.join(OUT_DIR, "global_model.npz"), Xte=Xte, yte=yte, baseline=baseline,
             **final)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"histories": histories, "client_counts": client_counts,
                   "classes": CLASSES, "alerts": alerts,
                   "config": {"rounds": rounds, "local_epochs": local_epochs,
                              "clients": n_clients, "alpha": alpha, "dp": dp_cfg}}, f, indent=2)
    print("\n[5] Saved results.png, results.json and global_model.npz")
    print("    Next: run  python dashboard.py  and open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
