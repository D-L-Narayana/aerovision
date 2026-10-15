"""Benchmarks: (1) CLIP zero-shot vs linear probe on UC Merced (21 aerial classes, 2100 images), few-shot curve,
per-class report; (2) deterministic thermal detector precision/recall vs ground truth; (3) CPU latency.
Writes evals/results.json, results.md and exports web assets (class text embeddings, probe head, cached embeddings)."""
from __future__ import annotations
import io, json, statistics, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aerovision.clip_model import ClipEncoder
from aerovision.labels import CLASSES
from aerovision.probe import LinearProbe
from aerovision.thermal import detect_hotspots

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evals"
CACHE = ROOT / "data" / "ucm_clip_embeddings.npz"


def load_ucm():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("blanchon/UC_Merced", "data/train-00000-of-00001.parquet", repo_type="dataset",
                        local_dir=str(ROOT / "data" / "ucm"))
    t = pq.read_table(p).to_pydict()
    return t["image"], np.array(t["label"])


def embed_all(enc: ClipEncoder, images):
    if CACHE.exists():
        d = np.load(CACHE); return d["X"], d["y"]
    X = []
    for s in range(0, len(images), 50):
        ims = [Image.open(io.BytesIO(im["bytes"])) for im in images[s:s + 50]]
        X.append(enc.encode_images(ims)); print(f"  embedded {s + len(ims)}/{len(images)}", flush=True)
    return np.vstack(X)


def main():
    enc = ClipEncoder(local_dir=ROOT.parent / "_dl" / "clip")
    images, y = load_ucm()
    X = embed_all(enc, images) if not CACHE.exists() else np.load(CACHE)["X"]
    np.savez_compressed(CACHE, X=X, y=y)
    T = enc.class_text_embeddings()
    rng = np.random.default_rng(0)

    # stratified 50/50 split per class
    tr, te = [], []
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.nonzero(y == c)[0]); tr += list(idx[:50]); te += list(idx[50:])
    tr, te = np.array(tr), np.array(te)

    # 1) zero-shot (prompt ensemble) on the test split and on all images
    zs = enc.zero_shot(X, T)
    top1_all = float((zs.argmax(1) == y).mean()); top3_all = float(np.mean([y[i] in np.argsort(-zs[i])[:3] for i in range(len(y))]))
    top1_te = float((zs[te].argmax(1) == y[te]).mean())
    single_T = np.stack([enc.encode_texts([f"an aerial photo of {c}."])[0] for c in CLASSES])
    top1_single = float((enc.zero_shot(X, single_T).argmax(1) == y).mean())

    # 2) linear probe + few-shot curve
    probe = LinearProbe().fit(X[tr], y[tr])
    pred = probe.predict(X[te]); acc_probe = float((pred == y[te]).mean())
    per_class = {CLASSES[c]: {"zero_shot": round(float((zs[te][y[te] == c].argmax(1) == c).mean()), 3),
                              "probe": round(float((pred[y[te] == c] == c).mean()), 3)} for c in range(len(CLASSES))}
    curve = {}
    for k in (1, 2, 5, 10, 20, 50):
        accs = []
        for seed in range(3):
            r = np.random.default_rng(seed); sub = np.concatenate([r.choice(tr[y[tr] == c], k, replace=False) for c in range(len(CLASSES))])
            accs.append(float((LinearProbe().fit(X[sub], y[sub]).predict(X[te]) == y[te]).mean()))
        curve[str(k)] = round(statistics.mean(accs), 4)
    probe.export_json(ROOT / "web" / "probe_head.json", CLASSES)

    # 3) thermal detector vs ground truth
    frames = np.load(ROOT / "data" / "thermal" / "frames.npz")["frames"]; truth = json.loads((ROOT / "data" / "thermal" / "truth.json").read_text())
    tp = fp = fn = 0; sev_ok = sev_n = 0; tl = []
    for f, t in zip(frames, truth):
        t0 = time.perf_counter(); spots, _ = detect_hotspots(f); tl.append((time.perf_counter() - t0) * 1000)
        gt = [g for g in t["hotspots"] if g["delta_c"] >= 10]           # SOP: only ≥10 °C is reportable
        matched = set()
        for s in spots:
            hit = next((j for j, g in enumerate(gt) if j not in matched and abs(s.x + s.w / 2 - g["x"] - g["w"] / 2) < 6 and abs(s.y + s.h / 2 - g["y"] - g["h"] / 2) < 6), None)
            if hit is None: fp += 1
            else: matched.add(hit); tp += 1; sev_n += 1; sev_ok += s.severity == gt[hit]["severity"]
        fn += len(gt) - len(matched)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    # 4) latency
    lat = enc.time_per_image_ms(Image.open(io.BytesIO(images[0]["bytes"])))

    out = {"model": enc.model_id, "device": enc.device, "dataset": "UC Merced land use (21 classes x 100, 256px aerial)",
           "n_images": int(len(y)), "split": "stratified 50 train / 50 test per class",
           "zero_shot": {"top1_all": round(top1_all, 4), "top3_all": round(top3_all, 4), "top1_test": round(top1_te, 4),
                         "top1_single_prompt_all": round(top1_single, 4), "templates": 4},
           "linear_probe": {"top1_test": round(acc_probe, 4), "train_per_class": 50, "few_shot_curve": curve},
           "per_class": per_class,
           "thermal_detector": {"frames": len(frames), "precision": round(prec, 4), "recall": round(rec, 4),
                                "severity_agreement": round(sev_ok / max(sev_n, 1), 4), "latency_p50_ms": round(statistics.median(tl), 2)},
           "clip_latency_ms_per_image_cpu": round(lat, 1)}
    (EV / "results.json").write_text(json.dumps(out, indent=2))
    np.save(ROOT / "web" / "text_embs.npy", T)
    json.dump({"classes": CLASSES, "T": np.round(T, 5).tolist()}, open(ROOT / "web" / "text_embs.json", "w"))
    md = [f"# AeroVision eval — {enc.model_id} on UC Merced ({len(y)} images, 21 classes)", "",
          "| method | top-1 | notes |", "|---|---|---|",
          f"| zero-shot, single prompt | {top1_single:.1%} | 'an aerial photo of {{class}}.' |",
          f"| zero-shot, 4-template prompt ensemble | {top1_all:.1%} (top-3 {top3_all:.1%}) | test split {top1_te:.1%} |",
          f"| linear probe on frozen CLIP features | **{acc_probe:.1%}** | 50 labelled images / class |", "",
          "Few-shot probe accuracy (images per class → test top-1, mean of 3 seeds): " + ", ".join(f"{k}→{v:.1%}" for k, v in curve.items()), "",
          f"Thermal detector vs ground truth ({len(frames)} synthetic radiometric frames, ≥10 °C hotspots): precision {prec:.1%}, recall {rec:.1%}, severity agreement {sev_ok / max(sev_n, 1):.1%}, p50 {statistics.median(tl):.2f} ms/frame.", "",
          f"CLIP image encoding: {lat:.0f} ms/image on {enc.device} (2 vCPU)."]
    (EV / "results.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
