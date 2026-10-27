"""
Train HighwayViTModel on Highway2 and evaluate binary jammed / not-jammed transfer to Highway1.

Expects caches from experiments/build_cache.py. Highway1 is never trained on:
  - hw1 train.txt split (12,911 samples, 110 jammed) is used to compare runs
  - hw1 test.txt split (3,473 samples, 143 jammed) is the held-out report
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from radiomana.models import HighwayViTModel

HW2_JAMMED_MIN = 4  # Highway2 classes 4-8 are interference
HW1_JAMMED_MIN = 3  # Highway1 classes 3-10 are interference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-blocks", type=int, default=0)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--normalization", default="per_sample", choices=["absolute", "per_sample"])
    parser.add_argument("--db-min", type=float, default=-10.0)
    parser.add_argument("--db-max", type=float, default=50.0)
    parser.add_argument("--task", default="multi", choices=["multi", "binary"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--augment", default="", help="comma list of: time,freq,flip,gain,synth")
    parser.add_argument("--synth-p", type=float, default=0.15)
    parser.add_argument("--synth-min-db", type=float, default=10.0, help="weakest synthetic jammer, dB above the reference level")
    parser.add_argument("--synth-relative-row", action="store_true", help="measure synthetic tone power from the row's own level instead of the global median")
    parser.add_argument("--spur-p", type=float, default=0.0, help="chance of painting a faint benign spur onto a clean sample (label stays clean)")
    parser.add_argument("--balance", action="store_true", help="oversample jammed samples to ~25% of each batch")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_cache(cache: Path, name: str, subset: str):
    x = np.load(cache / f"{name}_{subset}_x.npy", mmap_mode="r")
    y = np.load(cache / f"{name}_{subset}_y.npy")
    return x, y


def augment(x, y_bin, kinds, args, gen):
    """x: (B, 224, 224) dB with freq on dim 1, time on dim 2. Returns augmented x and binary labels."""
    bsz = x.shape[0]
    dev = x.device
    if "time" in kinds:
        shifts = torch.randint(0, x.shape[2], (bsz,), device=dev, generator=gen)
        idx = (torch.arange(x.shape[2], device=dev)[None, :] + shifts[:, None]) % x.shape[2]
        x = torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    if "freq" in kinds:
        shifts = torch.randint(-24, 25, (bsz,), device=dev, generator=gen)
        idx = (torch.arange(x.shape[1], device=dev)[None, :] + shifts[:, None]) % x.shape[1]
        x = torch.gather(x, 1, idx[:, :, None].expand(-1, -1, x.shape[2]))
    if "flip" in kinds:
        flip = torch.rand(bsz, device=dev, generator=gen) < 0.5
        x = torch.where(flip[:, None, None], x.flip(1), x)
    if "gain" in kinds:
        x = x + (torch.rand(bsz, 1, 1, device=dev, generator=gen) * 30 - 15)
    if "synth" in kinds:
        x, y_bin = inject_synthetic(x, y_bin, args.synth_p, gen, args.synth_min_db, args.synth_relative_row)
    if args.spur_p > 0:
        x = inject_benign_spurs(x, y_bin, args.spur_p, gen)
    return x, y_bin


def inject_benign_spurs(x, y_bin, p, gen):
    """
    Paint 1-2 faint (2-7 dB), full-duration, 1-bin lines onto clean samples without changing the label.

    Receivers at other sites pick up weak constant spurs (clock harmonics etc.) that are not jammers;
    Highway2 backgrounds have almost none, so without this the tone detector fires on them.
    """
    dev = x.device
    bsz, n_freq, n_time = x.shape
    chosen = ((torch.rand(bsz, device=dev, generator=gen) < p) & (y_bin == 0)).nonzero().flatten().tolist()
    if not chosen:
        return x
    x = x.clone()
    for idx in chosen:
        for _ in range(int(torch.randint(1, 3, (1,), generator=gen, device=dev))):
            f0 = int(torch.randint(0, n_freq, (1,), generator=gen, device=dev))
            level = x[idx, f0].median() + 2 + 5 * torch.rand(1, device=dev, generator=gen).item()
            x[idx, f0] = torch.logaddexp(x[idx, f0] / 4.343, level / 4.343 + torch.randn_like(x[idx, f0]) * 0.2) * 4.343
    return x


def inject_synthetic(x, y_bin, p, gen, min_db=10.0, relative_row=False):
    """
    Turn a fraction of clean samples into synthetic narrowband (tone) or pulsed interference.

    Highway2 only contains chirp and cigarette-lighter jammers; Highway1 also has CW tones and
    pulsed jammers. Painting simple versions of those onto Highway2 backgrounds teaches the
    model "a strong structured emitter above the noise floor = jammed" without touching Highway1.
    """
    dev = x.device
    bsz, n_freq, n_time = x.shape
    chosen = (torch.rand(bsz, device=dev, generator=gen) < p) & (y_bin == 0)
    if not chosen.any():
        return x, y_bin
    x = x.clone()
    floor = x.flatten(1).median(dim=1).values
    for idx in chosen.nonzero().flatten().tolist():
        power = floor[idx] + min_db + 25 * torch.rand(1, device=dev, generator=gen).item()
        kind = torch.randint(0, 3, (1,), generator=gen, device=dev).item()
        if kind == 0:  # CW tone: 1-3 bin horizontal line, full or partial duration
            width = int(torch.randint(1, 4, (1,), generator=gen, device=dev))
            f0 = int(torch.randint(0, n_freq - width, (1,), generator=gen, device=dev))
            if relative_row:
                power = power - floor[idx] + x[idx, f0 : f0 + width].median()
            t0 = int(torch.randint(0, n_time // 2, (1,), generator=gen, device=dev)) if torch.rand(1, generator=gen, device=dev) < 0.3 else 0
            band = x[idx, f0 : f0 + width, t0:]
            x[idx, f0 : f0 + width, t0:] = torch.maximum(band, power + torch.randn_like(band) * 2)
        elif kind == 1:  # pulsed: periodic broadband vertical lines
            period = int(torch.randint(4, 40, (1,), generator=gen, device=dev))
            phase = int(torch.randint(0, period, (1,), generator=gen, device=dev))
            f_lo = int(torch.randint(0, n_freq // 2, (1,), generator=gen, device=dev))
            f_hi = int(torch.randint(f_lo + n_freq // 4, n_freq + 1, (1,), generator=gen, device=dev))
            cols = torch.arange(phase, n_time, period, device=dev)
            band = x[idx, f_lo:f_hi][:, cols]
            x[idx, f_lo:f_hi, cols] = torch.maximum(band, power - 5 + torch.randn_like(band) * 2)
        else:  # wideband noise jammer: raise a contiguous band for part or all of the snapshot
            width = int(torch.randint(n_freq // 16, n_freq // 3, (1,), generator=gen, device=dev))
            f0 = int(torch.randint(0, n_freq - width, (1,), generator=gen, device=dev))
            t0 = int(torch.randint(0, n_time // 2, (1,), generator=gen, device=dev)) if torch.rand(1, generator=gen, device=dev) < 0.5 else 0
            band = x[idx, f0 : f0 + width, t0:]
            x[idx, f0 : f0 + width, t0:] = torch.maximum(band, power - 5 + torch.randn_like(band) * 3)
        y_bin[idx] = 1
    return x, y_bin


def param_groups(model, args):
    """layer-wise lr decay: head gets head_lr, block i gets lr * decay^(12 - i), embeddings the smallest."""
    vit = model.submodel
    n_layers = len(vit.encoder.layers)
    groups = {}
    for name, param in vit.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("heads."):
            key, lr = "head", args.head_lr
        elif name.startswith("encoder.layers.encoder_layer_"):
            layer = int(name.split(".")[2].rsplit("_", 1)[1])
            key, lr = f"l{layer}", args.lr * args.layer_decay ** (n_layers - layer)
        elif name.startswith("encoder.ln"):
            key, lr = "final_ln", args.lr
        else:
            key, lr = "embed", args.lr * args.layer_decay ** (n_layers + 1)
        no_wd = param.ndim <= 1 or name.endswith("pos_embedding") or name.endswith("class_token")
        gkey = key + ("_nowd" if no_wd else "")
        groups.setdefault(gkey, {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else args.weight_decay})
        groups[gkey]["params"].append(param)
    return list(groups.values())


@torch.no_grad()
def predict(model, x_np, batch_size=128):
    """returns softmax probabilities, computed in chunks from the memmapped cache"""
    model.eval()
    probs = []
    for start in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(np.asarray(x_np[start : start + batch_size])).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            probs.append(torch.softmax(model(xb).float(), dim=1).cpu())
    model.train()
    return torch.cat(probs).numpy()


def jam_prob(probs, task):
    return probs[:, 1] if task == "binary" else probs[:, HW2_JAMMED_MIN:].sum(1)


def binary_report(p_jam, y_true_bin, threshold=0.5):
    pred = p_jam >= threshold
    tp = int((pred & (y_true_bin == 1)).sum())
    tn = int((~pred & (y_true_bin == 0)).sum())
    fp = int((pred & (y_true_bin == 0)).sum())
    fn = int((~pred & (y_true_bin == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "acc": (tp + tn) / len(pred),
        "bal_acc": (recall + spec) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": spec,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "auroc": roc_auc_score(y_true_bin, p_jam),
        "ap": average_precision_score(y_true_bin, p_jam),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def best_f1_threshold(p_jam, y_true_bin):
    candidates = np.unique(np.quantile(p_jam, np.linspace(0.5, 0.9999, 2000)))
    scores = [binary_report(p_jam, y_true_bin, t)["f1"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def per_class_recall(p_jam, y_true, threshold=0.5):
    return {int(c): round(float((p_jam[y_true == c] >= threshold).mean()), 3) for c in np.unique(y_true)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = set(filter(None, args.augment.split(",")))

    hw2_x, hw2_y = load_cache(args.cache, "fiot_highway2_f32", "train")
    hw2_test_x, hw2_test_y = load_cache(args.cache, "fiot_highway2_f32", "test")
    hw1_sel_x, hw1_sel_y = load_cache(args.cache, "fiot_highway1_subset", "train")
    hw1_test_x, hw1_test_y = load_cache(args.cache, "fiot_highway1_subset", "test")

    # same stratified 90/10 split as HighwayDataModule
    train_idx, val_idx = train_test_split(np.arange(len(hw2_y)), test_size=0.1, stratify=hw2_y, random_state=0xD00D1E)
    train_x = torch.from_numpy(np.asarray(hw2_x[np.sort(train_idx)]))  # ~1.2 GB fp16 in RAM
    train_y = torch.from_numpy(hw2_y[np.sort(train_idx)])
    val_x, val_y = hw2_x[np.sort(val_idx)], hw2_y[np.sort(val_idx)]

    num_classes = 2 if args.task == "binary" else 9
    model = HighwayViTModel(
        num_classes=num_classes,
        freeze_backbone=args.linear_probe,
        db_min=args.db_min,
        db_max=args.db_max,
        pretrained=args.pretrained,
        freeze_blocks=args.freeze_blocks,
        normalization=args.normalization,
    ).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.name}] trainable params {trainable / 1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(param_groups(model, args))
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    scaler = torch.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    is_jammed = (train_y >= HW2_JAMMED_MIN).numpy()
    if args.balance:
        # jammed samples drawn ~25% of the time instead of ~7.5%
        weights = np.where(is_jammed, 0.25 / is_jammed.sum(), 0.75 / (~is_jammed).sum())
        sampler_weights = torch.from_numpy(weights)
    steps_per_epoch = len(train_y) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = min(300, total_steps // 10)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if args.balance:
            order = torch.multinomial(sampler_weights, len(train_y), replacement=True)
        else:
            order = torch.randperm(len(train_y))
        running = []
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size : (b + 1) * args.batch_size]
            xb = train_x[idx].cuda(non_blocking=True).float()
            yb = train_y[idx].cuda(non_blocking=True)
            yb_bin = (yb >= HW2_JAMMED_MIN).long()
            if kinds:
                xb, yb_bin_aug = augment(xb, yb_bin.clone(), kinds, args, gen)
            else:
                yb_bin_aug = yb_bin
            if args.task == "binary":
                target = yb_bin_aug
            else:
                # synthetic jammers have no Highway2 class; label them chirp-high-distance-like generic class 4
                target = torch.where(yb_bin_aug > yb_bin, torch.full_like(yb, HW2_JAMMED_MIN), yb)

            lr_scale = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(xb), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running.append(loss.item())
            step += 1

        val_probs = predict(model, val_x)
        val_bin = binary_report(jam_prob(val_probs, args.task), (val_y >= HW2_JAMMED_MIN).astype(int))
        sel_p = jam_prob(predict(model, hw1_sel_x), args.task)
        sel = binary_report(sel_p, (hw1_sel_y >= HW1_JAMMED_MIN).astype(int))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "hw2_val_multi_acc": float((val_probs.argmax(1) == val_y).mean()) if args.task == "multi" else None,
            "hw2_val_bal_acc": val_bin["bal_acc"],
            "hw1_sel": sel,
            "minutes": (time.time() - t0) / 60,
        }
        history.append(record)
        print(
            f"[{args.name}] ep {epoch} loss {record['train_loss']:.3f} | hw2 val bal {val_bin['bal_acc']:.3f}"
            f" | hw1-sel bal {sel['bal_acc']:.3f} R {sel['recall']:.3f} spec {sel['specificity']:.4f} F1 {sel['f1']:.3f}"
            f" AUROC {sel['auroc']:.3f} AP {sel['ap']:.3f} | {record['minutes']:.1f} min",
            flush=True,
        )

    # final evaluation
    test_probs = predict(model, hw2_test_x)
    hw2_test = binary_report(jam_prob(test_probs, args.task), (hw2_test_y >= HW2_JAMMED_MIN).astype(int))
    hw2_test["multi_acc"] = float((test_probs.argmax(1) == hw2_test_y).mean()) if args.task == "multi" else None
    test_p = jam_prob(predict(model, hw1_test_x), args.task)
    hw1_test_bin = (hw1_test_y >= HW1_JAMMED_MIN).astype(int)
    sel_bin = (hw1_sel_y >= HW1_JAMMED_MIN).astype(int)
    tuned_t = best_f1_threshold(sel_p, sel_bin)
    all_p = np.concatenate([sel_p, test_p])
    all_y = np.concatenate([hw1_sel_y, hw1_test_y])
    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "history": history,
        "hw2_test": hw2_test,
        "hw1_sel": history[-1]["hw1_sel"],
        "hw1_test": binary_report(test_p, hw1_test_bin),
        "hw1_all": binary_report(all_p, (all_y >= HW1_JAMMED_MIN).astype(int)),
        "hw1_all_class_recall": per_class_recall(all_p, all_y),
        "tuned_threshold": tuned_t,
        "hw1_test_tuned": binary_report(test_p, hw1_test_bin, tuned_t),
    }
    (args.out / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    np.save(args.out / f"{args.name}_hw1_p.npy", all_p)
    torch.save(model.state_dict(), args.out / f"{args.name}.pt")
    h1, ha = result["hw1_test"], result["hw1_all"]
    print(f"[{args.name}] HW2 test bal {hw2_test['bal_acc']:.3f} multi {hw2_test['multi_acc']}")
    print(f"[{args.name}] HW1 test: acc {h1['acc']:.4f} bal {h1['bal_acc']:.3f} P {h1['precision']:.3f} R {h1['recall']:.3f} F1 {h1['f1']:.3f} AUROC {h1['auroc']:.3f} AP {h1['ap']:.3f}")
    print(f"[{args.name}] HW1 all : acc {ha['acc']:.4f} bal {ha['bal_acc']:.3f} P {ha['precision']:.3f} R {ha['recall']:.3f} F1 {ha['f1']:.3f} AUROC {ha['auroc']:.3f} AP {ha['ap']:.3f}")
    print(f"[{args.name}] HW1 class recall {result['hw1_all_class_recall']}")
    print(f"[{args.name}] tuned thr {tuned_t:.4f} -> HW1 test F1 {result['hw1_test_tuned']['f1']:.3f} bal {result['hw1_test_tuned']['bal_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
