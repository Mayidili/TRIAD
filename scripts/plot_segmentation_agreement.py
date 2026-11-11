"""
Agreement of MedSAM2 / official masks with the MAMA-MIA expert masks.
(a) Dice box plots per cohort; (b) three representative cases per cohort (Dice at the cohort's 25th, 50th and
    75th percentile): expert contour in green, MedSAM2 in magenta, official I-SPY mask as a dashed yellow line;
    the slice with the largest expert-mask area is shown, cropped around the tumour.
Font: Times New Roman. Output: <figures>/figS_expert_agreement.{pdf,png}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import segmentation_qa as q  # noqa: E402
from prism.paths import FIGURES_DIR, FONT_DIR  # noqa: E402

TNR = os.path.expanduser(FONT_DIR)
for f in os.listdir(TNR):
    font_manager.fontManager.addfont(os.path.join(TNR, f))
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6})
OUT = FIGURES_DIR
COL = {"medsam2": "#c51b7d", "official": "#e6ab02", "expert": "#1b9e77"}
NAME = {"Duke": "Duke", "ISPY1": "I-SPY1", "ISPY2": "I-SPY2"}


def load_case(case, coh, pid):
    phase = "0002" if coh == "ISPY2" else "0001"
    theirs = q.arr(os.path.join(q.MM, "images", case, f"{case}_{phase}.nii.gz"))
    expert = q.arr(os.path.join(q.MM, "segmentations/expert", f"{case}.nii.gz")) > 0
    if coh == "Duke":
        img = q.arr(os.path.join(q.PD, "seg_images", f"{pid}_image.nii.gz"))
        masks = {"medsam2": q.arr(os.path.join(q.PD, "seg_masks", f"{pid}_mask.nii.gz")) > 0}
    else:
        img = q.arr(os.path.join(q.FIX, coh, "early", f"{pid}.nii.gz"))
        masks = {k: q.arr(os.path.join(q.FIX, coh, f"mask_{k}", f"{pid}.nii.gz")) > 0 for k in ["medsam2", "official"]}
    r, perm, ax = q.best_transform(img, theirs)
    assert r >= 0.99, (case, r)
    return img, q.apply(expert, perm, ax), masks


def slice_axis(coh):
    return 2 if coh == "ISPY1" else 0  # I-SPY1 is acquired sagittally (slices along left-right after reordering), the others axially


def panel(ax, img, e, masks, coh, dice):
    a = slice_axis(coh)
    area = e.sum(axis=tuple(i for i in range(3) if i != a))
    k = int(np.argmax(area))
    take = lambda v: np.take(v, k, axis=a)
    im, es = take(img).astype(float), take(e)
    ys, xs = np.nonzero(es | take(masks["medsam2"]))
    pad = 15
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.shape[1])
    crop = lambda v: v[y0:y1, x0:x1]
    lo, hi = np.percentile(im, [1, 99.5])
    ax.imshow(crop(im), cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.contour(crop(es), [0.5], colors=COL["expert"], linewidths=0.9)
    ax.contour(crop(take(masks["medsam2"])), [0.5], colors=COL["medsam2"], linewidths=0.9)
    if "official" in masks:
        ax.contour(crop(take(masks["official"])), [0.5], colors=COL["official"], linewidths=0.8, linestyles="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{NAME[coh]}, Dice {dice:.2f}", fontsize=7)


def main():
    d = pd.read_csv(os.path.join(q.OUT, "mamamia_agreement.csv"))
    ok = d[d.status == "ok"]
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.25, 0.32, 1, 1, 1], hspace=0.35, wspace=0.05)
    axb = fig.add_subplot(gs[0, :])
    groups = [("Duke", "medsam2"), ("ISPY1", "medsam2"), ("ISPY1", "official"), ("ISPY2", "medsam2"), ("ISPY2", "official")]
    data = [ok[(ok.cohort == c) & (ok["mask"] == m)].dice.values for c, m in groups]
    bp = axb.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": "black", "linewidth": 0.8})
    for p, (c, m) in zip(bp["boxes"], groups):
        p.set_facecolor(COL[m]); p.set_alpha(0.55); p.set_linewidth(0.6)
    axb.set_xticks(range(1, len(groups) + 1))
    axb.set_xticklabels([f"{NAME[c]}\n{'MedSAM2' if m == 'medsam2' else 'official'} (n = {len(v)})" for (c, m), v in zip(groups, data)])
    axb.set_ylabel("Dice with expert mask")
    axb.set_ylim(0, 1)
    axb.text(-0.11, 1.04, "(a)", transform=axb.transAxes, fontsize=9, fontweight="bold")
    for row, coh in enumerate(["Duke", "ISPY1", "ISPY2"], start=2):
        sub = ok[(ok.cohort == coh) & (ok["mask"] == "medsam2")].sort_values("dice").reset_index(drop=True)
        for col, qq in enumerate([0.25, 0.5, 0.75]):
            r = sub.iloc[int(round(qq * (len(sub) - 1)))]
            img, e, masks = load_case(r.case, coh, r.pid)
            ax = fig.add_subplot(gs[row, col])
            panel(ax, img, e, masks, coh, r.dice)
            if row == 2 and col == 0:
                ax.text(-0.25, 1.15, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    handles = [plt.Line2D([], [], color=COL["expert"], lw=1, label="Expert (MAMA-MIA)"),
               plt.Line2D([], [], color=COL["medsam2"], lw=1, label="MedSAM2"),
               plt.Line2D([], [], color=COL["official"], lw=1, ls="--", label="Official I-SPY mask")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figS_expert_agreement.{ext}"), dpi=300, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
