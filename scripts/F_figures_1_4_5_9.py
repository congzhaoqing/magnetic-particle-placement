#!/usr/bin/env python3
"""Redraw Figures 1, 4, 5 and 9 of the placement manuscript from the deposited tables.

Inputs (all in tables/ of this repository):
    Table1_parameter_provenance_merged.csv
    F_membrane_arm_force_ceiling.csv
    F_source_contradictions.csv
    F_membrane_arm_threshold_crossing.csv
    F_membrane_arm_tension_profile.csv
    F_two_path_matched_moment.csv
    F_genelevel_DE_counts.csv
    F_effect_sizes_per_dataset.csv
Outputs: Figure1.png, Figure4.png, Figure5.png, Figure9.png (300 dpi).
"""
import re, sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else "tables")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

CAP, ROD, GRY, RED, GRN, ORA, PUR = "#1F4E79", "#9E2A1F", "#6E7781", "#B2182B", "#1B7837", "#D95F02", "#7570B3"

def style(sizes=(10, 9, 8)):
    base, med, small = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": med, "axes.titlesize": base,
        "axes.labelsize": med, "xtick.labelsize": small, "ytick.labelsize": small,
        "legend.fontsize": small, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "savefig.dpi": 300, "figure.dpi": 150})

def panel_letter(ax, L):
    ax.text(-0.17, 1.06, L, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

def measure(path):
    from PIL import Image
    from scipy import ndimage
    im = Image.open(path); w, _ = im.size; dpi = im.info.get("dpi", (300,))[0]
    lab, _ = ndimage.label(np.array(im.convert("L")) < 128)
    hs = [s[0].stop - s[0].start for s in ndimage.find_objects(lab)
          if 5 <= s[0].stop - s[0].start <= 45 and 2 <= s[1].stop - s[1].start <= 40]
    return w / dpi, float(np.median(hs)) / 2.6 * 6.9 / (w / dpi)

# ---------------------------------------------------------------- Figure 1
def figure1():
    style()
    tb = pd.read_csv(D / "Table1_parameter_provenance_merged.csv")
    fc = pd.read_csv(D / "F_membrane_arm_force_ceiling.csv")
    sc = pd.read_csv(D / "F_source_contradictions.csv")

    def famcls(x):
        x = str(x).lower()
        for k, v in [("measured", "measured"), ("inferred", "inferred"), ("computed", "computed here"),
                     ("not measured", "never measured"), ("assumed", "assumed"),
                     ("literature", "literature"), ("exact", "exact")]:
            if x.startswith("not measured"): return "never measured"
            if x.startswith(k): return v
        return "unclassified"
    order = ["measured", "inferred", "computed here", "unclassified", "assumed", "never measured", "literature", "exact"]
    cl = tb.measured_or_inferred.map(famcls).value_counts().reindex(order).fillna(0).astype(int)

    LBL = {"force on one 10 nm magnetite core, bench NdFeB gradient": ("one 10 nm core", GRY),
           "Reffay body text: force per iron-loaded endosome": ("body text: per endosome", ROD),
           "Reffay body text: force per cell": ("body text: per cell", ROD),
           "Reffay Table 3: force per cell, moderate gradients (LOWER)": ("table 3: moderate (low)", ROD),
           "Reffay Table 3: force per cell, moderate gradients (UPPER)": ("table 3: moderate (high)", ROD),
           "Reffay Table 3: force per cell, strong gradients": ("table 3: strong", ROD),
           "gradient force on a hemispherical 25 nm-cube cap (textured)": ("textured 25 nm cap*", CAP),
           "gradient force on a hemispherical 10 nm-core cap (textured)": ("textured 10 nm cap*", CAP),
           "ion-channel activation force (reference band, lower)": ("channel activation (lower)", GRY),
           "ion-channel activation force (reference band, upper)": ("channel activation (upper)", GRY)}
    fcv = fc.dropna(subset=["value_pN"]).copy()
    fcv["lab"] = fcv.quantity.map(lambda q: LBL[q][0]); fcv["col"] = fcv.quantity.map(lambda q: LBL[q][1])
    fcv = fcv.sort_values("value_pN")
    i_lo = list(fcv.lab).index("body text: per cell"); i_hi = list(fcv.lab).index("table 3: moderate (low)")

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.60, wspace=0.42)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])
    y = np.arange(len(fcv))
    a.hlines(y, 2e-6, fcv.value_pN, color="0.88", lw=1.0, zorder=1)
    a.scatter(fcv.value_pN, y, c=fcv.col, s=34, zorder=3)
    a.set_yticks(y); a.set_yticklabels(fcv.lab, fontsize=7.6)
    a.set_xscale("log"); a.set_xlim(2e-6, 2e4); a.set_ylim(-0.7, len(fcv) - 0.3)
    a.tick_params(axis="x", labelsize=7.6)
    a.annotate("", xy=(10, i_hi), xytext=(1, i_lo), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
    a.text(45, (i_lo + i_hi) / 2, "100$\\times$,\none review", color=RED, fontsize=7.4, ha="left", va="center")
    a.set_xlabel("force, in the unit each source states (pN)", fontsize=8.4)
    a.set_title("Six decades, and one review\ncontradicts itself inside them", fontsize=8.8, loc="left")

    colsb = [GRN, ORA, CAP, "0.75", PUR, RED, GRY, "0.45"]
    yb = np.arange(len(cl))[::-1]
    b.barh(yb, np.maximum(cl.values, 0.9), color=colsb, height=0.6)
    for yy, v in zip(yb, cl.values): b.text(max(v, 0.9) + 2.5, yy, str(v), va="center", fontsize=7.8)
    b.set_yticks(yb); b.set_yticklabels(cl.index, fontsize=7.8); b.tick_params(axis="x", labelsize=7.6)
    b.set_xlim(0, 82); b.set_xlabel("parameters (%d total)" % int(cl.sum()), fontsize=8.4)
    b.set_title("Two inputs have never\nbeen measured", fontsize=8.8, loc="left")

    SYM = {"phi_0": "$\\varphi_0$", "~1e-2": "~10$^{-2}$", "Table 3": "table 3",
           "10-100 pN": "10\u2013100 pN", " -- ": "; ", "(their ref 113)": "", "(their ref 143)": ""}
    c.axis("off"); c.set_title("Five inputs the sources leave open or contradict", fontsize=8.8, loc="left")
    col_x = [0.0, 0.52]; place = [(0, 0.95), (0, 0.64), (0, 0.40), (1, 0.95), (1, 0.58)]
    for (_, r), (cx, cy) in zip(sc.iterrows(), place):
        st = str(r.statement)
        for k_, v_ in SYM.items(): st = st.replace(k_, v_)
        st = re.sub(r"\s+\;", ";", re.sub(r"\s+", " ", st)).strip()
        c.text(col_x[cx], cy, f"{r.id} \u00b7 {r.source}", transform=c.transAxes, fontsize=7.9,
               fontweight="bold", color=RED, va="top")
        c.text(col_x[cx], cy - 0.075, "\n".join(textwrap.wrap(st, 52)), transform=c.transAxes,
               fontsize=7.4, color="0.2", va="top", linespacing=1.26)
    for ax_, L in [(a, "a"), (b, "b"), (c, "c")]: panel_letter(ax_, L)
    fig.savefig(OUT / "Figure1.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
def figure4():
    style()
    tc = pd.read_csv(D / "F_membrane_arm_threshold_crossing.csv").iloc[:10]
    tp = pd.read_csv(D / "F_membrane_arm_tension_profile.csv")
    cm = pd.read_csv(D / "F_membrane_arm_cap_moment_vs_coverage.csv")
    thr = [5.10, 4.50, 3.83, 1.40, 1.00]
    pick = lambda x, bb: float(tc[np.isclose(tc.threshold_mNm, x) & np.isclose(tc.B_T, bb)].coverage_phi_required.iloc[0])
    dk = [pick(x, 0.26) for x in thr]; lt = [pick(x, 0.10) for x in thr]
    s = cm[np.isclose(cm.coverage_phi, 0.5, atol=0.004)]
    g = s.groupby(["particle", "B_label", "texture"]).tension_resultant_mNm.max()
    cv = [float(g[("Zn-ferrite 25 nm cube (Li)", "0.26 T (Li)", "T3_blocked_textured")]),
          float(g[("magnetite 10 nm core (Golovin)", "0.26 T (Li)", "T3_blocked_textured")]),
          float(g[("Zn-ferrite 25 nm cube (Li)", "0.26 T (Li)", "T2_blocked_random")]),
          float(g[("Zn-ferrite 25 nm cube (Li)", "0.26 T (Li)", "T1_superparamagnetic")])]

    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.15),
                            gridspec_kw=dict(wspace=0.42, left=0.075, right=0.985, top=0.80, bottom=0.26))
    A, Bx, C = axs
    x = np.arange(5); w = 0.38
    A.bar(x - w / 2, [max(v, 0) for v in dk], w, color=CAP, label="$B$ = 0.26 T (Li)")
    A.bar(x + w / 2, lt, w, color="#6BAED6", label="$B$ = 0.10 T")
    A.axhline(0.5, color=GRY, ls="--", lw=1.1, zorder=0)
    A.text(4.58, 0.80, "dashed line:\nhemispherical cap,\n$\\varphi$ = 0.5", color=GRY, fontsize=6.9,
           ha="right", va="top", linespacing=1.3)
    A.annotate("exceeded\nat $\\varphi\\to$ 0", xy=(4 - w / 2, 0.015), xytext=(3.15, 0.23), fontsize=7.0,
               color=CAP, ha="center", arrowprops=dict(arrowstyle="->", color=CAP, lw=0.9))
    A.set_xticks(x); A.set_xticklabels([f"{v:g}" for v in thr], fontsize=7.8)
    A.set_xlabel("tension threshold (mN m$^{-1}$)", fontsize=8.5)
    A.set_ylabel("coverage $\\varphi$ required", fontsize=8.5); A.set_ylim(0, 1.14); A.set_xlim(-0.6, 4.6)
    A.legend(frameon=False, fontsize=7.4, loc="upper right", handlelength=1.1, borderpad=0.05,
             labelspacing=0.2, bbox_to_anchor=(1.02, 0.99))
    A.set_title("Only the decortexed bound is\nreachable with a Janus cap", fontsize=8.7, loc="left")

    for cc, lb, co in [("Lame_cortexfree", "cortex-free (Lam\u00e9)", GRY),
                       ("screened_lambda_10.0um", "$\\lambda$ = 10 \u00b5m", "#9ECAE1"),
                       ("screened_lambda_1.0um", "$\\lambda$ = 1 \u00b5m", "#4292C6"),
                       ("screened_lambda_0.1um", "$\\lambda$ = 0.1 \u00b5m", CAP)]:
        Bx.plot(tp.r_over_a, tp[cc], color=co, lw=1.7, label=lb)
    Bx.set_yscale("log"); Bx.set_ylim(1e-4, 1.8); Bx.set_xlim(1, tp.r_over_a.max())
    Bx.set_xlabel("distance from cap rim, $r/a$", fontsize=8.5)
    Bx.set_ylabel("$\\delta T(r)\\,/\\,\\delta T(a)$", fontsize=8.5)
    Bx.legend(frameon=False, fontsize=7.4, handlelength=1.2, borderpad=0.05, labelspacing=0.2, loc="upper right")
    Bx.set_title("The screening length is unmeasured\nand it sets the reach", fontsize=8.7, loc="left")

    C.bar(np.arange(4), [max(v, 1.2e-5) for v in cv], 0.6, color=[CAP, "#6BAED6", ORA, GRY])
    for i, v in enumerate(cv): C.text(i, max(v, 1.2e-5) * 1.45, ("0" if v == 0 else f"{v:.3g}"), ha="center", fontsize=7.4)
    C.set_yscale("log"); C.set_ylim(6e-6, 120)
    C.axhline(5.1, color=RED, ls="-.", lw=1.0); C.text(3.5, 6.4, "Cox 5.1", color=RED, fontsize=7.2, ha="right", va="bottom")
    C.axhline(1.4, color=ORA, ls="-.", lw=1.0); C.text(3.5, 0.55, "L&G 1.4", color=ORA, fontsize=7.2, ha="right", va="top")
    C.set_xticks(np.arange(4))
    C.set_xticklabels(["textured 25 nm", "textured 10 nm", "random easy axes", "superparamagnetic"],
                      fontsize=7.4, rotation=28, ha="right", rotation_mode="anchor")
    C.set_ylabel("$\\tau_s$ at $\\varphi$ = 0.5 (mN m$^{-1}$)", fontsize=8.5)
    C.set_title("A superparamagnetic cap\ndelivers no torque at all", fontsize=8.7, loc="left")
    for ax_, L in zip(axs, "abc"): panel_letter(ax_, L)
    fig.savefig(OUT / "Figure4.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5
def figure5():
    style()
    tm = pd.read_csv(D / "F_two_path_matched_moment.csv")
    LAB = ["$N$ = 7 rod in lysosome (Li 2026)", "same rod, cytosol caliber", "same rod, 453 nm lysosome",
           "cap = one rod", "cap = one rod, adherent", "cap = 100 rods", "cap = 100 rods, adherent",
           "cap ceiling, 25 nm textured", "cap ceiling, 10 nm textured", "cap ceiling, 25 nm random",
           "cap ceiling, 25 nm\nsuperparamagnetic", "intracellular load matched to cap",
           "Jiang 2023 swimmer, unanchored"]
    col = [ROD if p == "intracellular" else (CAP if p == "membrane-anchored" else GRY) for p in tm.placement]
    yv = np.arange(len(tm))[::-1]
    mg = tm.stepout_margin_fc_over_f.values
    ut = tm.ceiling_utilisation.values * 100

    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.62, 0.95], wspace=0.40)
    A = fig.add_subplot(gs[0, 0]); Bx = fig.add_subplot(gs[0, 1], sharey=A); C = fig.add_subplot(gs[0, 2])
    A.barh(yv, np.maximum(mg, 3e-5), 0.66, color=col)
    A.set_xscale("log"); A.set_xlim(3e-5, 1e6); A.axvline(1, color=RED, ls="--", lw=1.2)
    A.text(1.7, 12.85, "synchronous above 1", color=RED, fontsize=7.2, ha="left", va="bottom")
    A.set_yticks(yv); A.set_yticklabels(LAB, fontsize=7.5); A.set_ylim(-0.7, 13.8)
    A.tick_params(axis="x", labelsize=7.8); A.set_xlabel("step-out margin $f_c/f$", fontsize=8.8)
    A.set_title("Thirteen configurations at one\ndrive, ranked by margin", fontsize=8.9, loc="left")

    Bx.barh(yv, np.nan_to_num(np.maximum(ut, 1e-3), nan=1e-3), 0.66, color=col)
    Bx.set_xscale("log"); Bx.set_xlim(1e-3, 3e3); Bx.axvline(100, color=RED, ls="--", lw=1.2)
    Bx.text(1.3e-3, 2.0, "adherent: $T/mB$ = 1", color=RED, fontsize=6.9, ha="left", va="center")
    plt.setp(Bx.get_yticklabels(), visible=False); Bx.tick_params(axis="y", length=0)
    Bx.tick_params(axis="x", labelsize=7.8); Bx.set_xlabel("ceiling used (%)", fontsize=8.8)
    Bx.set_title("$T/mB = f/f_c$: share\nof the ceiling spent", fontsize=8.9, loc="left")

    n_match = 93281.11
    C.bar([0, 1], [100, n_match], 0.5, color=[GRY, ROD])
    for i, val in zip([0, 1], [100, n_match]): C.text(i, val * 1.7, f"{val:,.0f}", ha="center", fontsize=8.0)
    C.set_yscale("log"); C.set_ylim(30, 2e7); C.set_xlim(-0.62, 1.62)
    C.set_xticks([0, 1]); C.set_xticklabels(["in a cell\n(measured)", "to match\nthe cap"], fontsize=7.6)
    C.tick_params(axis="y", labelsize=7.8); C.set_ylabel("lysosomes per cell", fontsize=8.8)
    C.set_title("Matching the cap intra-\ncellularly is 933$\\times$ off", fontsize=8.9, loc="left")
    for ax_, L in zip((A, Bx, C), "abc"): panel_letter(ax_, L)
    fig.savefig(OUT / "Figure5.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 9
def figure9():
    style()
    dec = pd.read_csv(D / "F_genelevel_DE_counts.csv")
    es = pd.read_csv(D / "F_effect_sizes_per_dataset.csv")
    tot = lambda acc, term: int(dec[(dec.accession == acc) & (dec.term == term)][["n_up", "n_down"]].sum(axis=1).iloc[0])
    v = [tot("GSE324656", "particle[T.MAN]"), tot("GSE324656", "field[T.F]"),
         tot("GSE324656", "particle[T.MAN]:field[T.F]")]
    NICE = {"AP1_IEG": "AP-1 immediate early", "HEAT_SHOCK": "heat shock", "LYSOSOME_CLEAR": "lysosome clearance",
            "UPR_ER_STRESS": "UPR / ER stress", "NLRP3_INFLAMMASOME": "NLRP3 inflammasome",
            "NFKB_INFLAMMATORY": "NF-$\\kappa$B inflammatory", "PIEZO_MECHANOSENSOR": "PIEZO mechanosensors",
            "YAP_TAZ_TARGETS": "YAP/TAZ targets", "ANTIGEN_CROSSPRESENTATION": "antigen cross-presentation",
            "GSE310169_PAPER_DECLARED": "GSE310169 declared set"}
    ix = es[(es.accession == "GSE324656") & es.term.str.contains(":")].sort_values("estimate").reset_index(drop=True)
    cols9 = [RED if s else GRY for s in ix.padj_BH_within_dataset < 0.05]
    ypos = np.arange(len(ix))

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.1),
                            gridspec_kw=dict(hspace=0.62, wspace=0.95, width_ratios=[0.92, 1.12]))
    A, Bx = axs[0]; C, Dd = axs[1]
    A.bar([0, 1, 2], v, 0.62, color=[GRY, GRY, RED])
    for i, val in enumerate(v): A.text(i, val + 32, f"{val:,}", ha="center", fontsize=8.2)
    A.set_xticks([0, 1, 2]); A.set_xticklabels(["particle", "field", "particle\n$\\times$ field"], fontsize=8.0)
    A.set_ylim(0, 1450); A.set_ylabel("DE genes\n($q$ < 0.05, |log$_2$FC| > 1)", fontsize=8.4)
    A.text(0.03, 0.96, "%.1f$\\times$ the particle term,\n%.1f$\\times$ the field term" % (v[2] / v[0], v[2] / v[1]),
           transform=A.transAxes, color=RED, fontsize=7.6, va="top")
    A.set_title("GSE324656: the response\nlives in the interaction", fontsize=9.0, loc="left")

    Bx.axvline(0, color="0.65", lw=1.0)
    Bx.hlines(ypos, ix.ci_low, ix.ci_high, color=cols9, lw=1.6)
    Bx.scatter(ix.estimate, ypos, c=cols9, s=30, zorder=3)
    Bx.set_yticks(ypos); Bx.set_yticklabels([NICE[g] for g in ix.gene_set], fontsize=7.8)
    Bx.set_ylim(-0.6, len(ix) - 0.4); Bx.tick_params(axis="x", labelsize=7.8)
    Bx.set_xlabel("ssGSEA interaction coefficient (95 % CI)", fontsize=8.4)
    Bx.set_title("Every positive set-level effect\nsits in the interaction", fontsize=9.0, loc="left")

    n_ax = tot("GSE197808", "field[T.F]")
    C.bar([0, 1], [n_ax, 1278], 0.5, color=[GRY, CAP])
    for i, val in zip([0, 1], [n_ax, 1278]): C.text(i, val + 35, f"{val:,}", ha="center", fontsize=8.2)
    C.set_xticks([0, 1]); C.set_xticklabels(["soma", "axon"], fontsize=8.2); C.set_xlim(-0.75, 1.75)
    C.set_ylim(0, 1600); C.set_ylabel("DE genes, field vs null gradient", fontsize=8.4)
    C.set_title("GSE197808: one force,\ntwo compartments", fontsize=9.0, loc="left")

    st = [5, 15]
    ng = [tot("GSE310169", "armf[T.strain_5pct]"), tot("GSE310169", "armf[T.strain_15pct]")]
    g3 = es[(es.accession == "GSE310169") & (es.gene_set == "GSE310169_PAPER_DECLARED") &
            es.term.str.startswith("C(armf")]
    est = list(g3.estimate); lo = list(g3.ci_low); hi = list(g3.ci_high)
    Dd.plot(st, ng, "o-", color=CAP, lw=1.8, ms=6)
    Dd.set_xlabel("magnetically delivered strain (%)", fontsize=8.4)
    Dd.set_ylabel("DE genes vs 0 %", fontsize=8.4, color=CAP)
    Dd.tick_params(axis="y", labelcolor=CAP, labelsize=7.8); Dd.tick_params(axis="x", labelsize=7.8)
    Dd.set_ylim(0, 520); Dd.set_xlim(2, 18); Dd.set_xticks([5, 10, 15])
    D2 = Dd.twinx()
    D2.errorbar(st, est, yerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o-", color=ORA, lw=1.8, ms=6, capsize=3)
    D2.set_ylabel("depositors' 11-gene set\n(ssGSEA shift)", fontsize=8.4, color=ORA)
    D2.tick_params(axis="y", labelcolor=ORA, labelsize=7.8); D2.set_ylim(0, 0.31); D2.spines["top"].set_visible(False)
    Dd.text(0.03, 0.98, "3$\\times$ strain buys %.2f$\\times$ the genes,\n%.2f$\\times$ the score"
            % (ng[1] / ng[0], est[1] / est[0]), transform=Dd.transAxes, fontsize=7.4, va="top")
    Dd.set_title("GSE310169: the amplitude\nresponse saturates", fontsize=9.0, loc="left", y=1.0)
    for ax_, L in zip((A, Bx, C, Dd), "abcd"): panel_letter(ax_, L)
    fig.savefig(OUT / "Figure9.png", dpi=300, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    for fn in (figure1, figure4, figure5, figure9):
        fn()
    for n in (1, 4, 5, 9):
        w, pt = measure(OUT / f"Figure{n}.png")
        print(f"Figure{n}.png  {w:.2f} in wide, {pt:.1f} pt effective type at 6.9 in column width")
