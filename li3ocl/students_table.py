"""
Per-model numbers of the Li3OCl tables at 1000 K (main text: students; Supplemental Material: pooled and mechanism-resolved values).

    predicted    whole-cell predictions on the pooled reactive ensemble of the teacher
    measured     ln(Gamma_model / Gamma_teacher) for all hops and for the single-vacancy state (defect_states.py),
                 time in the extra-defect state, fraction of hops that fuse in transit (clean_channel.py)
    resolved     bracket on the clean single-vacancy channel with equilibrium leverage weights (bracket_vs_rate.py)
    python students_table.py
"""
import csv, glob, os, numpy as np

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "clean_channel.py")).read()
ns = {}; exec(compile(src[:src.index('print(f"{\'run\'')], "clean_channel", "exec"), ns)

def predicted(model):
    f = [g for g in glob.glob(os.path.join(here, "../data/li3ocl_pred_%s_1000K.csv" % model))]
    if not f: return {}
    r = np.genfromtxt(f[0], delimiter=",", names=True); w = r[r["R_A"] == -1][0]
    return dict(pooled_spread=w["sd_beta_dV"], pooled_upper=w["upper"], pooled_series=w["series"])

states = {r["name"]: r for r in csv.DictReader(open(os.path.join(here, "../data/li3ocl_defect_states.csv")))}
rate = lambda n, key: float(states[n][key])
t_single = rate("teacher_1000K_censored", "rate_single"); t_all = (float(states["teacher_1000K_censored"]["hops_single"]) + float(states["teacher_1000K_censored"]["hops_defect"])) / (float(states["teacher_1000K_censored"]["ns_single"]) + float(states["teacher_1000K_censored"]["ns_defect"]))
resolved = {r["student"]: r for r in csv.DictReader(open(os.path.join(here, "../data/li3ocl_bracket_vs_rate.csv")))}
rmse = {"S1": 6.1, "S2": 6.5, "S3": 7.5, "S4": 9.3, "S5": 13.7, "S6": 19.9, "S6b": 16.1, "S6c": 15.1}      # held-out force RMSE, meV/A (step 6 and step 8 reports)

rows = []
for m in ("S1", "S2", "S3", "S4", "S5", "S6", "S6b", "S6c"):
    n = f"{m}_1000K"; st = states.get(n)
    hop_file = os.path.join(here, "../colab/step7_results/hops_%s_1000K.npz" % m)
    if not os.path.exists(hop_file): hop_file = os.path.join(here, "../colab/step9_results/hops_%s_1000K.npz" % m)
    na, nc, _ = ns["analyse"](hop_file)
    row = dict(model=m, force_rmse=rmse[m], fused_in_transit_pct=100 * (1 - nc / na))
    if st:
        tot = (float(st["hops_single"]) + float(st["hops_defect"])) / (float(st["ns_single"]) + float(st["ns_defect"]))
        row.update(ln_all=np.log(tot / t_all), ln_single=np.log(float(st["rate_single"]) / t_single), defect_time_pct=100 * float(st["frac_defect_time"]))
    row.update(predicted(m))
    if m in resolved:
        r = resolved[m]; row.update(res_lower=float(r["lower"]), res_upper=float(r["upper"]), res_spread=float(r["sd_nu"]), clean_measured=float(r["measured"]), clean_se=float(r["se"]))
    rows.append(row); print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "model"}, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(os.path.join(here, "../data/li3ocl_students_1000K.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "force_rmse", "ln_all", "defect_time_pct", "fused_in_transit_pct", "ln_single", "pooled_spread", "pooled_upper", "pooled_series", "res_spread", "res_lower", "res_upper", "clean_measured", "clean_se"])
    w.writeheader(); [w.writerow(r) for r in rows]
