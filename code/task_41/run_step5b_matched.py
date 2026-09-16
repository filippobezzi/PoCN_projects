#!/usr/bin/env python3
# ==============================================================================
# AI ASSISTANCE DISCLAIMER
# In accordance with academic guidelines, the author acknowledges the use of
# generative artificial intelligence assistance (DeepMind's Antigravity AI assistant)
# for writing assistance, code development, data formatting, and proofreading the
# manuscript. Some analytical derivations, mathematical proofs, numerical simulations,
# and physical interpretations still need rigorous verification by the author.
# ==============================================================================
"""
run_step5b_matched.py
---------------------
Step 5b: amplitude-matched perturbation ensemble.

The step-5 protocol (dn_i = -min(n_i,5)) leaves the whitened initial kick
    ||dx_i(0)||_w^2 = (dn_i/N)^2 / n_i
varying by a factor of 74 across the 86 targets, so the measured distance
matrix is dominated by which targets were hit hardest rather than by how the
community responded.  Here dn_i = -round(sqrt(n_i)) (clamped to [1, n_i]),
which equalises that quantity by construction (residual spread 2.7x, entirely
from integer rounding at n <= 5).

Everything else is identical to step 5: same checkpoint, R = 60, T = 50, same
per-replica seeds (common random numbers), same sham block.
"""
import os, sys, json, math, time, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
_data     = os.path.join(REPO_DIR, "data", "task_41") if os.path.isdir(
                os.path.join(REPO_DIR, "data", "task_41")) else "data/step3"
CKPT_DAT  = os.path.join(_data, "checkpoint_microstate_tstar.dat")
CKPT_JSON = os.path.join(_data, "checkpoint_microstate_tstar.json")
BIN       = os.environ.get("TNM_BIN",
                os.path.join(BASE_DIR, "tnm_sim") if os.path.exists(os.path.join(BASE_DIR, "tnm_sim"))
                else "./bin/tnm_sim")
OUT       = os.path.join(_data, "step5b")
SHAM, PERT = f"{OUT}/sham", f"{OUT}/perturbed"
T_STAR, WINDOW_T, REPLICAS, BASE_SEED = 3402, 50, 60, 5000
TARGET_GENS = T_STAR + WINDOW_T

os.makedirs(SHAM, exist_ok=True); os.makedirs(PERT, exist_ok=True)

def matched_delta(n):
    """dn_i = round(sqrt(n_i)), clamped to [1, n_i]."""
    return max(1, min(int(n), int(round(math.sqrt(n)))))

def run_cmd(c):
    return subprocess.run(c, shell=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0

def complete_output(path):
    """
    A run counts as done only if its last line is well formed AND reaches the
    final generation.  Size alone is not enough: a run killed mid-write leaves
    a large file whose tail is truncated, which would otherwise be silently
    zero-filled into the ensemble mean.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 5000:
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(-4096, os.SEEK_END)
            tail = fh.read().decode("utf-8", "ignore")
        last = [l for l in tail.split("\n") if l.strip()][-1].split()
        if int(last[0]) != TARGET_GENS:
            return False
        return all(":" in tok for tok in last[3:])
    except Exception:
        return False


def parse_occupancy(path, sa_list):
    idx = {sa: i for i, sa in enumerate(sa_list)}
    f = np.zeros((WINDOW_T + 1, len(sa_list)), dtype=np.float32)
    seen = np.zeros(WINDOW_T + 1, dtype=bool)
    if not os.path.exists(path):
        return f, seen
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p: continue
            t = int(p[0]) - T_STAR
            if t < 0 or t > WINDOW_T: continue
            npop = int(p[1])
            if npop == 0: continue
            seen[t] = True
            for it in p[3:]:
                if ':' not in it:          # truncated final line
                    seen[t] = False
                    break
                s, c = it.split(':'); s = int(s)
                if s in idx: f[t, idx[s]] = int(c) / float(npop)
    return f, seen

def main():
    t0 = time.time()
    ck = json.load(open(CKPT_JSON))
    spp = ck["extant_species"]
    sa  = [s["sa"] for s in spp]
    pop = [s["population"] for s in spp]
    S   = len(spp)
    dn  = [matched_delta(p) for p in pop]
    kick = [d * d / p for d, p in zip(dn, pop)]
    print(f"S={S}  N={ck['Npop']}  amplitude-matched dn in [{min(dn)},{max(dn)}]")
    print(f"whitened kick dn^2/n: {min(kick):.3f}-{max(kick):.3f} (spread {max(kick)/min(kick):.2f}x)")
    print(f"full extinctions forced by integer floor: {sum(d >= p for d, p in zip(dn, pop))}/{S}")

    tasks = []
    for r in range(1, REPLICAS + 1):
        tag = f"sham_r{r}"
        tasks.append((f"{SHAM}/{tag}_occupancy.dat",
                      f"{BIN} --load-checkpoint {CKPT_DAT} --sim-seed {BASE_SEED+r} "
                      f"--gens {TARGET_GENS} --record-initial --out-dir {SHAM} --tag {tag}"))
    for i, s in enumerate(spp):
        d = dn[i]
        parg = (f"--perturb-sa {s['sa']} --perturb-extinct" if d >= s["population"]
                else f"--perturb-sa {s['sa']} --perturb-delta -{d}")
        for r in range(1, REPLICAS + 1):
            tag = f"pert_sp{i}_sa{s['sa']}_r{r}"
            tasks.append((f"{PERT}/{tag}_occupancy.dat",
                          f"{BIN} --load-checkpoint {CKPT_DAT} --sim-seed {BASE_SEED+r} {parg} "
                          f"--gens {TARGET_GENS} --record-initial --out-dir {PERT} --tag {tag}"))

    pending = [t for t in tasks if not complete_output(t[0])]
    print(f"total {len(tasks)} runs, {len(pending)} pending", flush=True)
    nw = max(1, (os.cpu_count() or 4))
    done = 0
    if pending:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(run_cmd, c): c for _, c in pending}
            for _ in as_completed(futs):
                done += 1
                if done % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(pending)}  {done/el:.1f} runs/s  "
                          f"eta {(len(pending)-done)/(done/el):.0f}s", flush=True)
    print(f"simulation done in {time.time()-t0:.0f}s; aggregating", flush=True)

    sham = np.zeros((REPLICAS, WINDOW_T + 1, S), dtype=np.float32)
    bad = 0
    for r in range(1, REPLICAS + 1):
        f, seen = parse_occupancy(f"{SHAM}/sham_r{r}_occupancy.dat", sa)
        bad += int((~seen).sum()); sham[r-1] = f
    sham_mean = sham.mean(0)

    pert_mean = np.zeros((S, WINDOW_T + 1, S), dtype=np.float32)
    delta_x   = np.zeros((S, WINDOW_T + 1, S), dtype=np.float32)
    for i, s in enumerate(spp):
        acc = np.zeros((REPLICAS, WINDOW_T + 1, S), dtype=np.float32)
        for r in range(1, REPLICAS + 1):
            f, seen = parse_occupancy(f"{PERT}/pert_sp{i}_sa{s['sa']}_r{r}_occupancy.dat", sa)
            bad += int((~seen).sum()); acc[r-1] = f
        pert_mean[i] = acc.mean(0)
        delta_x[i]   = pert_mean[i] - sham_mean
    if bad:
        raise SystemExit(f"ABORT: {bad} (replica, generation) records missing or "
                         f"malformed; rerun to regenerate the affected files")

    np.savez_compressed(f"{OUT}/perturbation_responses_matched_R60.npz",
        extant_sa=np.array(sa, np.int32), extant_pop=np.array(pop, np.int32),
        extant_ranks=np.array([s["index"] for s in spp], np.int32),
        delta_n=np.array(dn, np.int32), whitened_kick=np.array(kick, np.float64),
        sham_trajectories=sham, sham_mean=sham_mean, pert_mean=pert_mean,
        delta_x=delta_x, T=WINDOW_T, R=REPLICAS, T_STAR=T_STAR)
    json.dump({"status": "COMPLETED", "protocol": "amplitude-matched dn=-round(sqrt(n))",
               "num_species": S, "replicas": REPLICAS, "total_simulations": len(tasks),
               "missing_records_zero_filled": bad,
               "kick_min": min(kick), "kick_max": max(kick),
               "kick_spread": max(kick)/min(kick),
               "forced_extinctions": int(sum(d >= p for d, p in zip(dn, pop))),
               "runtime_s": time.time()-t0},
              open(f"{OUT}/step5b_summary.json", "w"), indent=2)
    print(f"WROTE {OUT}/perturbation_responses_matched_R60.npz  ({time.time()-t0:.0f}s total)", flush=True)

if __name__ == "__main__":
    main()
