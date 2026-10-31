"""
Benchmark: lets brains learn the simulated level (sim.py) side by side and compares how fast
they get good. Every brain starts from scratch in its own temporary folder.

    python bench.py                                   # the current brain, 3 runs of 10 minutes
    python bench.py --minutes 15 old.py ../brain/brain2.py "../brain/brain2.py|--experiments 0"

A brain is a brain2.py file, optionally with "|" and extra arguments for it. Prints, per brain:
time to the first full-run win, full-run wins, the median of the last 20 full runs, and the same
at an equal number of decisions for all (the fewest any run got), since a slower brain gets
fewer decisions in the same time. Results go to tests/results/.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(sim, brain_log):
    runs = sim["runs"]
    pcts = [r[1] for r in runs]
    demos = re.findall(r"\((\d+) demo decisions\)", brain_log)
    return {
        "attempts": sim["attempts"],
        "decisions": sim["decisions"],
        "full_runs": len(runs),
        "wins": sim["wins"]["normal"],
        "first_win_s": next((r[0] for r in runs if r[1] >= 100), None),
        "best": max(pcts, default=0.0),
        "median_last20": statistics.median(pcts[-20:]) if pcts else 0.0,
        "experiments_better": len(re.findall(r"^Experiment:", brain_log, re.M)),
        "demo_decisions": int(demos[-1]) if demos else 0,
        "runs": runs,
    }


def at_budget(result, budget):
    """Best run and median of the last 20 full runs within the first `budget` decisions."""
    pcts = [r[1] for r in result["runs"] if r[2] <= budget]
    return max(pcts, default=0.0), (statistics.median(pcts[-20:]) if pcts else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("brains", nargs="*", default=[str(HERE.parent / "brain" / "brain2.py")])
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--runs", type=int, default=3, help="runs per brain (they differ by chance)")
    parser.add_argument("--threads", type=int, default=0, help="training threads per brain (0 = as the brain says)")
    parser.add_argument("--port", type=int, default=5600)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="gdai-bench-"))
    jobs = []
    port = args.port
    for b, spec in enumerate(args.brains):
        path, _, extra = spec.partition("|")
        for r in range(args.runs):
            folder = work / f"{b}-{Path(path).stem}-{r}"
            folder.mkdir()
            source = Path(path).read_text(encoding="utf-8")
            if args.threads:
                # Many brains at once fight over the CPU; fewer threads each keeps it fair.
                source = re.sub(r"^TRAIN_THREADS = \d+", f"TRAIN_THREADS = {args.threads}", source, flags=re.M)
            (folder / "brain2.py").write_text(source, encoding="utf-8")
            out = (folder / "brain.out").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", "brain2.py", "--port", str(port), "--no-review", *extra.split()],
                                    cwd=folder, stdout=out, stderr=subprocess.STDOUT)
            jobs.append({"brain": spec, "run": r, "port": port, "folder": folder, "proc": proc, "out": out})
            port += 1
    # Wait until every brain listens, then start the levels.
    deadline = time.time() + 120
    for j in jobs:
        while "Waiting for GD" not in (j["folder"] / "brain.out").read_text(encoding="utf-8", errors="replace"):
            if time.time() > deadline or j["proc"].poll() is not None:
                sys.exit(f"brain {j['folder'].name} didn't start:\n" + (j["folder"] / "brain.out").read_text())
            time.sleep(0.5)
        j["sim_out"] = (j["folder"] / "sim.out").open("w", encoding="utf-8")
        j["sim"] = subprocess.Popen([sys.executable, "-u", str(HERE / "sim.py"), str(j["port"]), str(args.minutes * 60)],
                                    stdout=j["sim_out"], stderr=subprocess.STDOUT)
    print(f"{len(jobs)} brains learning for {args.minutes:g} minutes in {work}", flush=True)
    for j in jobs:
        j["sim"].wait()
    results = []
    for j in jobs:
        j["proc"].kill()
        j["proc"].wait()
        j["out"].close()
        j["sim_out"].close()
        sim_text = (j["folder"] / "sim.out").read_text(encoding="utf-8", errors="replace")
        line = next((l for l in sim_text.splitlines() if l.startswith("SIM ")), None)
        if line is None:
            print(f"{j['folder'].name}: the level didn't finish:\n{sim_text[-2000:]}")
            continue
        brain_log = (j["folder"] / "brain.out").read_text(encoding="utf-8", errors="replace")
        results.append({"brain": j["brain"], "run": j["run"], **summarize(json.loads(line[4:]), brain_log)})
    if not results:
        sys.exit("no results")

    budget = min(r["decisions"] for r in results)
    for r in results:
        r["best_at_budget"], r["median_at_budget"] = at_budget(r, budget)
    keys = ["first_win_s", "wins", "median_last20", "best", "decisions", "median_at_budget", "best_at_budget",
            "experiments_better", "demo_decisions"]
    print("\n" + " | ".join(["brain", "run"] + keys))
    for r in results:
        print(" | ".join([r["brain"], str(r["run"])] + [str(r[k]) for k in keys]))
    print(f"\nAverages (\"same decisions\" = the first {budget:,} decisions of every run):")
    for spec in args.brains:
        mine = [r for r in results if r["brain"] == spec]
        if not mine:
            continue
        mean = lambda k: statistics.mean(r[k] for r in mine)
        wins = [r["first_win_s"] for r in mine if r["first_win_s"] is not None]
        print(f"  {spec}\n    first win after {statistics.mean(wins) if wins else float('nan'):.0f}s "
              f"({len(wins)}/{len(mine)} runs won), wins {mean('wins'):.0f}, median of last 20 full runs "
              f"{mean('median_last20'):.1f}%, best {mean('best'):.1f}%, decisions {mean('decisions'):,.0f}\n"
              f"    at the same decisions: median of last 20 {mean('median_at_budget'):.1f}%, best {mean('best_at_budget'):.1f}%")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"minutes": args.minutes, "budget": budget, "results": results}, indent=1),
                    encoding="utf-8")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
