"""Evaluation: how well, how fast and how safely does DriveMemory learn a driver?

For every persona we simulate N weeks twice, on identical trips:

* **baseline**  – the car never presets anything (memory is not used)
* **assisted**  – before each trip the agents preset the cabin from memory

and measure after every week:

* **recall**    – share of the persona's hidden ground-truth preferences learned correctly
* **precision** – share of the system's *active* beliefs (on ground-truth keys) that are correct
* **manual adjustments per trip** – how often the driver still had to fix a setting
* **unplanned takeovers** – surprise handovers from autonomy (planned ones don't count)
* **drift adaptation** – weeks until a changed preference is picked up

Run:  python eval/run_eval.py --weeks 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from drivememory.agents import Orchestrator
from drivememory.memory import MemoryManager
from drivememory.signals.models import TripContext
from drivememory.signals.simulator import PERSONAS, TripSimulator

TOL = {"climate.cabin_temp_c": 0.5, "media.volume": 2.0, "autonomy.following_gap_s": 0.25}
OUT = Path(__file__).parent / "results"


def matches(key: str, learned, truth) -> bool:
    if key in TOL:
        return isinstance(learned, (int, float)) and abs(float(learned) - float(truth)) <= TOL[key]
    return learned == truth


def score(memory: MemoryManager, driver_id: str, truth: list[tuple[str, str, object]]) -> tuple[float, float]:
    active = {(p.key, p.context): p.value for p in memory.profile(driver_id)}
    hits = sum(1 for k, c, v in truth if (k, c) in active and matches(k, active[(k, c)], v))
    truth_keys = {(k, c): v for k, c, v in truth}
    judged = [(kc, v) for kc, v in active.items() if kc in truth_keys]
    correct = sum(1 for kc, v in judged if matches(kc[0], v, truth_keys[kc]))
    recall = hits / len(truth)
    precision = correct / len(judged) if judged else 1.0
    return recall, precision


def run_persona(driver_id: str, weeks: int, assisted: bool, seed: int) -> list[dict]:
    persona = PERSONAS[driver_id]
    sim = TripSimulator(persona, seed=seed)
    memory = MemoryManager()
    orch = Orchestrator(memory, use_llm=False)
    rows = []
    for week, plans in sim.iter_weeks(weeks):
        adj = takeovers = 0
        for plan in plans:
            presets = orch.on_trip_start(TripContext.from_plan(plan), compose=False).presets if assisted else {}
            trip = sim.simulate(plan, presets)
            memory.ingest_trip(trip)
            adj += trip.manual_adjustments
            takeovers += trip.unplanned_takeovers
        recall, precision = score(memory, driver_id, persona.at_week(week).ground_truth())
        rows.append({"week": week + 1, "trips": len(plans), "recall": round(recall, 3),
                     "precision": round(precision, 3), "adjustments_per_trip": round(adj / len(plans), 3),
                     "unplanned_takeovers": takeovers})
    return rows


def drift_adaptation(rows: list[dict], drift_week: int) -> int | None:
    """Weeks after the drift until recall is back at its pre-drift level."""
    before = rows[drift_week - 1]["recall"]
    for r in rows[drift_week:]:
        if r["recall"] >= before - 1e-9:
            return r["week"] - drift_week
    return None


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    weeks = [r["week"] for r in summary["mean"]["assisted"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    ax = axes[0]
    ax.plot(weeks, [r["recall"] * 100 for r in summary["mean"]["assisted"]], color=blue, lw=2,
            marker="o", ms=5, label="Recall")
    ax.plot(weeks, [r["precision"] * 100 for r in summary["mean"]["assisted"]], color=aqua, lw=2,
            marker="o", ms=5, label="Precision")
    ax.set_ylim(0, 105)
    ax.set_title("What the memory gets right", loc="left", color=ink, fontsize=12, fontweight="bold")
    ax.set_ylabel("% of preferences", color=ink2)

    ax = axes[1]
    base = [r["adjustments_per_trip"] for r in summary["mean"]["baseline"]]
    assist = [r["adjustments_per_trip"] for r in summary["mean"]["assisted"]]
    ax.plot(weeks, base, color=orange, lw=2, marker="o", ms=5, label="No memory (baseline)")
    ax.plot(weeks, assist, color=blue, lw=2, marker="o", ms=5, label="With DriveMemory")
    ax.set_ylim(0, max(base) * 1.25)
    ax.set_title("Manual adjustments per trip", loc="left", color=ink, fontsize=12, fontweight="bold")
    ax.set_ylabel("adjustments / trip", color=ink2)

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel("week of driving", color=ink2)
        ax.set_xticks(weeks)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2)
        ax.legend(frameon=False, labelcolor=ink, loc="lower right" if ax is axes[0] else "upper right")
    fig.suptitle(f"DriveMemory evaluation: {len(summary['personas'])} simulated drivers, "
                 f"{len(weeks)} weeks", x=0.01, ha="left", color=ink2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3, help="independent runs per persona")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    per_persona: dict = {}
    for pid, persona in PERSONAS.items():
        runs = {"baseline": [], "assisted": []}
        for seed in range(args.seeds):
            for mode in runs:
                runs[mode].append(run_persona(pid, args.weeks, mode == "assisted", seed=seed + 1))
        avg = {mode: [{k: round(mean(r[i][k] for r in rs), 3) for k in rs[0][i]} for i in range(args.weeks)]
               for mode, rs in runs.items()}
        per_persona[pid] = {"name": persona.name, **avg}
        if persona.drifts and persona.drifts[0].week < args.weeks:
            per_persona[pid]["drift_adaptation_weeks"] = drift_adaptation(avg["assisted"], persona.drifts[0].week)

    mean_curves = {mode: [{k: round(mean(p[mode][i][k] for p in per_persona.values()), 3)
                           for k in per_persona["anna"][mode][i]} for i in range(args.weeks)]
                   for mode in ("baseline", "assisted")}
    last_b, last_a = mean_curves["baseline"][-1], mean_curves["assisted"][-1]
    takeovers_b = sum(r["unplanned_takeovers"] for p in per_persona.values() for r in p["baseline"])
    takeovers_a = sum(r["unplanned_takeovers"] for p in per_persona.values() for r in p["assisted"])
    headline = {
        "final_recall": last_a["recall"],
        "final_precision": last_a["precision"],
        "adjustment_reduction": round(1 - last_a["adjustments_per_trip"] / last_b["adjustments_per_trip"], 3),
        "unplanned_takeover_reduction": round(1 - takeovers_a / takeovers_b, 3) if takeovers_b else None,
        "weeks_to_80pct_recall": next((r["week"] for r in mean_curves["assisted"] if r["recall"] >= 0.8), None),
    }
    summary = {"weeks": args.weeks, "seeds": args.seeds, "personas": per_persona, "mean": mean_curves,
               "headline": headline}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Driver | Recall | Precision | Adjustments/trip (baseline → DriveMemory) | Drift adapted in |",
             "|---|---|---|---|---|"]
    for p in per_persona.values():
        b, a = p["baseline"][-1], p["assisted"][-1]
        drift = f"{p['drift_adaptation_weeks']} week(s)" if p.get("drift_adaptation_weeks") is not None else "No drift"
        lines.append(f"| {p['name']} | {a['recall']:.0%} | {a['precision']:.0%} | "
                     f"{b['adjustments_per_trip']:.2f} → {a['adjustments_per_trip']:.2f} | {drift} |")
    table = "\n".join(lines)
    (OUT / "results.md").write_text(table + "\n")

    print(table)
    print("\nHeadline:", json.dumps(headline, indent=2))
    if not args.no_plot:
        plot(summary, OUT / "learning_curves.png")
        print(f"\nChart written to {OUT / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
