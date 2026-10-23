"""Command-line interface.

    drivememory demo --driver anna --weeks 4      # end-to-end walkthrough
    drivememory chat --driver anna                # interactive chat after a simulation
    drivememory profile --driver anna             # print the learned profile
    drivememory personas                          # list simulated drivers
"""

from __future__ import annotations

import argparse
import sys

from .agents import Orchestrator
from .agents.describe import describe
from .memory import MemoryManager, MemoryStore
from .runtime import simulate_driver
from .signals.models import WEEKDAYS, TripContext
from .signals.simulator import PERSONAS, TripSimulator

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    BOLD = DIM = CYAN = GREEN = YELLOW = RESET = ""


def _system(db: str) -> tuple[MemoryManager, Orchestrator]:
    memory = MemoryManager(MemoryStore(db))
    return memory, Orchestrator(memory)


def _print_profile(memory: MemoryManager, driver: str, candidates: bool = False) -> None:
    now = memory.now(driver)
    for p in memory.profile(driver, include_candidates=candidates):
        day = p.context.split("-")[0]
        if p.key == "routine.destination" and day in WEEKDAYS and p.value in ("home", "work"):
            continue
        conf = p.effective_confidence(now)
        bar = "█" * round(conf * 10) + "░" * (10 - round(conf * 10))
        drift = f" {YELLOW}(changed {len(p.history)}x){RESET}" if p.history else ""
        print(f"  {bar} {conf:4.0%}  {describe(p)}{drift}")


def cmd_demo(args) -> None:
    memory, orch = _system(args.db)
    persona = PERSONAS[args.driver]
    print(f"{BOLD}DriveMemory demo: {persona.name}{RESET}\n")
    print(f"{CYAN}1. Learning: simulating {args.weeks} weeks of driving...{RESET}")
    stats = simulate_driver(memory, orch, args.driver, args.weeks, assisted=True, seed=args.seed)
    print(f"   {stats['trips']} trips ingested, {stats['active_preferences']} active preferences.\n")

    print(f"{CYAN}2. What the assistant has learned:{RESET}")
    _print_profile(memory, args.driver)

    sim = TripSimulator(persona, seed=args.seed + 1)
    plans = sim.plan_week(args.weeks)
    picks = [next((p for p in plans if p.rain_onset_min is not None and p.highway), plans[0]),
             next((p for p in plans if p.duration_min >= 75), None)]
    print(f"\n{CYAN}3. Next trips: the agents personalise the start (A2A fan-out):{RESET}")
    for plan in filter(None, picks):
        ctx = TripContext.from_plan(plan)
        res = orch.on_trip_start(ctx)
        print(f"   {DIM}{plan.departure:%a %H:%M}, {plan.outside_temp_c}°C, "
              f"rain expected: {ctx.rain_expected}, {plan.duration_min} min{RESET}")
        print(f"   {GREEN}Assistant:{RESET} {res.greeting}")
        print(f"   {DIM}presets: {res.presets}{RESET}\n")

    print(f"{CYAN}4. Conversation:{RESET}")
    for q in ["Why is it this temperature?", "What do you know about me?", "Actually, I prefer 20 degrees",
              "When did I last take a break on a long trip?", "Forget my music preferences"]:
        print(f"   {BOLD}You:{RESET} {q}")
        print(f"   {GREEN}Assistant:{RESET} {orch.chat(args.driver, q)['reply']}\n")


def cmd_chat(args) -> None:
    memory, orch = _system(args.db)
    if not memory.profile(args.driver):
        print(f"{DIM}Simulating {args.weeks} weeks for {args.driver} first...{RESET}")
        simulate_driver(memory, orch, args.driver, args.weeks)
    print(f"Chatting as {args.driver}. Type 'quit' to exit.")
    while True:
        try:
            text = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if text:
            print(f"{GREEN}Assistant:{RESET} {orch.chat(args.driver, text)['reply']}")


def cmd_profile(args) -> None:
    memory, orch = _system(args.db)
    if not memory.profile(args.driver, include_candidates=True):
        simulate_driver(memory, orch, args.driver, args.weeks)
    _print_profile(memory, args.driver, candidates=args.all)


def cmd_personas(_args) -> None:
    for p in PERSONAS.values():
        print(f"  {p.driver_id:6s} {p.name}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="drivememory", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("demo", cmd_demo), ("chat", cmd_chat), ("profile", cmd_profile), ("personas", cmd_personas)]:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if name != "personas":
            sp.add_argument("--driver", default="anna", choices=sorted(PERSONAS))
            sp.add_argument("--weeks", type=int, default=4)
            sp.add_argument("--seed", type=int, default=7)
            sp.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
        if name == "profile":
            sp.add_argument("--all", action="store_true", help="include candidate (not yet confident) memories")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
