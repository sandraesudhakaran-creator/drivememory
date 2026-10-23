"""The orchestrator: the conversational front of the in-car assistant.

On trip start it fans out A2A tasks to the specialist agents, merges their
recommendations into vehicle presets and a greeting, and remembers what it did
so it can explain itself ("why is it 21.5 degrees?"). In chat it handles
explicit preferences, "forget" requests (privacy), profile questions and
memory search, and falls back to an LLM grounded in the profile if configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..llm import LLM, get_llm
from ..memory import MemoryManager
from ..signals.models import TripContext
from .describe import describe, key_from_text
from .protocol import AgentBus, Message, Part
from .specialists import AutonomyAgent, ComfortAgent, NavigationAgent

PRESETTABLE = {
    "climate.cabin_temp_c", "climate.seat_heating", "media.source", "media.volume",
    "autonomy.following_gap_s", "autonomy.rain_handover_advisory",
}
FORGET_DOMAINS = {
    "media": ["music", "media", "podcast", "radio", "listen"],
    "climate": ["temperature", "climate", "seat", "heating"],
    "autonomy": ["autonomy", "autopilot", "self-driving", "driving style", "gap"],
    "routine": ["places", "destination", "routine", "where i go", "location", "trips"],
}


@dataclass
class TripStartResult:
    driver_id: str
    presets: dict[str, Any]
    greeting: str
    recommendations: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)


class Orchestrator:
    def __init__(self, memory: MemoryManager, llm: LLM | None = None, use_llm: bool = True):
        self.memory = memory
        self.llm = llm if llm is not None else (get_llm() if use_llm else None)
        self.bus = AgentBus()
        for agent in (ComfortAgent(memory), NavigationAgent(memory), AutonomyAgent(memory)):
            self.bus.register(agent)
        self._last: dict[str, TripStartResult] = {}

    # ------------------------------------------------------------ trip start
    def on_trip_start(self, ctx: TripContext, compose: bool = True) -> TripStartResult:
        ctx_data = {
            "driver_id": ctx.driver_id, "departure": ctx.departure.isoformat(),
            "outside_temp_c": ctx.outside_temp_c, "origin": ctx.origin,
            "rain_expected": ctx.rain_expected, "expected_duration_min": ctx.expected_duration_min,
        }
        recs: list[dict] = []
        tasks: list[dict] = []
        for name in self.bus.agents:
            msg = Message("user", [Part.of_data({"skill": "recommend", "context": ctx_data})],
                          context_id=f"trip-{ctx.driver_id}-{ctx.departure:%Y%m%d%H%M}")
            task = self.bus.send(name, msg)
            tasks.append({"agent": name, "task_id": task.id, "state": task.state, "error": task.error})
            recs.extend(task.result().get("recommendations", []))

        presets = {r["key"]: r["value"] for r in recs
                   if r["key"] in PRESETTABLE and r["action"] in ("apply", "advise")}
        greeting = self._greeting(ctx, recs) if compose else ""
        result = TripStartResult(ctx.driver_id, presets, greeting, recs, tasks)
        self._last[ctx.driver_id] = result
        return result

    def _greeting(self, ctx: TripContext, recs: list[dict]) -> str:
        if not recs:
            return "Good to see you. I'm still getting to know your preferences, so I've kept the defaults."
        if self.llm:
            try:
                bullet = "\n".join(f"- {r['action']}: {r['key']} = {r['value']} ({r['reason']})" for r in recs)
                return self.llm.complete(
                    "You are a concise, warm in-car assistant. Write a 1-3 sentence spoken greeting for the "
                    "start of a trip. Mention applied settings briefly, voice suggestions as questions, and "
                    "state safety advisories clearly. No lists, no emojis.",
                    f"Time: {ctx.departure:%A %H:%M}, outside {ctx.outside_temp_c}°C, rain expected: "
                    f"{ctx.rain_expected}.\nRecommendations:\n{bullet}",
                    max_tokens=150,
                )
            except Exception:
                pass  # fall back to the template below
        hello = {"morning": "Good morning", "evening": "Good evening"}.get(ctx.slot, "Hello")
        parts: list[str] = []
        applied = [r for r in recs if r["action"] == "apply"]
        bits = []
        for r in applied:
            if r["key"] == "climate.cabin_temp_c":
                bits.append(f"the cabin is set to {r['value']:g}°C")
            elif r["key"] == "climate.seat_heating" and r["value"]:
                bits.append(f"seat heating is on level {r['value']}")
            elif r["key"] == "media.source":
                bits.append(f"{r['value'].replace('_', ' ')} is ready")
        if bits:
            parts.append(", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1] + ".")
        for r in recs:
            if r["key"] == "routine.destination":
                dest = str(r["value"])
                parts.append("Heading home?" if dest == "home" else f"Heading to {dest.replace('_', ' ')}?")
            elif r["key"] == "routine.suggest_break":
                parts.append(f"It's a long drive; want me to find a café about {r['value']['after_min']} "
                             "minutes in?")
            elif r["key"] == "autonomy.rain_handover_advisory" and r["value"] is True:
                parts.append("Rain is expected on the highway, so I'll give you an early, planned handover "
                             "before it starts.")
        text = " ".join(parts)
        return f"{hello}! {text[0].upper() + text[1:]}" if text else f"{hello}!"

    # ------------------------------------------------------------------ chat
    def chat(self, driver_id: str, text: str) -> dict:
        t = text.lower().strip()
        for handler in (self._forget, self._stated, self._why, self._profile, self._recall_episode):
            out = handler(driver_id, t, text)
            if out:
                return out
        return self._fallback(driver_id, text)

    def _reply(self, intent: str, text: str, **data) -> dict:
        return {"intent": intent, "reply": text, **({"data": data} if data else {})}

    def _forget(self, d: str, t: str, raw: str) -> dict | None:
        if not re.search(r"\b(forget|delete|erase|remove)\b", t):
            return None
        if re.search(r"\b(everything|all|me)\b", t) and not any(
                w in t for ws in FORGET_DOMAINS.values() for w in ws):
            n = self.memory.forget(d)
            return self._reply("forget", f"Done. I've deleted everything I knew about you ({n} items).", deleted=n)
        for domain, words in FORGET_DOMAINS.items():
            if any(w in t for w in words):
                n = self.memory.forget(d, domain=domain)
                return self._reply("forget", f"Done. I've forgotten your {domain} preferences ({n} items).",
                                   deleted=n, domain=domain)
        return self._reply("forget", "What should I forget? Your music, climate, driving or routine preferences, "
                                     "or everything?")

    def _stated(self, d: str, t: str, raw: str) -> dict | None:
        if not re.search(r"\b(i prefer|i like|i want|i'd like|i'm|i am|always|set|i don't|i do not|i hate)\b", t):
            return None
        m = re.search(r"(\d{2}(?:[.,]5)?)\s*(?:°|degrees|deg\b|c\b)", t)
        if m:
            v = float(m.group(1).replace(",", "."))
            self.memory.remember(d, "climate.cabin_temp_c", "*", v, "numeric", raw)
            return self._reply("remember", f"Got it, I'll keep the cabin at {v:g}°C from now on.",
                               key="climate.cabin_temp_c", value=v)
        m = re.search(r"(?:gap|distance)[^\d]*(\d(?:[.,]\d)?)|(\d(?:[.,]\d)?)\s*(?:s|sec|seconds)\b", t)
        if m and ("gap" in t or "distance" in t):
            v = float((m.group(1) or m.group(2)).replace(",", "."))
            self.memory.remember(d, "autonomy.following_gap_s", "highway", v, "numeric", raw)
            return self._reply("remember", f"Noted: I'll keep a {v:g}-second gap in autonomous mode.",
                               key="autonomy.following_gap_s", value=v)
        if "rain" in t and any(w in t for w in ("autonomy", "autopilot", "self-driving", "drive myself", "take over")):
            negative = any(w in t for w in ("don't", "do not", "not comfortable", "hate", "nervous", "drive myself",
                                             "take over", "uncomfortable"))
            self.memory.remember(d, "autonomy.rain_comfort", "rain", not negative, "bool", raw)
            msg = ("Understood. When rain is coming I'll plan an early handover so you can take over calmly."
                   if negative else "Understood, I'll keep autonomy available in rain.")
            return self._reply("remember", msg, key="autonomy.rain_comfort", value=not negative)
        m = re.search(r"(?:i like|i prefer|i want)\s+(?:to listen to\s+)?([a-z_ ]+?)\s+in the (morning|evening)", t)
        if m:
            src = m.group(1).strip().replace(" ", "_")
            self.memory.remember(d, "media.source", m.group(2), src, "categorical", raw)
            return self._reply("remember", f"Nice, I'll queue up {src.replace('_', ' ')} in the {m.group(2)}.",
                               key="media.source", value=src)
        return None

    def _why(self, d: str, t: str, raw: str) -> dict | None:
        if not re.search(r"\b(why|how come|how did you know)\b", t):
            return None
        key = key_from_text(t)
        last = self._last.get(d)
        rec = next((r for r in (last.recommendations if last else []) if r["key"] == key), None)
        if rec is None and key:
            cands = sorted((p for p in self.memory.profile(d) if p.key == key),
                           key=lambda p: p.confidence, reverse=True)
            named = [p for p in cands if any(tok in t for tok in str(p.value).lower().split("_") if len(tok) > 2)]
            p = (named or cands or [None])[0]
            if p:
                rec = {"key": p.key, "value": p.value, "reason": describe(p), "source": p.id}
        if rec is None:
            return self._reply("explain", "I didn't change that based on memory; it's the vehicle default.")
        src = rec.get("source") or ""
        key_, _, ctx = src.partition("@")
        ex = self.memory.explain(d, key_, ctx or "*") if src else None
        detail = ""
        if ex:
            n = ex["evidence_count"]
            basis = "you told me so" if ex["user_confirmed"] else f"I've seen it on about {round(n)} trips"
            last_ev = ex["recent_evidence"][-1]["evidence"] if ex["recent_evidence"] else ""
            detail = f" {basis.capitalize()} (confidence {ex['confidence']:.0%})."
            if last_ev:
                detail += f" Most recently: {last_ev}."
        return self._reply("explain", f"Because {rec['reason']}.{detail} You can say \"forget that\" any time.",
                           explanation=ex)

    def _profile(self, d: str, t: str, raw: str) -> dict | None:
        if not re.search(r"(what do you know|about me|my profile|my preferences|what have you learned)", t):
            return None
        prefs = [p for p in self.memory.profile(d)
                 if not (p.key == "routine.destination" and p.context.split("-")[0] in
                         ("mon", "tue", "wed", "thu", "fri", "sat", "sun") and p.value in ("home", "work"))]
        if not prefs:
            return self._reply("profile", "I don't know much about you yet. I learn quietly as you drive.")
        lines = [describe(p) for p in prefs]
        return self._reply("profile", "Here's what I've learned: " + "; ".join(lines) + ".",
                           preferences=[p.id for p in prefs])

    def _recall_episode(self, d: str, t: str, raw: str) -> dict | None:
        if not re.search(r"\b(when did i|last time|remember when|have i ever)\b", t):
            return None
        hits = self.memory.search(d, raw, k=3)
        if not hits:
            return self._reply("search", "I don't have any trips that match.")
        top = hits[0]
        when = datetime.fromisoformat(top["timestamp"]).strftime("%A %d %B at %H:%M")
        return self._reply("search", f"Closest match: {when}. {top['summary']}", results=hits)

    def _fallback(self, d: str, raw: str) -> dict:
        if self.llm:
            try:
                prof = "\n".join(f"- {describe(p)}" for p in self.memory.profile(d))
                ctx = "\n".join(f"- {h['summary']}" for h in self.memory.search(d, raw, k=3))
                answer = self.llm.complete(
                    "You are an in-car assistant. Answer briefly (1-3 sentences) using ONLY the driver memory "
                    "below. If the memory doesn't contain the answer, say so.",
                    f"Driver profile:\n{prof or '(empty)'}\n\nRelevant trips:\n{ctx or '(none)'}\n\n"
                    f"Driver: {raw}",
                    max_tokens=200,
                )
                return self._reply("llm", answer)
            except Exception:
                pass
        return self._reply("help", "I can explain my settings (\"why is it so warm?\"), remember preferences "
                                   "(\"I prefer 20 degrees\"), forget things (\"forget my music\"), or tell you "
                                   "what I've learned (\"what do you know about me?\").")
