"""Evolving virtual world with adversarial misinformation injection."""

import json
import random
from typing import Dict, List, Optional


class EvolvingWorld:
    """Simulates a temporally evolving web environment with adversarial content."""

    def __init__(self, corpus: List[Dict] = None):
        self.corpus = corpus or []
        self.time_steps = list(range(1, 8))  # t1 through t7
        self.injected_misinformation = {}

    def load_corpus(self, path: str):
        """Load adversarial corpus from JSONL file."""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus.append(json.loads(line))

    def inject_at_time(self, topic: str, time_step: int,
                       misinformation: Optional[str] = None) -> Dict:
        """Inject misinformation at a specific time step."""
        entry = {
            "topic": topic,
            "time_step": time_step,
            "type": "misinformation",
            "content": misinformation or f"Fake news about {topic}",
        }
        self.injected_misinformation[time_step] = entry
        return entry

    def get_corpus_at_time(self, time_step: int) -> List[Dict]:
        """Get corpus state at a specific time step."""
        # Return all items whose injection time <= current time step
        available = [c for c in self.corpus if c.get("time_step", 1) <= time_step]
        return available

    def simulate_adversarial_query(self, question: str, golden: str,
                                   inject_misinfo: bool = True) -> Dict:
        """Simulate an adversarial search scenario."""
        topic = self._extract_topic(question)
        adv_item = next((c for c in self.corpus if c.get("topic") == topic), None)

        result = {
            "question": question,
            "golden": golden,
            "adversarial": inject_misinfo and adv_item is not None,
            "misinformation": adv_item.get("misinformation") if adv_item else None,
            "truth": adv_item.get("truth") if adv_item else None,
        }
        return result

    def _extract_topic(self, question: str) -> str:
        """Extract topic from question for corpus matching."""
        question_lower = question.lower()
        for item in self.corpus:
            if item["topic"].lower() in question_lower:
                return item["topic"]
        # Fallback: use first word
        return question_lower.split()[0] if question.split() else "general"

    def create_temporal_scenario(self, question: str, golden: str) -> Dict:
        """Create a full temporal scenario for a question."""
        scenario = {
            "question": question,
            "golden": golden,
            "timeline": [],
            "adversarial_injections": [],
        }

        # Simulate knowledge evolution over time
        for t in self.time_steps:
            entry = {
                "time_step": t,
                "status": "available" if random.random() > 0.3 else "unavailable",
                "has_misinformation": False,
            }
            scenario["timeline"].append(entry)

        # Inject adversarial content at random time steps
        if random.random() > 0.5:
            adv_time = random.choice([2, 4, 6])
            topic = self._extract_topic(question)
            adv_item = next((c for c in self.corpus if c.get("topic") == topic), None)
            if adv_item:
                scenario["adversarial_injections"].append({
                    "time_step": adv_time,
                    "misinformation": adv_item["misinformation"],
                    "plausibility": adv_item.get("plausibility", "medium"),
                })
                scenario["timeline"][adv_time - 1]["has_misinformation"] = True

        return scenario


class StaticWorld:
    """Static environment baseline (no temporal dynamics, no adversarial content)."""

    def __init__(self):
        pass

    def simulate_query(self, question: str, golden: str) -> Dict:
        """Simulate a clean search scenario."""
        return {
            "question": question,
            "golden": golden,
            "adversarial": False,
            "misinformation": None,
            "truth": None,
        }


def create_world(mode: str = "evolving", corpus_path: str = None) -> EvolvingWorld:
    """Factory function to create world instances."""
    world = EvolvingWorld()
    if corpus_path:
        world.load_corpus(corpus_path)
    return world


def create_baseline_world() -> StaticWorld:
    """Create static baseline world."""
    return StaticWorld()
