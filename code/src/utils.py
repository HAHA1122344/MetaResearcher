"""Utils for MetaResearcher experiments."""

import json
import os
import re
import random
from typing import Dict, List, Tuple


def load_config(config_path: str = None) -> Dict:
    """Load experiment configuration."""
    if config_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, "configs", "config.yaml")

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            # Fallback: parse simple YAML manually
            return _parse_simple_yaml(config_path)
    return _default_config()


def _parse_simple_yaml(path: str) -> Dict:
    """Simple YAML parser for basic configs."""
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                config[key.strip()] = val.strip().strip('"').strip("'")
    return config


def _default_config() -> Dict:
    """Default configuration."""
    return {
        "model": "gpt-4o-mini",
        "max_steps": 10,
        "adversarial": True,
        "num_rollouts": 3,
        "reward_weights": {"correctness": 0.4, "efficiency": 0.2,
                          "reflection": 0.2, "diversity": 0.2},
        "beta": {"backtrack": 0.3, "strategy_change": 0.3, "source_diversity": 0.4},
        "gamma": 0.5,
        "T_min": 3,
        "T_max": 10,
    }


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer.lower()


def compute_accuracy(predictions: List[str], goldens: List[str]) -> float:
    """Compute accuracy between predictions and goldens."""
    if not predictions or not goldens:
        return 0.0
    correct = 0
    for pred, gold in zip(predictions, goldens):
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        is_correct = False
        # String containment check
        if gold_norm in pred_norm or pred_norm in gold_norm:
            is_correct = True
        # Numeric check (only if not already matched)
        if not is_correct:
            try:
                pred_match = re.search(r'[\d,]+\.?\d*', pred_norm.replace(',', ''))
                gold_match = re.search(r'[\d,]+\.?\d*', gold_norm.replace(',', ''))
                if pred_match and gold_match:
                    pred_num = float(pred_match.group())
                    gold_num = float(gold_match.group())
                    if abs(pred_num - gold_num) / max(abs(gold_num), 1) < 0.15:
                        is_correct = True
            except:
                pass
        if is_correct:
            correct += 1
    return correct / len(predictions)


def compute_loop_rate(trajectories: List[Dict]) -> Tuple[float, float]:
    """Compute loop rate (identical call ratio) across trajectories."""
    total_calls = 0
    identical_calls = 0
    for traj in trajectories:
        tool_calls = traj.get("tool_calls", [])
        signatures = []
        for tc in tool_calls:
            if tc.get("name") == "search":
                queries = tc.get("arguments", {}).get("query", [])
                for q in queries:
                    signatures.append(q.lower().strip())
            elif tc.get("name") == "visit":
                url = tc.get("arguments", {}).get("url", [""])[0]
                signatures.append(url.lower().strip())
        total_calls += len(signatures)
        unique = len(set(signatures))
        identical_calls += len(signatures) - unique

    rate = identical_calls / max(total_calls, 1)
    return rate, total_calls


def save_results(results: Dict, output_path: str):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def load_results(input_path: str) -> Dict:
    """Load experiment results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Format a table as ASCII text."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Format
    lines = []
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    lines.append(header_line)
    lines.append(separator)
    for row in rows:
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
    return "\n".join(lines)


def compute_confidence_interval(accuracy: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96 if confidence == 0.95 else 2.576
    p = float(accuracy)
    p = max(0.0, min(1.0, p))  # Clamp to [0, 1]
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    variance_term = p * (1.0 - p) / n + z * z / (4.0 * n * n)
    variance_term = max(0.0, variance_term)  # Prevent negative due to float errors
    margin = z * (variance_term ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def simulate_trajectory(question: str, golden: str, level: str = "1",
                        adversarial: bool = False, use_meta_reward: bool = True,
                        seed: int = None) -> Dict:
    """Simulate a single agent trajectory for preliminary validation.

    Produces realistic accuracy patterns based on:
    - Question difficulty level (1=easy, 2=medium, 3=hard)
    - Whether adversarial content is injected
    - Whether meta-reward is used (affects efficiency/diversity)
    """
    # Use deterministic seed based on question string (hash() varies across Python sessions)
    if seed is not None:
        rng_seed = seed
    else:
        rng_seed = int.from_bytes(question.encode(), 'big') % (2**31)
    rng = random.Random(rng_seed)

    # Difficulty-dependent base accuracy (mimicking LiteResearcher ~71.3% on full GAIA)
    base_acc = {"1": 0.85, "2": 0.65, "3": 0.45}
    accuracy_prob = base_acc.get(level, 0.6)

    # Adversarial content reduces accuracy
    if adversarial:
        accuracy_prob *= 0.65  # ~35% drop under adversarial conditions

    # Meta-reward improves efficiency but not necessarily correctness in simulation
    # (The real benefit comes from training, not inference)

    # Determine number of steps (meta-reward encourages efficiency)
    if use_meta_reward:
        num_steps = rng.randint(2, 6)  # More efficient
    else:
        num_steps = rng.randint(3, 10)  # Less efficient, more loops

    # Generate tool calls
    tool_calls = []
    for _ in range(num_steps):
        if rng.random() > 0.25:
            # Search call with varied queries
            query_variants = [
                f"what is {golden.lower()}",
                f"{question[:20].lower()} facts",
                f"location of {golden.lower()}",
                f"who wrote {golden.lower()}",
            ]
            query = rng.choice(query_variants)
            tool_calls.append({"name": "search", "arguments": {"query": [query]}})
        else:
            tool_calls.append({
                "name": "visit",
                "arguments": {"url": [f"https://example.com/page_{rng.randint(1,20)}"],
                             "goal": rng.choice(["find facts", "verify claim", "get details"])}
            })

    # Generate thinking traces
    thinking = []
    thinking.append(rng.choice([
        "I need to search for information about this topic.",
        "Let me look up the relevant details.",
        "I'll search for more context.",
    ]))
    # Backtrack signal (more likely with meta-reward)
    if use_meta_reward and rng.random() > 0.5:
        thinking.append(rng.choice([
            "Wait, let me reconsider my approach.",
            "I may have been mistaken. Let me check another source.",
            "Actually, I should verify this information.",
        ]))
    elif rng.random() > 0.7:
        thinking.append(rng.choice([
            "Hmm, this doesn't seem right. Let me try a different search.",
            "I found conflicting information. Let me check again.",
        ]))

    # Generate answer
    if rng.random() < accuracy_prob:
        final_answer = f"The answer is {golden}."
    else:
        # Generate a plausible but wrong answer
        wrong_answers = {
            "Australia": "Sydney", "1984": "Aldous Huxley", "gold": "Ag",
            "continents": "5", "World War II": "1944", "ocean": "Atlantic",
            "Mona Lisa": "Michelangelo", "boiling": "90", "prime": "1",
            "Red Planet": "Venus", "Japan": "100 million", "Moon": "Buzz Aldrin, 1968",
            "currency": "Dollar", "79": "Gadolinium", "Moon distance": "400,000",
            "relativity": "Niels Bohr", "river": "Amazon", "iPhone": "2006",
            "square root": "14", "Tesla": "Jeff Bezos", "Sputnik": "1958",
            "time zones": "Russia", "deepest": "Puerto Rico Trench",
            "Nobel": "Marie Curie, Chemistry", "galaxy": "Sombrero",
            "Berlin Wall": "1987", "atmosphere": "Oxygen",
            "border": "Mexico and United States", "particle": "Z boson",
            "Darwin": "Alfred Wallace, 1860",
        }
        # Extract topic from golden for wrong answer lookup
        topic_key = None
        for k in wrong_answers:
            if k.lower() in question.lower() or k.lower() in golden.lower():
                topic_key = k
                break
        if topic_key:
            final_answer = f"The answer is {wrong_answers[topic_key]}."
        else:
            final_answer = f"The answer is approximately {rng.choice(['unknown', 'unclear', 'requires more research'])}."

    # Count unique queries
    search_queries = [tc["arguments"]["query"][0] for tc in tool_calls if tc["name"] == "search"]
    unique_queries = len(set(q.lower() for q in search_queries))

    trajectory = {
        "question": question,
        "final_answer": final_answer,
        "num_steps": num_steps,
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "thinking": thinking,
        "unique_queries": unique_queries,
        "adversarial": adversarial,
        "use_meta_reward": use_meta_reward,
        "level": level,
    }
    return trajectory
