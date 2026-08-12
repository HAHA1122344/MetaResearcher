"""Meta-reward computation for MetaResearcher."""

import re
from typing import Dict, List, Tuple


class MetaRewardComputer:
    """Computes the four-component meta-reward for agent trajectories."""

    def __init__(self, weights: Dict[str, float] = None,
                 beta: Dict[str, float] = None,
                 gamma: float = 0.5,
                 T_min: int = 3, T_max: int = 10, alpha: float = 0.5):
        self.w_c = weights["correctness"] if weights else 0.4
        self.w_e = weights["efficiency"] if weights else 0.2
        self.w_r = weights["reflection"] if weights else 0.2
        self.w_d = weights["diversity"] if weights else 0.2

        self.beta_1 = beta["backtrack"] if beta else 0.3
        self.beta_2 = beta["strategy_change"] if beta else 0.3
        self.beta_3 = beta["source_diversity"] if beta else 0.4

        self.gamma = gamma
        self.T_min = T_min
        self.T_max = T_max
        self.alpha = alpha

    def compute_correctness(self, predicted: str, golden: str) -> float:
        """Binary correctness reward."""
        pred = predicted.lower().strip()
        gold = golden.lower().strip()
        # Normalized matching: check if golden is contained in predicted or vice versa
        if gold in pred or pred in gold:
            return 1.0
        # Also check numeric equivalence
        try:
            pred_num = float(re.search(r'[\d,]+\.?\d*', pred.replace(',', ''))
                           .group() if re.search(r'[\d,]+\.?\d*', pred) else '0')
            gold_num = float(re.search(r'[\d,]+\.?\d*', gold.replace(',', ''))
                            .group() if re.search(r'[\d,]+\.?\d*', gold) else '0')
            if abs(pred_num - gold_num) / max(gold_num, 1) < 0.1:
                return 1.0
        except:
            pass
        return 0.0

    def compute_efficiency(self, num_steps: int) -> float:
        """Efficiency reward based on step count."""
        if num_steps <= self.T_min:
            return 1.0
        elif num_steps <= self.T_max:
            return 1.0 - self.alpha * (num_steps - self.T_min) / (self.T_max - self.T_min)
        else:
            return 0.0

    def compute_reflection(self, thinking: List[str], tool_calls: List[Dict],
                          unique_queries: int, total_calls: int) -> float:
        """Reflection depth reward."""
        backtrack_patterns = r"mistaken|incorrect|reconsider|wait|however|actually|i was wrong"
        has_backtrack = any(re.search(backtrack_patterns, t.lower()) for t in thinking)

        # Strategy change: detect distinct query patterns
        query_prefixes = set()
        for tc in tool_calls:
            if tc.get("name") == "search":
                queries = tc.get("arguments", {}).get("query", [])
                for q in queries:
                    query_prefixes.add(q.lower()[:30])
        has_strategy_change = len(query_prefixes) > 1

        # Source diversity
        source_diversity = unique_queries / max(total_calls, 1)

        reward = (self.beta_1 * (1.0 if has_backtrack else 0.0) +
                 self.beta_2 * (1.0 if has_strategy_change else 0.0) +
                 self.beta_3 * source_diversity)

        return min(reward, 1.0)  # Cap at 1.0

    def compute_diversity(self, unique_queries: int, total_calls: int,
                         unique_domains: int = None, total_visits: int = None) -> float:
        """Tool call diversity reward."""
        query_diversity = unique_queries / max(total_calls, 1)
        if unique_domains is not None and total_visits is not None:
            domain_diversity = unique_domains / max(total_visits, 1)
            return self.gamma * query_diversity + (1 - self.gamma) * domain_diversity
        return query_diversity

    def compute_meta_reward(self, trajectory: Dict, golden: str) -> Dict[str, float]:
        """Compute full meta-reward for a trajectory."""
        r_correctness = self.compute_correctness(
            trajectory.get("final_answer", ""), golden)
        r_efficiency = self.compute_efficiency(trajectory.get("num_steps", 0))
        r_reflection = self.compute_reflection(
            trajectory.get("thinking", []),
            trajectory.get("tool_calls", []),
            trajectory.get("unique_queries", 0),
            trajectory.get("num_tool_calls", 0))
        r_diversity = self.compute_diversity(
            trajectory.get("unique_queries", 0),
            trajectory.get("num_tool_calls", 0))

        meta = (self.w_c * r_correctness +
               self.w_e * r_efficiency +
               self.w_r * r_reflection +
               self.w_d * r_diversity)

        return {
            "correctness": r_correctness,
            "efficiency": r_efficiency,
            "reflection": r_reflection,
            "diversity": r_diversity,
            "meta": meta,
        }

    def compute_team_reward(self, individual_rewards: List[float]) -> float:
        """Compute team reward (Eq. 8, revised for normalization).

        Original: L_team = -log(σ(1/3 Σ r_a^indiv))
        Revised: Use sigmoid on normalized rewards (clamped to [0, 10])
        """
        import math
        # Clamp individual rewards to reasonable range before averaging
        normalized = [max(0.0, min(10.0, r)) for r in individual_rewards]
        avg = sum(normalized) / len(normalized)
        # Sigmoid
        sigma = 1.0 / (1.0 + math.exp(-avg))
        return -math.log(sigma)

    def count_identical_calls(self, tool_calls: List[Dict]) -> Tuple[int, int]:
        """Count identical vs. unique tool calls for loop analysis."""
        call_signatures = []
        for tc in tool_calls:
            if tc.get("name") == "search":
                queries = tc.get("arguments", {}).get("query", [])
                for q in queries:
                    call_signatures.append(q.lower().strip())
            elif tc.get("name") == "visit":
                url = tc.get("arguments", {}).get("url", [""])[0]
                call_signatures.append(url.lower().strip())

        total = len(call_signatures)
        unique = len(set(call_signatures))
        identical = total - unique
        return identical, total


class RewardAnalyzer:
    """Analyzes reward component contributions across multiple trajectories."""

    def __init__(self):
        self.computer = MetaRewardComputer()

    def analyze_batch(self, trajectories: List[Dict], goldens: List[str]) -> Dict:
        """Analyze reward components across a batch of trajectories."""
        all_rewards = []
        for traj, golden in zip(trajectories, goldens):
            r = self.computer.compute_meta_reward(traj, golden)
            r["correct"] = r["correctness"] > 0.5
            all_rewards.append(r)

        # Aggregate statistics
        n = len(all_rewards)
        if n == 0:
            return {}

        return {
            "n_samples": n,
            "accuracy": sum(1 for r in all_rewards if r["correct"]) / n,
            "avg_correctness": sum(r["correctness"] for r in all_rewards) / n,
            "avg_efficiency": sum(r["efficiency"] for r in all_rewards) / n,
            "avg_reflection": sum(r["reflection"] for r in all_rewards) / n,
            "avg_diversity": sum(r["diversity"] for r in all_rewards) / n,
            "avg_meta": sum(r["meta"] for r in all_rewards) / n,
            "std_meta": (sum((r["meta"] - sum(r2["meta"] for r2 in all_rewards) / n) ** 2
                            for r in all_rewards) / n) ** 0.5,
        }

    def compare_rewards(self, traj_meta: List[Dict], traj_baseline: List[Dict],
                        goldens: List[str]) -> Dict:
        """Compare meta-reward vs. outcome-only reward across trajectories."""
        meta_results = self.analyze_batch(traj_meta, goldens)
        # For baseline, only correctness matters
        n = len(traj_baseline)
        baseline_correct = sum(1 for t in traj_baseline
                              for g in [goldens[0]] if  # simplified
                              True)  # will be filled properly in run_experiment
        baseline_accuracy = sum(1 for t, g in zip(traj_baseline, goldens)
                                if self.computer.compute_correctness(
                                    t.get("final_answer", ""), g) > 0.5) / n

        return {
            "meta_accuracy": meta_results.get("accuracy", 0),
            "baseline_accuracy": baseline_accuracy,
            "meta_avg_reward": meta_results.get("avg_meta", 0),
            "improvement": meta_results.get("accuracy", 0) - baseline_accuracy,
        }
