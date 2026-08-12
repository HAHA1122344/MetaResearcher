"""Experiment configuration."""
MODEL = "gpt-4o-mini"
MAX_STEPS = 10
NUM_ROLLOUTS = 3
ADVERSARIAL = True
REWARD_WEIGHTS = {"correctness": 0.4, "efficiency": 0.2, "reflection": 0.2, "diversity": 0.2}
BETA = {"backtrack": 0.3, "strategy_change": 0.3, "source_diversity": 0.4}
GAMMA = 0.5
T_MIN = 3
T_MAX = 10
LAMBDA_TEAM = 0.1
ALPHA = 0.5
