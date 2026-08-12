"""MetaResearcher agent: ReAct-based deep research agent with meta-reward support."""

import json
import re
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Try Anthropic first, fallback to OpenAI
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SYSTEM_PROMPT = """You are MetaResearcher, a deep research assistant capable of conducting multi-step investigations.

You have access to two tools:
1. search: Perform Google web searches to find information
2. visit: Visit specific webpages to read detailed content

Your output must follow this format exactly:
- For thinking: <think>YOUR REASONING</think>
- For tool calls: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
- For final answer: <answer>YOUR ANSWER</answer>

Rules:
- Always think before acting
- Be efficient: avoid repeating the same search
- If you find conflicting information, note it and try to resolve
- Only output the final answer inside <answer> tags
- For yes/no questions, answer only yes or no
"""

SWARM_PROMPT_SCOUT = """You are the Scout agent in a research team. Your role is to generate high-quality search queries.

Given a research question and the current conversation history, generate 1-3 optimized search queries that would help find the answer.

Output format:
<tool_call>{"name": "search", "arguments": {"query": ["query1", "query2"]}}</tool_call>
"""

SWARM_PROMPT_FILTER = """You are the Filter agent in a research team. Your role is to assess the relevance of search results.

Given search results (title, URL, snippet), identify which results are most relevant to the research question.

Output your analysis:

<tool_call>{"name": "visit", "arguments": {"url": ["most_relevant_url"], "goal": "specific_information_goal"}}</tool_call>
"""

SWARM_PROMPT_SYNTHESIZER = """You are the Synthesizer agent in a research team. Your role is to produce the final answer.

Given all collected information, synthesize a comprehensive and accurate answer to the research question.

Output your final answer:
<answer>YOUR FINAL ANSWER</answer>
"""


def extract_answer(text: str) -> str:
    """Extract the answer from agent output."""
    if not isinstance(text, str):
        return ""
    # Try <answer> tags first
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try XML-style
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_thinking(text: str) -> str:
    """Extract thinking/reasoning from agent output."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_tool_calls(text: str) -> List[Dict]:
    """Extract tool calls from agent output."""
    calls = []
    pattern = r"<tool_call>\{(.+?)\}</tool_call>"
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


class MetaResearcherAgent:
    """Single-agent MetaResearcher with meta-reward support."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_steps: int = 10, system_prompt: str = SYSTEM_PROMPT,
                 base_url: Optional[str] = None, api_format: str = "auto"):
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt

        # Detect API format (default to OpenAI for compatibility)
        if api_format == "auto":
            self.api_format = "openai"  # Default to OpenAI format
        else:
            self.api_format = api_format

        # Initialize client based on format
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            raise RuntimeError("OpenAI client not available. Install 'openai' package.")
        self.reset()

    def reset(self):
        """Reset agent state."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.tool_calls_history = []
        self.step_count = 0
        self.start_time = None

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM API (supports both Anthropic and OpenAI formats)."""
        if self.api_format == "anthropic":
            # Anthropic format
            # Convert OpenAI-style messages to Anthropic format
            system_msg = None
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    anthropic_messages.append(msg)

            response = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=anthropic_messages,
                temperature=0.3,
                max_tokens=2048,
            )
            return response.content[0].text if response.content else ""
        else:
            # OpenAI format (default)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""

    def _simulate_search(self, query: str, adversarial: bool = False) -> str:
        """Simulate search results (for offline testing without API)."""
        # In production, this calls the LiteResearcher search server
        # For preliminary validation, we simulate with known answers
        results = f"Search results for '{query}':\n"
        if adversarial:
            # Return some adversarial content mixed with correct info
            results += "[ADVERSARIAL] This is misleading information about the topic.\n"
        results += "[LEGITIMATE] Accurate information about the search topic.\n"
        return results

    def _simulate_visit(self, url: str, goal: str) -> str:
        """Simulate webpage visit."""
        return f"Content from {url} relevant to: {goal}\n[LEGITIMATE] Key findings from the page."

    def run(self, question: str, adversarial: bool = False) -> Dict[str, Any]:
        """Run the agent on a question. Returns trajectory and metrics."""
        self.reset()
        self.start_time = time.time()
        self.messages.append({"role": "user", "content": question})

        trajectory = {
            "question": question,
            "steps": [],
            "tool_calls": [],
            "thinking": [],
            "final_answer": "",
            "correct": False,
            "adversarial": adversarial,
        }

        for step in range(self.max_steps):
            self.step_count = step + 1
            response = self._call_llm(self.messages)
            trajectory["steps"].append(response)

            # Extract thinking
            thought = extract_thinking(response)
            if thought:
                trajectory["thinking"].append(thought)

            # Extract tool calls
            tool_calls = extract_tool_calls(response)
            for tc in tool_calls:
                trajectory["tool_calls"].append(tc)
                self.tool_calls_history.append(tc)

                # Simulate tool execution
                if tc.get("name") == "search":
                    queries = tc.get("arguments", {}).get("query", [])
                    result = self._simulate_search(queries[0] if queries else "", adversarial)
                elif tc.get("name") == "visit":
                    url = tc.get("arguments", {}).get("url", [""])[0]
                    goal = tc.get("arguments", {}).get("goal", "general")
                    result = self._simulate_visit(url, goal)
                else:
                    result = "Tool not implemented"

                self.messages.append({"role": "user", "content": f"<tool_response>{result}</tool_call>"})

            # Check for final answer
            answer = extract_answer(response)
            if answer and answer not in ["", "I need more information"]:
                trajectory["final_answer"] = answer
                break

            # Add response to messages
            self.messages.append({"role": "assistant", "content": response})

        # Extract final answer if not already found
        if not trajectory["final_answer"]:
            trajectory["final_answer"] = extract_answer(self.messages[-1]["content"])

        trajectory["elapsed_time"] = time.time() - self.start_time
        trajectory["num_steps"] = self.step_count
        trajectory["num_tool_calls"] = len(trajectory["tool_calls"])
        trajectory["unique_queries"] = len(set(tc.get("arguments", {}).get("query", [""])[0]
                                               for tc in trajectory["tool_calls"] if tc.get("name") == "search"))

        return trajectory

    def compute_meta_reward(self, trajectory: Dict, golden: str) -> Dict[str, float]:
        """Compute meta-reward components for a trajectory."""
        reward = {}

        # 1. Correctness reward
        pred = trajectory["final_answer"].lower().strip()
        gold = golden.lower().strip()
        # Simple string matching for preliminary validation
        reward["correctness"] = 1.0 if gold in pred or pred in gold else 0.0

        # 2. Efficiency reward
        T_min, T_max = 3, 10
        t = trajectory["num_steps"]
        if t <= T_min:
            reward["efficiency"] = 1.0
        elif t <= T_max:
            reward["efficiency"] = 1.0 - 0.5 * (t - T_min) / (T_max - T_min)
        else:
            reward["efficiency"] = 0.0

        # 3. Reflection depth reward
        beta_1, beta_2, beta_3 = 0.3, 0.3, 0.4
        backtrack_pattern = r"mistaken|incorrect|reconsider|wait|however|actually"
        has_backtrack = any(re.search(p, t.lower()) for t in trajectory["thinking"])
        reward["reflection_backtrack"] = 1.0 if has_backtrack else 0.0

        # Strategy change detection (simplified)
        queries = [tc.get("arguments", {}).get("query", [""])[0]
                   for tc in trajectory["tool_calls"] if tc.get("name") == "search"]
        strategy_change = 1.0 if len(set(q.lower()[:20] for q in queries)) > 1 else 0.0
        reward["reflection_strategy_change"] = strategy_change

        # Source diversity
        n_distinct = trajectory.get("unique_queries", 0)
        n_total = max(trajectory["num_tool_calls"], 1)
        reward["reflection_source_diversity"] = n_distinct / n_total

        reward["reflection"] = (beta_1 * reward["reflection_backtrack"] +
                                beta_2 * reward["reflection_strategy_change"] +
                                beta_3 * reward["reflection_source_diversity"])

        # 4. Diversity reward
        gamma = 0.5
        unique_queries = trajectory.get("unique_queries", 0)
        total_calls = max(trajectory["num_tool_calls"], 1)
        reward["diversity"] = gamma * (unique_queries / total_calls) + (1 - gamma) * 0.5

        # 5. Meta-reward (weighted sum)
        w_c, w_e, w_r, w_d = 0.4, 0.2, 0.2, 0.2
        reward["meta"] = (w_c * reward["correctness"] +
                         w_e * reward["efficiency"] +
                         w_r * reward["reflection"] +
                         w_d * reward["diversity"])

        return reward


class SwarmAgent:
    """Multi-agent swarm: Scout + Filter + Synthesizer."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_steps: int = 10, base_url: Optional[str] = None):
        self.model = model
        self.max_steps = max_steps
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.scout = MetaResearcherAgent(model=model, api_key=self.api_key,
                                         max_steps=3, system_prompt=SWARM_PROMPT_SCOUT,
                                         base_url=self.base_url)
        self.filter = MetaResearcherAgent(model=model, api_key=self.api_key,
                                          max_steps=3, system_prompt=SWARM_PROMPT_FILTER,
                                          base_url=self.base_url)
        self.synthesizer = MetaResearcherAgent(model=model, api_key=self.api_key,
                                               max_steps=2, system_prompt=SWARM_PROMPT_SYNTHESIZER,
                                               base_url=self.base_url)

    def run(self, question: str, adversarial: bool = False) -> Dict[str, Any]:
        """Run the swarm on a question."""
        # Phase 1: Scout generates queries
        scout_traj = self.scout.run(question, adversarial=adversarial)
        scout_queries = [tc.get("arguments", {}).get("query", [""])[0]
                        for tc in scout_traj["tool_calls"] if tc.get("name") == "search"]

        # Phase 2: Filter assesses and selects
        filter_traj = self.filter.run(question, adversarial=adversarial)
        filter_urls = [tc.get("arguments", {}).get("url", [""])[0]
                      for tc in filter_traj["tool_calls"] if tc.get("name") == "visit"]

        # Phase 3: Synthesizer produces final answer
        # Combine all information
        collected_info = f"Question: {question}\n"
        collected_info += f"Scout queries: {scout_queries}\n"
        collected_info += f"Filter URLs: {filter_urls}\n"
        collected_info += f"Scout reasoning: {scout_traj['thinking']}\n"
        collected_info += f"Filter reasoning: {filter_traj['thinking']}\n"

        self.synthesizer.messages = [
            {"role": "system", "content": SWARM_PROMPT_SYNTHESIZER},
            {"role": "user", "content": collected_info}
        ]
        synthesize_response = self.synthesizer._call_llm(self.synthesizer.messages)
        final_answer = extract_answer(synthesize_response)

        return {
            "question": question,
            "final_answer": final_answer,
            "num_steps": scout_traj["num_steps"] + filter_traj["num_steps"] + 1,
            "num_tool_calls": len(scout_traj["tool_calls"]) + len(filter_traj["tool_calls"]),
            "unique_queries": len(set(scout_queries)),
            "correct": False,  # Will be set by evaluator
            "adversarial": adversarial,
            "scout_trajectory": scout_traj,
            "filter_trajectory": filter_traj,
            "synthesize_response": synthesize_response,
        }

    def compute_meta_reward(self, trajectory: Dict, golden: str) -> Dict[str, float]:
        """Compute meta-reward for swarm trajectory."""
        # Use synthesizer's trajectory for reward computation
        synth_traj = {"final_answer": trajectory["final_answer"],
                      "num_steps": trajectory["num_steps"],
                      "num_tool_calls": trajectory["num_tool_calls"],
                      "unique_queries": trajectory["unique_queries"],
                      "thinking": [],
                      "adversarial": trajectory.get("adversarial", False)}
        agent = MetaResearcherAgent(model=self.model)
        return agent.compute_meta_reward(synth_traj, golden)
