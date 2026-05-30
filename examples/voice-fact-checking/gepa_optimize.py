#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Optimize the fact-checking trigger prompt with GEPA.

GEPA (Genetic-Pareto, Agrawal et al. 2025) is a *reflective* prompt optimizer:
instead of blind search it runs the current prompt on a few examples, reads the
natural-language feedback from its mistakes, and asks a reflection LLM to write
a better prompt. Good prompts are kept on a Pareto frontier (best on at least
one example) so the search does not collapse onto a single average-case winner.

Here the thing being optimized is ``DEFAULT_EXTRACTOR_PROMPT`` from
``prompts.py`` -- i.e. *what counts as a check-worthy claim* (the trigger) and
how queries are written. The metric rewards correct triggering (don't fire on
opinions/small talk, do fire on verifiable facts) and good claim coverage.

Usage::

    # Real optimization (needs OPENAI_API_KEY):
    uv run python examples/voice-fact-checking/gepa_optimize.py \
        --budget 60 --task-model gpt-4o-mini --reflection-model gpt-4o

    # Offline self-test of the GEPA loop (no API key, deterministic):
    uv run python examples/voice-fact-checking/gepa_optimize.py --self-test

The best prompt found is written to ``optimized_extractor_prompt.txt``, which
``prompts.load_extractor_prompt()`` picks up automatically the next time the bot
runs. Nothing else needs to change to ship an improved trigger.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
DATASET_PATH = HERE / "dataset.json"
OUTPUT_PATH = HERE / "optimized_extractor_prompt.txt"

# A "task function" maps (prompt, utterance) -> list of extracted claim strings.
TaskFn = Callable[[str, str], list[str]]
# A "reflection function" maps (current_prompt, feedback) -> improved prompt.
ReflectFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Metric: score one example and produce natural-language feedback for GEPA.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score_example(predicted: list[str], example: dict) -> tuple[float, str]:
    """Score predicted claims against an example, returning (score, feedback).

    The feedback string is the signal GEPA reflects on, so it is written for an
    LLM to read: it names false triggers, missed claims, and over-extraction.
    """
    should = bool(example["should_trigger"])
    expected = example.get("claims", [])
    triggered = len(predicted) > 0

    if not should:
        if not triggered:
            return 1.0, "Correctly stayed silent on a non-factual / chit-chat line."
        return (
            0.0,
            "FALSE TRIGGER: this line is opinion/small-talk/a question and should "
            f"NOT be fact-checked, but the prompt extracted {predicted!r}. Tighten "
            "the trigger so it ignores this kind of statement.",
        )

    if not triggered:
        return (
            0.0,
            "MISSED CLAIM: this line contains the verifiable fact(s) "
            f"{expected!r} but nothing was extracted. Loosen the trigger to catch "
            "specific, checkable facts like this.",
        )

    # Both expected and predicted are non-empty: measure coverage via overlap.
    matched_exp = sum(1 for e in expected if any(_overlap(e, p) >= 0.3 for p in predicted))
    matched_pred = sum(1 for p in predicted if any(_overlap(e, p) >= 0.3 for e in expected))
    recall = matched_exp / len(expected)
    precision = matched_pred / len(predicted)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    notes = []
    if recall < 1.0:
        notes.append(f"missed some expected claims (recall {recall:.2f}, expected {expected!r})")
    if precision < 1.0:
        notes.append(f"extracted some off-target claims (precision {precision:.2f})")
    feedback = "Correctly triggered." if not notes else "Triggered but " + "; ".join(notes) + "."
    return f1, feedback


# ---------------------------------------------------------------------------
# GEPA optimizer
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A prompt candidate and its per-example scores on the train set."""

    prompt: str
    scores: list[float] = field(default_factory=list)

    @property
    def avg(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


class GEPA:
    """A compact, faithful implementation of GEPA for a single prompt component."""

    def __init__(
        self,
        *,
        task_fn: TaskFn,
        reflect_fn: ReflectFn,
        trainset: list[dict],
        budget: int = 60,
        minibatch: int = 3,
        seed: int = 0,
    ):
        """Initialize the optimizer.

        Args:
            task_fn: Runs a prompt on an utterance and returns extracted claims.
            reflect_fn: Proposes an improved prompt from feedback.
            trainset: Labeled examples (see ``dataset.json``).
            budget: Max number of single-example metric evaluations (the cost
                knob -- each eval is one task-model call in real mode).
            minibatch: Examples sampled per reflection round.
            seed: RNG seed for reproducibility.
        """
        self.task_fn = task_fn
        self.reflect_fn = reflect_fn
        self.trainset = trainset
        self.budget = budget
        self.minibatch = minibatch
        self.rng = random.Random(seed)
        self.metric_calls = 0

    def _eval(self, prompt: str, examples: list[dict]) -> tuple[list[float], list[str]]:
        scores, feedback = [], []
        for ex in examples:
            predicted = self.task_fn(prompt, ex["text"])
            s, fb = score_example(predicted, ex)
            scores.append(s)
            feedback.append(fb)
            self.metric_calls += 1
        return scores, feedback

    def _pareto_select(self, pool: list[Candidate]) -> Candidate:
        """Sample a parent from the Pareto frontier.

        Pick a random training example, then among the candidates that achieve
        the best score on it, return the one with the best average. This keeps
        specialists alive instead of always exploiting the average-best prompt.
        """
        i = self.rng.randrange(len(self.trainset))
        best = max(c.scores[i] for c in pool)
        frontier = [c for c in pool if c.scores[i] >= best]
        return max(frontier, key=lambda c: c.avg)

    def optimize(self) -> Candidate:
        """Run the GEPA loop and return the best candidate found."""
        seed_scores, _ = self._eval_seed()
        pool = [Candidate(prompt=self._seed_prompt, scores=seed_scores)]
        best = pool[0]
        print(f"[gepa] seed avg score: {best.avg:.3f}")

        rounds = 0
        while self.metric_calls < self.budget:
            rounds += 1
            parent = self._pareto_select(pool)
            batch = self.rng.sample(self.trainset, min(self.minibatch, len(self.trainset)))

            parent_scores, feedbacks = self._eval(parent.prompt, batch)
            parent_avg = sum(parent_scores) / len(parent_scores)

            # Only reflect on the examples we got wrong, with their feedback.
            lessons = [
                f"- Utterance: {ex['text']!r}\n  Result: {fb}"
                for ex, s, fb in zip(batch, parent_scores, feedbacks)
                if s < 1.0
            ]
            if not lessons:
                continue  # nothing to learn from this batch

            child_prompt = self.reflect_fn(parent.prompt, "\n".join(lessons))
            child_scores, _ = self._eval(child_prompt, batch)
            child_avg = sum(child_scores) / len(child_scores)

            if child_avg > parent_avg:
                full_scores, _ = self._eval(child_prompt, self.trainset)
                child = Candidate(prompt=child_prompt, scores=full_scores)
                pool.append(child)
                if child.avg > best.avg:
                    best = child
                    print(
                        f"[gepa] round {rounds}: new best avg {best.avg:.3f} "
                        f"(metric calls {self.metric_calls}/{self.budget})"
                    )

        print(f"[gepa] done. best avg score: {best.avg:.3f}")
        return best

    # Seed prompt is set by the caller via attribute before optimize().
    _seed_prompt: str = ""

    def _eval_seed(self) -> tuple[list[float], list[str]]:
        return self._eval(self._seed_prompt, self.trainset)


# ---------------------------------------------------------------------------
# Real LLM backends (OpenAI)
# ---------------------------------------------------------------------------

REFLECTION_META_PROMPT = """\
You are improving the SYSTEM PROMPT of a real-time fact-checking trigger used on
a live conference transcript. The prompt decides which spoken lines contain a
check-worthy factual claim and writes a search query for each.

Here is the current prompt:
<current_prompt>
{prompt}
</current_prompt>

Here is feedback from running it on several lines (false triggers, missed
claims, over-extraction):
<feedback>
{feedback}
</feedback>

Rewrite the prompt so it fixes these mistakes while staying general. Keep it
concise and keep the required JSON output format exactly:
{{"claims": [{{"claim": "...", "query": "..."}}]}} (or {{"claims": []}}).
Respond with ONLY the new prompt text, no preamble.
"""


def make_openai_backends(task_model: str, reflection_model: str) -> tuple[TaskFn, ReflectFn]:
    """Build (task_fn, reflect_fn) backed by the OpenAI API."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY

    def task_fn(prompt: str, text: str) -> list[str]:
        resp = client.chat.completions.create(
            model=task_model,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            return [str(c.get("claim", "")) for c in data.get("claims", []) if c.get("claim")]
        except Exception:
            return []

    def reflect_fn(prompt: str, feedback: str) -> str:
        resp = client.chat.completions.create(
            model=reflection_model,
            temperature=1.0,
            max_tokens=900,
            messages=[
                {
                    "role": "user",
                    "content": REFLECTION_META_PROMPT.format(prompt=prompt, feedback=feedback),
                }
            ],
        )
        return resp.choices[0].message.content.strip()

    return task_fn, reflect_fn


# ---------------------------------------------------------------------------
# Offline self-test backends (deterministic, no network)
# ---------------------------------------------------------------------------


def make_selftest_backends() -> tuple[TaskFn, ReflectFn, list[dict]]:
    """A toy task where the prompt is a set of enabled trigger rules.

    The "task" fires when the utterance contains a fact cue word AND the prompt
    has enabled the matching rule. Reflection reads the feedback and toggles
    rules, so the GEPA loop provably climbs from a weak seed to a strong prompt
    without any API calls. This exists to verify the optimizer's control flow.
    """
    cues = {"percent": "STAT", "in 19": "DATE", "meters": "MEASURE", "bones": "MEASURE"}

    def enabled_rules(prompt: str) -> set[str]:
        return {r for r in {"STAT", "DATE", "MEASURE"} if f"RULE:{r}" in prompt}

    def task_fn(prompt: str, text: str) -> list[str]:
        rules = enabled_rules(prompt)
        low = text.lower()
        if "i think" in low or "best" in low:  # always-ignore opinion cues
            return []
        for cue, rule in cues.items():
            if cue in low and rule in rules:
                return [text]
        return []

    def reflect_fn(prompt: str, feedback: str) -> str:
        rules = enabled_rules(prompt)
        # Turn on whichever rule the missed examples need.
        for cue, rule in cues.items():
            if "MISSED" in feedback and cue in feedback.lower():
                rules.add(rule)
        return "Trigger on facts.\n" + "\n".join(f"RULE:{r}" for r in sorted(rules))

    trainset = [
        {
            "text": "Revenue grew 40 percent last year.",
            "should_trigger": True,
            "claims": ["Revenue grew 40 percent"],
        },
        {
            "text": "Python was released in 1991.",
            "should_trigger": True,
            "claims": ["Python released in 1991"],
        },
        {
            "text": "Everest is 8849 meters tall.",
            "should_trigger": True,
            "claims": ["Everest is 8849 meters"],
        },
        {"text": "The body has 206 bones.", "should_trigger": True, "claims": ["206 bones"]},
        {"text": "I think this is the best talk ever.", "should_trigger": False, "claims": []},
        {"text": "Let's take a short break now.", "should_trigger": False, "claims": []},
    ]
    return task_fn, reflect_fn, trainset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA-optimize the fact-check trigger prompt")
    parser.add_argument("--budget", type=int, default=60, help="Max metric evals (cost knob)")
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--task-model", default="gpt-4o-mini")
    parser.add_argument("--reflection-model", default="gpt-4o")
    parser.add_argument("--self-test", action="store_true", help="Run offline, no API key needed")
    args = parser.parse_args()

    if args.self_test:
        task_fn, reflect_fn, trainset = make_selftest_backends()
        seed_prompt = "Trigger on facts.\nRULE:STAT"  # deliberately incomplete
    else:
        from prompts import DEFAULT_EXTRACTOR_PROMPT

        trainset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        task_fn, reflect_fn = make_openai_backends(args.task_model, args.reflection_model)
        seed_prompt = DEFAULT_EXTRACTOR_PROMPT

    gepa = GEPA(
        task_fn=task_fn,
        reflect_fn=reflect_fn,
        trainset=trainset,
        budget=args.budget,
        minibatch=args.minibatch,
    )
    gepa._seed_prompt = seed_prompt
    best = gepa.optimize()

    if args.self_test:
        print("\n[self-test] best prompt:\n" + best.prompt)
        assert best.avg >= 0.99, f"self-test did not converge (avg={best.avg:.3f})"
        print("\n[self-test] PASSED: GEPA loop improved the prompt to a perfect score.")
        return

    OUTPUT_PATH.write_text(best.prompt + "\n", encoding="utf-8")
    print(f"\nWrote optimized trigger prompt to {OUTPUT_PATH}")
    print("The bot will use it automatically on the next run.")


if __name__ == "__main__":
    main()
