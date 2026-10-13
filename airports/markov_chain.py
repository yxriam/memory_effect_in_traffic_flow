"""Variable-order Markov chain over discrete states (airports or cities).

`MarkovChain(order)` counts transitions from an `order`-tuple of previous states to the next
state. `transition()` samples the next state (with a small exploration probability `alpha`
so that unseen contexts still produce a state), `most_likely()` returns the arg-max.
"""
import random
from collections import defaultdict


class MarkovChain:
    def __init__(self, order, alpha=0.1, seed=None):
        self.order = order
        self.alpha = alpha                      # probability of ignoring the context (exploration)
        self.transitions = {}                   # context tuple -> {next_state: count}
        self.state_counts = defaultdict(int)    # marginal state frequencies (fallback distribution)
        self._probabilities = None              # cache, invalidated by add_transition
        self._rng = random.Random(seed)

    # ---- fitting -------------------------------------------------------------------------
    def add_transition(self, *states):
        if len(states) != self.order + 1:
            raise ValueError(f"Expected {self.order + 1} states, got {len(states)}")
        for s in states:
            self.state_counts[s] += 1
        context, nxt = states[:-1], states[-1]
        self.transitions.setdefault(context, {})
        self.transitions[context][nxt] = self.transitions[context].get(nxt, 0) + 1
        self._probabilities = None

    def fit_sequence(self, sequence):
        """Add every length-(order+1) window of `sequence`."""
        k = self.order
        for i in range(k, len(sequence)):
            self.add_transition(*sequence[i - k:i + 1])

    # ---- querying ------------------------------------------------------------------------
    def get_probabilities(self):
        if self._probabilities is None:
            self._probabilities = {
                ctx: {s: c / sum(nxt.values()) for s, c in nxt.items()}
                for ctx, nxt in self.transitions.items()
            }
        return self._probabilities

    def _check(self, context):
        context = tuple(context)
        if len(context) != self.order:
            raise ValueError(f"Expected {self.order} states, got {len(context)}")
        return context

    def transition(self, context):
        """Sample the next state given `context` (a sequence of `order` states)."""
        probs = self.get_probabilities().get(self._check(context), {})
        if not probs or self._rng.random() < self.alpha:
            states, weights = zip(*self.state_counts.items())
            return self._rng.choices(states, weights=weights)[0]
        return self._rng.choices(list(probs), weights=list(probs.values()))[0]

    def most_likely(self, context):
        """Deterministic arg-max prediction; None if the context was never seen."""
        probs = self.get_probabilities().get(self._check(context))
        return max(probs, key=probs.get) if probs else None

    def predict(self, num_predictions, start=None):
        """Roll the chain forward `num_predictions` steps from `start` (default: most frequent context)."""
        current = tuple(start) if start is not None else self._most_frequent_context()
        out = []
        for _ in range(num_predictions):
            nxt = self.transition(current)
            out.append(nxt)
            current = current[1:] + (nxt,)
        return out

    def _most_frequent_context(self):
        return max(self.transitions, key=lambda ctx: sum(self.transitions[ctx].values()))
