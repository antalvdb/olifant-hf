"""
Mock TiMBL classifier for demonstration purposes.

This mock implementation shows that the HuggingFace integration architecture
works correctly. Replace this with real TiMBL backend for production use.
"""

import random
from typing import Tuple


class MockTimblClassifier:
    """
    Mock TiMBL classifier that returns plausible predictions.

    This is for demonstration only. In production, use:
    - python3-timbl Python bindings
    - Working TiMBL CLI wrapper
    - TiMBL server client
    """

    def __init__(self, fileprefix: str, timbloptions: str, format: str = "Tabbed"):
        self.fileprefix = fileprefix
        self.timbloptions = timbloptions
        self.format = format
        self.ibase_loaded = False

        # Common English tokens for mock predictions
        self.common_tokens = [
            "the", ".", "and", "to", "of", "a", "in", "that", "is", "for",
            "it", "with", "as", "was", "on", "be", "by", "at", "from", "this"
        ]

    def load(self):
        """Simulate loading an instance base."""
        print(f"[MOCK] Simulating load of {self.fileprefix}.ibase")
        self.ibase_loaded = True

    def classify(self, features: list, allowtopdistribution: bool = True) -> Tuple[str, str, float]:
        """
        Mock classification that returns plausible results.

        Returns:
            (predicted_token, distribution_string, distance)
        """
        if not self.ibase_loaded:
            raise RuntimeError("Instance base not loaded")

        # Choose a plausible prediction based on context
        # This is completely fake but demonstrates the interface

        # Look at the last context token
        last_token = features[-1] if features and features[-1] != '_' else None

        # Simple heuristic for more realistic mock predictions
        if last_token and last_token.lower() in ['the', 'a', 'an']:
            # After article, predict a noun-ish token
            candidates = ['man', 'woman', 'house', 'cat', 'dog', 'book', 'world']
        elif last_token and last_token in ['.', '!', '?']:
            # After punctuation, predict capital word
            candidates = ['The', 'I', 'He', 'She', 'It', 'We', 'They']
        else:
            candidates = self.common_tokens

        # Pick top prediction
        predicted = random.choice(candidates)

        # Generate mock distribution (would come from k-NN in real TiMBL)
        num_neighbors = random.randint(3, 7)
        distribution_parts = []

        # Top prediction gets highest score
        distribution_parts.append(f"{predicted} 0.{random.randint(40, 70)}")

        # Add some runner-ups
        for _ in range(num_neighbors - 1):
            token = random.choice([t for t in candidates if t != predicted])
            score = f"0.{random.randint(5, 30):02d}"
            distribution_parts.append(f"{token} {score}")

        distribution = ", ".join(distribution_parts)

        # Distance (0 = exact match, 1 = very different)
        distance = random.uniform(0.1, 0.6)

        return (predicted, distribution, distance)

    def append(self, features: list, classlabel: str):
        """
        Simulate appending an instance.

        Note: In production, this would add to the instance base.
        For mock, we just log it.
        """
        print(f"[MOCK] Would append: {features} -> {classlabel}")
