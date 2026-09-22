"""Print a transparent toy audit; never label a model compliant from these metrics."""

import json
from pearlmind.evaluation import FairnessAuditor

if __name__ == "__main__":
    print(
        json.dumps(
            FairnessAuditor().audit([0, 1, 0, 1], [0, 1, 1, 1], ["A", "A", "B", "B"]), indent=2
        )
    )
