# New Entity Adaptation Experiment - 2026-07-27

## Scope

One paired seed (`20260727`) was completed with the fixed test set: 73 new-entity mentions, 585 old-entity mentions, and 152 no-entity utterances. A uses the holdout model and old bank, B uses the same model and the full 230-surface bank, and C uses incremental SS/GL checkpoints and the full bank.

## SS exhaustive retrieval

| Metric | A old bank | B cold full bank | C incremental |
| --- | ---: | ---: | ---: |
| New Recall@1 | 0.00% | 69.86% | 73.97% |
| New Recall@5 | 0.00% | 97.26% | 98.63% |
| Old Recall@1 | 82.91% | 82.56% | 82.56% |
| Old Recall@5 | 99.83% | 99.83% | 100.00% |
| No-entity SS false positive | 43.42% | 44.74% | 57.24% |
| P50 latency | 120.0 ms | 131.8 ms | 132.1 ms |
| P95 latency | 311.2 ms | 145.4 ms | 154.8 ms |
| Peak allocated GPU memory | 182.6 MB | 198.4 MB | 198.4 MB |

Cold insertion already exceeds the 95% new Recall@5 rule. Incremental SS adds only 1.37 percentage points, below the 1-point target by a small margin but with a 12.50-point increase in no-entity SS false positives relative to B. The C SS checkpoint must not be released without threshold recalibration and end-to-end safety validation.

## GL oracle

| Metric | A cold model | C incremental | Difference |
| --- | ---: | ---: | ---: |
| New replace precision | 93.33% | 81.13% | -12.20 pp |
| New replace recall | 59.15% | 60.56% | +1.41 pp |
| New replace F1 | 72.41% | 69.35% | -3.06 pp |
| Old replace precision | 100.00% | 85.71% | -14.29 pp |
| Old replace recall | 94.74% | 94.74% | 0.00 pp |
| Old replace F1 | 97.30% | 90.00% | -7.30 pp |

The incremental GL checkpoint fails the acceptance rules: the new-entity gain is below 5 points and both new and old precision regress substantially.

## Decision

New entities can be inserted without SS or GL retraining at the current 230-surface scale. Use the B configuration: holdout model plus full frame-feature bank. Reject both C checkpoints from this run. Batch retraining should only be reconsidered after collecting real GL false-negative examples and must preserve old-entity replay and no-entity negatives more aggressively.

This is a one-seed result. The remaining two paired seeds and the final end-to-end ASR -> SS -> GL run are still required before treating standard deviations and harmful-edit rates as final acceptance evidence.
