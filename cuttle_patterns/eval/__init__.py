"""Embedding evaluation harness.

Scores any per-frame embedding (existing BEAST checkpoints, future pretrained
backbones) on how well it organizes by skin-pattern type vs. by video/individual
identity. See `cuttle_patterns/eval/README.md` for how to run it and
`docs/eval_plan.md` for the full design rationale.
"""
