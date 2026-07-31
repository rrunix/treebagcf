"""Experiment harness for the calf counterfactual benchmarks.

Config-driven, parallel, and resumable: enumerate (dataset, rf_variant, query,
method) tasks, run each under a per-method wall-clock cap, store one JSON shard
per task (the resume unit), and fold the shards into a tidy parquet/CSV on
demand for the paper tables.

Modules:
- ``store``   — task_id hashing, atomic shard read/write, resume set, aggregation.
- ``methods`` — task spec -> :class:`baselines.base.CFResult` (time-cap semantics).
- ``runner``  — config parsing, task enumeration, parallel execution, hard-kill.
"""
