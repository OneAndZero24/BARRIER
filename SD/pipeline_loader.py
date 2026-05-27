from typing import Callable, Dict

from SD.barrier_adapter import load_barrier_pipeline


# Minimal method registry used by experiment runners.
# Only checkpoint/pipeline loading is method-specific.
METHODS: Dict[str, Callable] = {
    "esd": load_barrier_pipeline,
    "uce": load_barrier_pipeline,
    "concept-ablation": load_barrier_pipeline,
    "barrier": load_barrier_pipeline,
}


def get_method_loader(method: str) -> Callable:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(METHODS.keys())}")
    return METHODS[method]
