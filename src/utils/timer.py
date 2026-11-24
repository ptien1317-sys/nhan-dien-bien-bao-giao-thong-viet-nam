"""Do thoi gian va thong ke - cung dinh dang voi its_utils.hpp (C++)."""
import math
import time


class Timer:
    """Dong ho ms dung perf_counter. Dung: t = Timer(); ...; t.elapsed_ms()."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *exc):
        self.ms = self.elapsed_ms()
        return False


def _percentile(sorted_v, p):
    if not sorted_v:
        return 0.0
    idx = max(1, min(len(sorted_v), math.ceil(p / 100.0 * len(sorted_v)))) - 1
    return sorted_v[idx]


def summarize(values) -> dict:
    """mean/std/min/p50/p95/p99/max (ms). Cong thuc percentile trung voi nhanh C++."""
    v = sorted(float(x) for x in values)
    if not v:
        return {k: 0.0 for k in ("mean", "std", "min", "p50", "p95", "p99", "max")}
    mean = sum(v) / len(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
    return {"mean": mean, "std": std, "min": v[0], "p50": _percentile(v, 50),
            "p95": _percentile(v, 95), "p99": _percentile(v, 99), "max": v[-1]}
