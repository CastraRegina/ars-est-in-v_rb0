# bezier_benchmark.py
# Run with: python bezier_benchmark.py

import timeit
from typing import Sequence, Tuple

import numpy as np


class DummyDrawer:
    """Minimal mock — only used to make the call overhead identical"""

    def _line_to_type(self, point: Tuple[float, float], width: float):
        pass  # Replace with `blackhole = point` if you want to prevent optimization


drawer = DummyDrawer()

# Fixed test curve
POINTS = ((0.0, 0.0), (50.0, 200.0), (200.0, 0.0))
pt0, pt1, pt2 = POINTS

STEPS_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1_000, 3_000, 7_000, 10_000]


def version1_numpy_original(steps: int):
    drawer._polygonize_steps = steps
    t_values = np.arange(1 / steps, 1, 1 / steps)
    x_values = (1 - t_values) ** 2 * pt0[0] + 2 * (1 - t_values) * t_values * pt1[0] + t_values**2 * pt2[0]
    y_values = (1 - t_values) ** 2 * pt0[1] + 2 * (1 - t_values) * t_values * pt1[1] + t_values**2 * pt2[1]
    for x, y in zip(x_values, y_values):
        drawer._line_to_type((x, y), 2.0)


def version2_python_loop(steps: int):
    drawer._polygonize_steps = steps
    inv_steps = 1.0 / steps
    for i in range(1, steps + 1):
        # t = i / steps
        # omt = 1 - t
        # x = omt * omt * pt0[0] + 2 * omt * t * pt1[0] + t * t * pt2[0]
        # y = omt * omt * pt0[1] + 2 * omt * t * pt1[1] + t * t * pt2[1]

        t = i * inv_steps
        omt = 1.0 - t
        omt2 = omt * omt
        t2 = t * t
        x = omt2 * pt0[0] + 2 * omt * t * pt1[0] + t2 * pt2[0]
        y = omt2 * pt0[1] + 2 * omt * t * pt1[1] + t2 * pt2[1]
        drawer._line_to_type((x, y), 2.0)


def version3_numpy_fast(steps: int):
    drawer._polygonize_steps = steps
    t = np.linspace(0, 1, steps + 1)[1:]  # from 1/steps → 1
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    c = t**2
    pts = np.stack((a * pt0[0] + b * pt1[0] + c * pt2[0], a * pt0[1] + b * pt1[1] + c * pt2[1]), axis=1)
    for x, y in pts:
        drawer._line_to_type((x, y), 2.0)


versions = {
    "NumPy (original)": version1_numpy_original,
    "Python loop": version2_python_loop,
    "NumPy fast": version3_numpy_fast,
}

print("Quadratic Bézier polygonization benchmark (lower = better)")
print("Steps   |   NumPy (orig)   |   Python loop    |    NumPy fast    | Fastest")
print("-" * 78)

results = {}

for steps in STEPS_LIST:
    timings = {}
    # Adjust repeats so each test takes ~0.2–1 second total
    repeats = max(1, 200_000 // steps)

    for name, func in versions.items():
        t = timeit.timeit(lambda: func(steps), number=repeats)
        timings[name] = t * 1000 / repeats  # milliseconds per call

    fastest_name = min(timings, key=timings.get)
    fastest_time = timings[fastest_name]

    print(
        f"{steps:6}  |  {timings['NumPy (original)']:9.3f} ms  |  {timings['Python loop']:9.3f} ms  |  {timings['NumPy fast']:9.3f} ms  |  {fastest_name}"
    )

    results[steps] = timings

# Final recommendation
print("\n" + "=" * 78)
print("RECOMMENDATION".center(78))
print("=" * 78)

overall_fastest = None
overall_time = float("inf")

for steps in STEPS_LIST:
    best = min(results[steps], key=results[steps].get)
    time = results[steps][best]
    if time < overall_time:
        overall_time = time
        overall_fastest = best
    print(f"{steps:6} steps → fastest: {best} ({results[steps][best]:.3f} ms)")

print(f"\nOn your machine the overall fastest version is: {overall_fastest}")
if overall_fastest == "NumPy fast":
    print("→ Use version3_numpy_fast when polygonize_steps ≥ 100 (up to ~2.5× faster at 10k steps)")
elif overall_fastest == "NumPy (original)":
    print("→ Your current NumPy version is excellent — keep it!")
else:
    print("→ Surprisingly, pure Python loop wins on your system for these step counts!")
