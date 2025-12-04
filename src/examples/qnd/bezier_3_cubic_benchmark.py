# cubic_bezier_benchmark.py
# Run with: python cubic_bezier_benchmark.py

import timeit
from typing import Sequence, Tuple

import numpy as np


class DummyDrawer:
    def _line_to_type(self, point: Tuple[float, float], width: float):
        pass  # Minimal overhead, prevents dead-code elimination


drawer = DummyDrawer()

# Test cubic Bezier: a nice "S" shape
POINTS = ((0.0, 0.0), (100.0, 200.0), (0.0, -200.0), (200.0, 0.0))
pt0, pt1, pt2, pt3 = POINTS

STEPS_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1_000, 3_000, 7_000, 10_000]


# Version 1: Your current pure Python loop
def python_loop(steps: int):
    drawer._polygonize_steps = steps
    inv = 1.0 / steps
    for i in range(1, steps + 1):
        t = i * inv
        t2 = t * t
        t3 = t2 * t
        omt = 1.0 - t
        omt2 = omt * omt
        omt3 = omt2 * omt

        x = omt3 * pt0[0] + 3 * omt2 * t * pt1[0] + 3 * omt * t2 * pt2[0] + t3 * pt3[0]
        y = omt3 * pt0[1] + 3 * omt2 * t * pt1[1] + 3 * omt * t2 * pt2[1] + t3 * pt3[1]
        drawer._line_to_type((x, y), 3.0)


# Version 2: NumPy vectorized (style that won for quadratic)
def numpy_original(steps: int):
    drawer._polygonize_steps = steps
    t = np.arange(1, steps + 1) / steps
    omt = 1.0 - t
    omt2 = omt * omt
    omt3 = omt2 * omt
    t2 = t * t
    t3 = t2 * t

    x = omt3 * pt0[0] + 3 * omt2 * t * pt1[0] + 3 * omt * t2 * pt2[0] + t3 * pt3[0]
    y = omt3 * pt0[1] + 3 * omt2 * t * pt1[1] + 3 * omt * t2 * pt2[1] + t3 * pt3[1]

    for x_val, y_val in zip(x, y):
        drawer._line_to_type((x_val, y_val), 3.0)


# Version 3: Ultra-optimized NumPy (Bernstein form + minimal temps)
def numpy_fast(steps: int):
    drawer._polygonize_steps = steps
    t = np.linspace(0, 1, steps + 1)[1:]  # 1/steps -> 1.0
    omt = 1.0 - t

    # Bernstein polynomial weights (very cache-friendly)
    b0 = omt**3
    b1 = 3 * omt**2 * t
    b2 = 3 * omt * t**2
    b3 = t**3

    x = b0 * pt0[0] + b1 * pt1[0] + b2 * pt2[0] + b3 * pt3[0]
    y = b0 * pt0[1] + b1 * pt1[1] + b2 * pt2[1] + b3 * pt3[1]

    for x_val, y_val in zip(x, y):
        drawer._line_to_type((x_val, y_val), 3.0)


versions = {
    "Python loop": python_loop,
    "NumPy (original)": numpy_original,
    "NumPy fast": numpy_fast,
}

print("Cubic Bezier Polygonization Benchmark (lower = better)")
print("Steps   |   Python loop    |  NumPy (orig)    |   NumPy fast     | Fastest")
print("-" * 82)

results = {}

for steps in STEPS_LIST:
    timings = {}
    # Auto-scale repeats so each test takes reasonable time
    repeats = max(1, 150_000 // steps)

    for name, func in versions.items():
        dt = timeit.timeit(lambda: func(steps), number=repeats)
        timings[name] = dt * 1000 / repeats  # ms per call

    fastest = min(timings, key=timings.get)
    results[steps] = timings

    print(
        f"{steps:6}  |  {timings['Python loop']:9.3f} ms  |  {timings['NumPy (original)']:9.3f} ms  |  {timings['NumPy fast']:9.3f} ms  |  {fastest}"
    )

# Final recommendation
print("\n" + "=" * 82)
print("RECOMMENDATION FOR CUBIC BEZIER".center(82))
print("=" * 82)

crossover = None
for steps in sorted(results.keys()):
    best = min(results[steps], key=results[steps].get)
    time = results[steps][best]
    print(f"{steps:6} steps -> fastest: {best.ljust(18)} ({time:6.3f} ms)")

    if crossover is None and best.startswith("NumPy"):
        crossover = steps

if crossover:
    print("\nSummary:")
    print(f"-> Use pure Python loop when steps < {crossover}")
    print(f"-> Use NumPy (either version) when steps >= {crossover}")
else:
    print("\nPure Python loop wins at all tested step counts!")

print("\nSuggested adaptive implementation:")
print(
    """
def _polygonize_cubic_bezier(self, points):
    pt0, pt1, pt2, pt3 = points
    steps = self._polygonize_steps

    if steps < 120:  # Adjust this threshold based on your results!
        # Fast Python path
        inv = 1.0 / steps
        for i in range(1, steps + 1):
            t = i * inv
            omt = 1.0 - t
            omt2 = omt * omt
            omt3 = omt2 * omt
            t2 = t * t
            t3 = t2 * t
            x = omt3*pt0[0] + 3*omt2*t*pt1[0] + 3*omt*t2*pt2[0] + t3*pt3[0]
            y = omt3*pt0[1] + 3*omt2*t*pt1[1] + 3*omt*t2*pt2[1] + t3*pt3[1]
            self._line_to_type((x, y), 3.0)
    else:
        # NumPy path (use the winner from your benchmark)
        t = np.arange(1, steps + 1) / steps
        omt = 1.0 - t
        x = omt**3*pt0[0] + 3*omt**2*t*pt1[0] + 3*omt*t**2*pt2[0] + t**3*pt3[0]
        y = omt**3*pt0[1] + 3*omt**2*t*pt1[1] + 3*omt*t**2*pt2[1] + t**3*pt3[1]
        for x_val, y_val in zip(x, y):
            self._line_to_type((x_val, y_val), 3.0)
"""
)
