"""
Functional smoke-test: execute all code cells except the long-running ones
(Monte Carlo and Sensitivity Analysis), injecting small dummy results so that
downstream cells (figures) still execute correctly.
"""
import json, sys, traceback
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no pop-up windows needed)

nb = json.loads(open('terry_health_economic_simulation.ipynb', encoding='utf-8').read())
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Running {len(code_cells)} code cells (Agg backend — no display windows)\n")

DUMMY_MC = """
import pandas as pd, numpy as np
_rng2 = np.random.default_rng(42)
_rows = [run_single_iteration(params, 50, _rng2) for _ in range(30)]
mc_results = pd.DataFrame(_rows)
valid = mc_results.dropna(subset=['icer'])
print(f"(Smoke-test: dummy mc_results, {len(valid)} valid iters)")
"""

DUMMY_SA = """
import pandas as pd
_sa = [{'parameter': f'Parameter {j}', 'icer_low': 5000+j*500,
        'icer_high': 25000+j*500, 'spread': 20000, 'base_icer': 15000}
       for j in range(10)]
sa_df = pd.DataFrame(_sa).sort_values('spread', ascending=False)
base_icer_val = 15000.0
print("(Smoke-test: dummy sa_df for tornado diagram)")
"""

# Identify which cells to skip (by checking for the long-running function calls)
globs = {}
errors = []

for i, cell in enumerate(code_cells):
    src = cell['source']

    # Replace the two long-running cells with short dummy versions
    if 'run_monte_carlo(' in src and 'def run_monte_carlo' not in src:
        print(f"Cell {i:2d}: [REPLACED with dummy Monte Carlo — 30 iters]")
        src = DUMMY_MC
    elif 'run_sensitivity_analysis(' in src and 'def run_sensitivity_analysis' not in src:
        print(f"Cell {i:2d}: [REPLACED with dummy DSA results]")
        src = DUMMY_SA
    else:
        print(f"Cell {i:2d}: Running...", end=' ')

    try:
        exec(compile(src, f'<cell_{i}>', 'exec'), globs)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        errors.append((i, e))

print()
if errors:
    print(f"FAILED: {len(errors)} cell(s) raised errors.")
    for i, e in errors:
        print(f"  Cell {i}: {e}")
    sys.exit(1)
else:
    print("All cells executed successfully.")
