"""
Compile-check every code cell in the notebook to catch syntax errors.
Does NOT execute — just parses each cell as Python code.
"""
import json, ast, sys

nb = json.loads(open('terry_health_economic_simulation.ipynb', encoding='utf-8').read())
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Checking {len(code_cells)} code cells for syntax errors...\n")

errors = []
for i, cell in enumerate(code_cells):
    src = cell['source']
    try:
        ast.parse(src)
        print(f"  Cell {i:2d}: OK")
    except SyntaxError as e:
        errors.append((i, e))
        print(f"  Cell {i:2d}: SYNTAX ERROR — line {e.lineno}: {e.msg}")
        # Show the offending line
        lines = src.split('\n')
        if e.lineno and e.lineno <= len(lines):
            print(f"           >>> {lines[e.lineno-1]!r}")

print()
if errors:
    print(f"FAILED: {len(errors)} cell(s) have syntax errors.")
    sys.exit(1)
else:
    print("All cells passed syntax check.")
