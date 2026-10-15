# MWPF 0.2.12's tests/python/test_visualize.py, run against stabilizer_qec.mwpf: imports changed, and files
# written to a temporary folder.
from mwpf_common import *
import os
import sys


# code = mwpf.CodeCapacityRepetitionCode(d=3, p=0.01)
# code = mwpf.CodeCapacityPlanarCode(d=3, p=0.01)
code = mwpf.CodeCapacityTailoredCode(d=5, pxy=0.001, pz=0.1)
# code = mwpf.CodeCapacityColorCode(d=3, p=0.01)

# either randomly generate errors
# code.generate_random_errors()

# or manually set errors
# code.set_physical_errors([1, 2, 3, 4, 5, 6])

# or manually set defect
code.set_defect_vertices([10, 11, 16, 17])

initializer = code.get_initializer()

# solver configuration
config = {"cluster_node_limit": 100, "timeout": 10.0}
solver = mwpf.SolverSerialJointSingleHair(initializer, config)

"""
run the solver
"""

visualizer = mwpf.Visualizer(positions=code.get_positions())
visualizer.snapshot("syndrome", code)

syndrome = code.get_syndrome()
print(f"syndrome: {syndrome}")
solver.solve(syndrome, visualizer)
subgraph, bound = solver.subgraph_range(visualizer)

print(f"subgraph: {subgraph}")
print(f"bound: {(bound.lower, bound.upper)}")
if bound.lower == bound.upper:
    print("[success] optimal! 🤩")
else:
    print("[potential failure] may be suboptimal... 😅")

visualizer.save_html(
    os.path.join(OUT_DIR, f"python_test_visualize.html")
)
