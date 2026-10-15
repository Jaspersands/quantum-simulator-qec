# Ported from fusion_blossom 0.2.13 (yuewuo/fusion-blossom; Yue Wu, MIT; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: fusion_blossom is stabilizer_qec.fusion_blossom, and the visualizer files go to a temporary folder.
import os
import tempfile
import stabilizer_qec.fusion_blossom as fb


data_folder = tempfile.mkdtemp(prefix="fusion_blossom_")


def solver_tester(max_tree_size=None) -> fb.SolverSerial:
    # construct solver
    code = fb.CodeCapacityPlanarCode(d=11, p=0.05, max_half_weight=500)
    positions = code.get_positions()
    initializer = code.get_initializer()
    solver = fb.SolverSerial(initializer, max_tree_size=max_tree_size)
    # construct visualizer
    filename = f"test_max_tree_size_{max_tree_size}.json"
    visualizer = fb.Visualizer(
        filepath=os.path.join(data_folder, filename), positions=positions)
    # decode
    syndrome = fb.SyndromePattern([39, 52, 63, 90, 100])
    solver.solve(syndrome, visualizer)
    solver.subgraph(visualizer)


def test_union_find_decoder():
    # http://localhost:8066/?filename=test_max_tree_size_0.json
    solver_tester(max_tree_size=0)


def test_mwpm_decoder():
    # http://localhost:8066/?filename=test_max_tree_size_None.json
    solver_tester()


def test_mixture_decoder():
    # http://localhost:8066/?filename=test_max_tree_size_10.json
    solver_tester(max_tree_size=10)
