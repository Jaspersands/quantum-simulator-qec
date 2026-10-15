# Ported from fusion_blossom 0.2.13's scripts/demo.py and scripts/graph_time_partition.py
# (yuewuo/fusion-blossom; Yue Wu, MIT; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec:
# fusion_blossom is stabilizer_qec.fusion_blossom, each script is a test, the visualizer file
# goes to a temporary folder, and the browser is not opened.
import os

import stabilizer_qec.fusion_blossom as fb

EXAMPLE_SYNDROME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_time_partition.syndromes")


def test_demo(tmp_path):
    # create an example code
    code = fb.CodeCapacityPlanarCode(d=11, p=0.05, max_half_weight=500)
    initializer = code.get_initializer()  # the decoding graph structure (you can easily construct your own)
    positions = code.get_positions()  # the positions of vertices in the 3D visualizer, optional

    # randomly generate a syndrome according to the noise model
    syndrome = code.generate_random_errors(seed=1000)
    with fb.PyMut(syndrome, "defect_vertices") as defect_vertices:
        defect_vertices.append(0)  # you can modify the defect vertices
    assert syndrome.defect_vertices[-1] == 0

    visualize_filename = fb.static_visualize_data_filename()
    visualizer = fb.Visualizer(filepath=str(tmp_path / visualize_filename), positions=positions)

    solver = fb.SolverSerial(initializer)
    solver.solve(syndrome, visualizer)  # enable visualizer for debugging
    perfect_matching = solver.perfect_matching()
    matched = sorted([a for pair in perfect_matching.peer_matchings for a in pair] + [a for a, _ in perfect_matching.virtual_matchings])
    assert matched == list(range(len(syndrome.defect_vertices)))
    subgraph = solver.subgraph(visualizer)
    # the subgraph's syndrome is the defects
    flips = {}
    for edge_index in subgraph:
        for v in initializer.weighted_edges[edge_index][:2]:
            flips[v] = flips.get(v, 0) ^ 1
    real = {v for v, f in flips.items() if f and v not in set(initializer.virtual_vertices)}
    assert real == set(syndrome.defect_vertices)
    solver.clear()  # clear is O(1) complexity, recommended for repetitive simulation
    assert len(visualizer.snapshots) > 0


def graph_time_partition(initializer: fb.SolverInitializer, positions) -> fb.PartitionConfig:
    assert len(positions) > 0
    partition = fb.PartitionConfig(initializer.vertex_num)
    # first check if the time is monotonically increasing
    last_t = positions[0].t
    t_list = [last_t]
    for position in positions:
        assert position.t >= last_t, "t not monotonically increasing, vertex reordering must be performed before calling this"
        if position.t != last_t:
            t_list.append(position.t)
        last_t = position.t
    # pick the t value in the middle to split it
    t_split = t_list[len(t_list) // 2]
    # find the vertices indices
    split_start_index = None
    split_end_index = None
    for vertex_index, position in enumerate(positions):
        if split_start_index is None and position.t == t_split:
            split_start_index = vertex_index
        if position.t == t_split:
            split_end_index = vertex_index + 1
    assert split_start_index is not None
    # partitions are found
    partition.partitions = [
        fb.VertexRange(0, split_start_index),
        fb.VertexRange(split_end_index, len(positions)),
    ]
    partition.fusions = [(0, 1)]
    return partition


def test_graph_time_partition():
    reader = fb.ErrorPatternReader(filename=EXAMPLE_SYNDROME_FILE)
    initializer, positions = reader.get_initializer(), reader.get_positions()
    partition = graph_time_partition(initializer, positions)
    assert partition.vertex_num == 216
    assert partition.fusions == [(0, 1)]
    assert fb.PartitionConfig.from_json(partition.to_json()).to_json() == partition.to_json()
    # the partitioned solvers agree with the serial one on the file's syndromes
    info = partition.info()
    serial = fb.SolverSerial(initializer)
    parallel = fb.SolverParallel(initializer, info, {})
    for syndrome in reader.syndrome_patterns:
        serial.solve(syndrome)
        parallel.solve(syndrome)
        assert serial.sum_dual_variables() == parallel.sum_dual_variables()
        weight = lambda s: sum(initializer.weighted_edges[e][2] for e in s)  # noqa: E731
        assert weight(serial.subgraph()) == weight(parallel.subgraph())
        serial.clear()
        parallel.clear()
