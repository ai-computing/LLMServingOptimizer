from planner.graph_model import (
    build_graph,
    default_link_params,
    device_inventory,
    parse_bandwidth_gbps_bytes,
    parse_latency_ns,
)


def test_parse_bandwidth_bits_vs_bytes():
    # 200 Gbps (bits) -> 25 GB/s
    assert parse_bandwidth_gbps_bytes("200Gbps") == 25.0
    # 600 GBps (bytes) -> 600 GB/s
    assert parse_bandwidth_gbps_bytes("600GBps") == 600.0
    assert parse_bandwidth_gbps_bytes("1TBps") == 1000.0


def test_parse_latency_units():
    assert parse_latency_ns("0.0005ms") == 500.0
    assert parse_latency_ns("100ns") == 100.0
    assert parse_latency_ns("1us") == 1000.0


def test_build_graph_nodes_and_inventory(spec):
    g = build_graph(spec)
    assert set(g.nodes) == {"node0"}
    inv = device_inventory(g)
    hw = {d.hardware: d.count for d in inv}
    assert hw == {"H100": 2, "A6000": 4}


def test_default_link_params(spec):
    bw, lat = default_link_params(spec)
    assert bw == 25.0 and lat == 500.0
