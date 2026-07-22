import textwrap

from planner.sim_evaluator import _parse_energy_from_stdout, parse_metrics_csv

# Mimics the real output/*.csv header; times in ns, ITL is a stringified list.
_MOCK_CSV = textwrap.dedent(
    """\
    instance id,request id,model,input,output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL
    0,0,m,10,4,0,4000000000,4000000000,0,1000000000,500000000,"[500000000, 500000000, 500000000]"
    0,1,m,10,3,0,3000000000,3000000000,0,2000000000,400000000,"[400000000, 400000000]"
    """
)


def test_parse_basic_aggregation(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text(_MOCK_CSV)
    m = parse_metrics_csv(p)

    # TTFT mean = (1e9 + 2e9)/2 ns = 1.5e9 ns = 1500 ms
    assert abs(m.ttft_ms - 1500.0) < 1e-6
    # TPOT mean = (5e8 + 4e8)/2 = 4.5e8 ns = 450 ms
    assert abs(m.tpot_ms - 450.0) < 1e-6
    assert m.num_requests == 2


def test_itl_p99_parsed_from_list(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text(_MOCK_CSV)
    m = parse_metrics_csv(p)
    # all ITL values are 4e8 or 5e8 ns -> p99 close to 5e8 ns = 500 ms
    assert 400.0 <= m.itl_p99_ms <= 500.0


def test_throughput_positive(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text(_MOCK_CSV)
    m = parse_metrics_csv(p)
    # total output = 7 tokens over 4s span -> 1.75 tok/s
    assert abs(m.throughput_toks_s - 1.75) < 1e-6


def test_energy_parsed_from_stdout_kj():
    stdout = (
        "some log line\n"
        "Total energy consumption (kJ):                    12.50\n"
        "more logs\n"
    )
    # 12.50 kJ -> 12500 J
    assert _parse_energy_from_stdout(stdout) == 12500.0


def test_energy_absent_returns_none():
    assert _parse_energy_from_stdout("no power modeling here\n") is None


def test_empty_csv_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("instance id,request id,model,input,output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL\n")
    try:
        parse_metrics_csv(p)
        assert False, "expected ValueError"
    except ValueError:
        pass
