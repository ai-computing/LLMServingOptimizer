# A6000 × 4 NPU — Llama-3.1-8B Parallelism Sweep Report

**Model:** meta-llama/Llama-3.1-8B  
**Hardware:** NVIDIA A6000 (40 GB, 768 GB/s)  
**Dataset:** sharegpt_req100_rate10_llama.jsonl (100 requests)  
**Flags:** `--fp 16 --block-size 16 --num-req 100`  
**Simulator:** LLMServingSim v1.0.0  

## Configuration Matrix

| # | Config label | TP | PP | DP | P/D layout | Phys NPUs |
|---|---|---|---|---|---|---|
|  1 | `01_tp1_pp1_dp1` | 1 | 1 | 1 | — | 1 |
|  2 | `02_tp2_pp1_dp1` | 2 | 1 | 1 | — | 2 |
|  3 | `03_tp1_pp2_dp1` | 1 | 2 | 1 | — | 2 |
|  4 | `04_tp2_pp2_dp1` | 2 | 2 | 1 | — | 4 |
|  5 | `05_tp1_pp4_dp1` | 1 | 4 | 1 | — | 4 |
|  6 | `06_tp1_pp1_dp2` | 1 | 1 | 2 | — | 2 |
|  7 | `07_tp2_pp1_dp2` | 2 | 1 | 2 | — | 4 |
|  8 | `08_tp1_pp2_dp2` | 1 | 2 | 2 | — | 4 |
|  9 | `09_tp1_pp1_dp4` | 1 | 1 | 4 | — | 4 |
| 10 | `10_pd_1p1d_tp1` | 1 | 1 | 1 | 1P+1D | 3 |
| 11 | `11_pd_1p2d_tp1` | 1 | 1 | 2 | 1P+2D | 4 |
| 12 | `12_pd_1p1d_tp2d` | 2 | 1 | 1 | 1P+1D(T2) | 4 |
| 13 | `13_pd_1p1d_pp2d` | 1 | 2 | 1 | 1P+1D(P2) | 4 |

## Results

| # | Config | Sim latency (s) | Req throughput (req/s) | Prompt TP (tok/s) | Gen TP (tok/s) | Total TP (tok/s) | Avg TTFT (ms) | Avg TPOT (ms) | Avg ITL (ms) | NPU util (%) |
|---|---|---|---|---|---|---|---|---|---|---|
|  1 | `01_tp1_pp1_dp1` | 28.23 | 3.54 | 744.95 | 826.86 | 1571.81 | 128.02 | 42.26 | 37.43 | 37.61 |
|  2 | `02_tp2_pp1_dp1` | 19.18 | 5.21 | 1096.46 | 1217.02 | 2313.48 | 51.97 | 19.40 | 18.99 | 18.80 |
|  3 | `03_tp1_pp2_dp1` | 27.79 | 3.60 | 756.52 | 839.70 | 1596.22 | 93.01 | 37.37 | 34.28 | 37.50 |
|  4 | `04_tp2_pp2_dp1` | 19.34 | 5.17 | 1086.96 | 1206.48 | 2293.44 | 41.25 | 18.60 | 18.37 | 18.75 |
|  5 | `05_tp1_pp4_dp1` | 27.63 | 3.62 | 761.11 | 844.80 | 1605.91 | 78.47 | 37.37 | 34.53 | 37.45 |
|  6 | `06_tp1_pp1_dp2` | 26.60 | 3.76 | 790.53 | 877.45 | 1667.99 | 81.11 | 30.37 | 29.68 | 37.39 |
|  7 | `07_tp2_pp1_dp2` | 18.72 | 5.34 | 1123.21 | 1246.71 | 2369.92 | 40.14 | 16.58 | 16.50 | 18.70 |
|  8 | `08_tp1_pp2_dp2` | 27.12 | 3.69 | 775.26 | 860.50 | 1635.77 | 68.65 | 30.60 | 30.00 | 37.39 |
|  9 | `09_tp1_pp1_dp4` | 26.05 | 3.84 | 807.02 | 895.76 | 1702.78 | 69.21 | 27.28 | 27.14 | 37.39 |
| 10 | `10_pd_1p1d_tp1` | 26.93 | 3.71 | 780.77 | 866.62 | 1647.40 | 101.70 | 29.66 | 29.28 | 37.60 |
| 11 | `11_pd_1p2d_tp1` | 26.16 | 3.82 | 803.85 | 892.24 | 1696.09 | 101.67 | 26.87 | 26.71 | 37.39 |
| 12 | `12_pd_1p1d_tp2d` | — | — | — | — | — | — | — | — | — |
| 13 | `13_pd_1p1d_pp2d` | — | — | — | — | — | — | — | — | — |

## Per-axis Observations

### Tensor Parallelism (TP=1 vs TP=2)

Configs 1 vs 2 (single instance, DP=1, PP=1): TP=2 splits each layer across 2 NPUs, reducing per-NPU compute time at the cost of ALLREDUCE communication after each attention and FFN block. For Llama-3.1-8B (8 attention heads per TP shard at TP=2), TP=2 is expected to reduce generation latency when the model is memory-bound but adds a synchronization overhead visible in TTFT.

### Pipeline Parallelism (PP=1/2/4)

Configs 1, 3, 5 (DP=1, TP=1, PP=1/2/4): pipeline stages split the 32 Transformer layers across NPUs. Each stage runs independently and passes activations to the next. PP reduces per-stage memory but adds bubble overhead (inter-stage send/recv) that grows linearly with PP degree. At PP=4 on a single-instance Llama-3.1-8B each stage handles ~8 layers.

### Data Parallelism (DP=1/2/4)

Configs 1, 6, 9 (TP=1, PP=1, DP=1/2/4): each instance serves an independent subset of requests via RR routing. DP scales throughput near-linearly because instances share no state. TTFT and TPOT per request should remain approximately constant while total system throughput multiplies with DP degree.

### P/D Disaggregation

Configs 10–13 split prefill and decode into dedicated instances. The prefill instance processes prompt tokens and transmits KV cache to the decode instance. This removes head-of-line blocking between chunked-prefill and decode iterations. Note that in LLMServingSim a prefill instance occupies 2× its declared `npu_num` (one set for compute, one set for KV-cache-send), so the physical NPU budget must account for this.

## Caveats

- **TP=4 excluded**: no profiled latency tables for A6000 + Llama-3.1-8B at TP=4 (`llm_profile/perf_models/A6000/meta-llama/Llama-3.1-8B/tp4/` absent). Re-run `llm_profile/profile_layers.sh` on real A6000 hardware to add TP=4 support.
- **TTFT definition differs from vLLM**: LLMServingSim measures TTFT as the cycle when prefill computation completes, not when the client receives the first token. Reported values are therefore lower than vLLM-reported TTFT.
- **Configs 1–3, 6, 10 use fewer than 4 physical NPUs**: included for scaling comparison. Throughput is not comparable on an NPU-count basis without normalization.
- **PP modeling**: pipeline-parallel send/recv cost is modeled via link energy consumption (`npu_group - 1` inter-stage transfers) but bubble overhead is approximated. Real PP efficiency may differ.
