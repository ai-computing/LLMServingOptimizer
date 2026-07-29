# Power profile schema (`profiles/power/<HW>.yaml`)

Canonical per-hardware power source for the planner's power-min mode and the
service layer (design doc §4.2). Loaded/validated by `planner/power_profiles.py`.

```yaml
device:                      # required
  name: A40                  # hardware key (must match topology device names)
  active_w: 300              # per-device active power ceiling, W (> 0)
  idle_w: 25                 # per-device idle power, W (>= 0)
  mem_gb: 48                 # device memory, GB (> 0)
host_overhead:               # optional (defaults 0/0)
  base_w: 250                # CPU/DRAM/NIC/fans per active host, W
  per_device_w: 15           # marginal host power per installed device, W
measured:                    # optional — wins over active_w when present
  - {model: "meta-llama/Llama-3.1-70B", tp: 4, load: 0.8, avg_w: 1180}
    # avg_w is whole-instance power (all tp devices), load in (0, 1]
meta:                        # optional provenance
  source: nvidia-smi | datasheet | estimate
  measured_at: "2026-07-01"
  stack: "vllm-0.19"         # stack version the measured rows belong to
```

Resolution order (`effective_power_w`): `measured` rows for (model, tp) with
linear interpolation over `load` (clamped outside the measured range) →
`device.active_w` × tp → legacy constant table with a warning when the file
itself is missing. The shipped profiles keep `active_w` identical to the old
`milp_solver` constants so solver output is unchanged either way.
