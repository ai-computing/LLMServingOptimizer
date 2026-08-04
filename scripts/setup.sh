#!/bin/bash
# LLMServingOptimizer one-shot environment setup.
#
# Reproduces the verified environment for BOTH simulator backends:
#   backends/legacy   — ai-computing/LLMServingSim fork  (main.py)
#   backends/upstream — casys-kaist/LLMServingSim v1.1+  (python -m serving)
#
# Idempotent: safe to re-run; completed steps are skipped cheaply.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/6] git submodules (recursive: astra-sim nested inside each backend)"
git submodule update --init --recursive

# ---------------------------------------------------------------- legacy ----
echo "==> [2/6] legacy backend: chakra (user-level) + ASTRA-Sim analytical"
# The legacy chakra (protobuf 3.x era) installs into the user site — it must
# NOT share an env with upstream's chakra (which needs protobuf>=7.35).
if ! python3 -c "import chakra" 2>/dev/null; then
    (cd backends/legacy/astra-sim/extern/graph_frontend/chakra && pip3 install --user .)
fi
LEGACY_BIN=backends/legacy/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
if [[ ! -f "$LEGACY_BIN" ]]; then
    (cd backends/legacy/astra-sim && bash ./build/astra_analytical/build.sh)
fi
mkdir -p backends/legacy/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin
ln -sf "$ROOT/$LEGACY_BIN" \
    backends/legacy/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra

# -------------------------------------------------------------- upstream ----
echo "==> [3/6] upstream backend: .venv + chakra(protobuf>=7.35) + msgspec"
if [[ ! -x backends/upstream/.venv/bin/python ]]; then
    python3 -m venv backends/upstream/.venv
    backends/upstream/.venv/bin/pip install --quiet --upgrade pip
    backends/upstream/.venv/bin/pip install --quiet rich numpy pandas pyyaml pyinstrument msgspec
    (cd backends/upstream/astra-sim/extern/graph_frontend/chakra \
        && "$ROOT/backends/upstream/.venv/bin/pip" install --quiet .)
    # chakra pins protobuf==6.* but ships gencode built with 7.35 — upgrade.
    backends/upstream/.venv/bin/pip install --quiet --upgrade protobuf
fi

echo "==> [4/6] upstream backend: ASTRA-Sim analytical"
UP_BIN=backends/upstream/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
if [[ ! -f "$UP_BIN" ]]; then
    (cd backends/upstream/astra-sim && bash ./build/astra_analytical/build.sh)
fi
mkdir -p backends/upstream/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin
ln -sf "$ROOT/$UP_BIN" \
    backends/upstream/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra

echo "==> [5/6] link our upstream-format profiles/configs into the submodule worktree"
# our self-profiled hardware (not in the casys-kaist remote) lives in
# profiles/upstream/ and is linked in so the upstream sim can find it
for hw in profiles/upstream/*/; do
    [[ -d "$hw" ]] || continue
    ln -sfn "../../../../profiles/upstream/$(basename "$hw")" \
        "backends/upstream/profiler/perf/$(basename "$hw")"
done
for f in cluster_config/upstream/*.json; do
    [[ -f "$f" ]] || continue
    ln -sf "../../../../$f" "backends/upstream/configs/cluster/$(basename "$f")"
done
# model configs for checkpoints the submodule does not ship (the quantized ones
# we profile). The simulator's get_config() only looks under the submodule's own
# configs/model/ — unlike the profiler there is no --model-config-root at
# simulation time — so a profile without this link lists in the catalog but
# fails to evaluate. Linked per file so vendors the submodule already has
# (meta-llama, Qwen) keep their real directory.
for cfg in configs/upstream_model/*/*.json; do
    [[ -f "$cfg" ]] || continue
    vendor="$(basename "$(dirname "$cfg")")"
    mkdir -p "backends/upstream/configs/model/$vendor"
    ln -sf "../../../../../$cfg" \
        "backends/upstream/configs/model/$vendor/$(basename "$cfg")"
done

echo "==> [6/6] smoke check"
python3 - <<'EOF'
from sim_backends import get_backend
for name in ("legacy", "upstream"):
    b = get_backend(name)
    hw = sorted(b.list_hardware())
    print(f"  {name:9s} available={b.available()}  hardware={hw}")
EOF

echo "setup complete."
echo "(optional, GPU work only) vLLM venv for upstream profiler/bench:"
echo "  cd backends/upstream && uv venv --python 3.12 .venv-vllm \\"
echo "    && VLLM_USE_PRECOMPILED=1 uv pip install --python .venv-vllm/bin/python vllm==0.19.0"
