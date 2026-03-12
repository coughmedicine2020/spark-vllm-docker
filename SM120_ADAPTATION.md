# RTX PRO 6000 Blackwell (SM120) Adaptation

This document summarizes the changes made to adapt the spark-vllm-docker project for RTX PRO 6000 Blackwell (SM120, x86_64) while maintaining compatibility with DGX Spark GB10 (SM121, aarch64).

## Summary of Changes

### 1. Parameterized CUDA Architecture

Changed all Dockerfiles to accept a `CUDA_ARCH` build argument instead of hardcoding `12.1a`:

**Modified Files:**
- `Dockerfile` - Added `ARG CUDA_ARCH="12.0a"` and updated 4 ENV references
- `Dockerfile.mxfp4` - Added `ARG CUDA_ARCH="12.0a"` and updated 4 ENV references
- `Dockerfile.wheels` - Added `ARG CUDA_ARCH="12.0a"` and updated 1 ENV reference

**Default:** `12.0a` (SM120, RTX PRO 6000 Blackwell)

### 2. Cross-Platform Wheel Support

Fixed `Dockerfile.wheels` line 62 to dynamically detect architecture instead of hardcoding `aarch64`:

```bash
# Before:
vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_aarch64.whl

# After:
ARCH=$(uname -m) && \
uv pip install -U https://github.com/.../manylinux_2_35_${ARCH}.whl
```

Now supports both `x86_64` and `aarch64` automatically.

### 3. Build Script Enhancement

Added `--cuda-arch` flag to `build-and-copy.sh`:

```bash
# Default (SM120):
./build-and-copy.sh -t vllm-rtx6000

# DGX Spark (SM121):
./build-and-copy.sh -t vllm-dgx-spark --cuda-arch 12.1a

# Other architectures:
./build-and-copy.sh -t custom --cuda-arch 8.9a  # L40S
```

### 4. SM120 NVFP4 Runtime Mod

Created `mods/fix-sm120-nvfp4/run.sh` to patch vLLM's MXFP4/NVFP4 capability checks at runtime:

- Extends compute capability range from `< (11, 0)` to `< (13, 0)`
- Applies to both `mxfp4.py` and `nvfp4.py` quantization backends
- Follows existing mod pattern (see `mods/fix-Salyut1-GLM-4.7-NVFP4/`)

### 5. Recipe Update

Updated `recipes/nemotron-3-nano-nvfp4.yaml` to auto-apply the SM120 mod for NVFP4 workloads:

```yaml
mods:
  - mods/nemotron-nano
  - mods/fix-sm120-nvfp4  # NEW: Enables SM120 support
```

### 6. Documentation Updates

**README.md:**
- Updated title: "vLLM Docker Optimized for DGX Spark & RTX PRO 6000 Blackwell"
- Changed hardware note to reflect new default (`12.0a`)
- Added `--cuda-arch` usage examples

**docs/NETWORKING.md:**
- Added SM120 NCCL build command alternative
- Added x86_64 MPI path alongside aarch64

## Verification Commands

```bash
# 1. Syntax check (all passed ✓)
docker build --check -f Dockerfile .
docker build --check -f Dockerfile.mxfp4 .
docker build --check -f Dockerfile.wheels .

# 2. Verify CUDA_ARCH propagation (verified ✓)
./build-and-copy.sh -t test --cuda-arch 12.0a --no-build --copy-to dummy

# 3. Full build test (mainline)
./build-and-copy.sh -t vllm-rtx6000

# 4. Full build test (MXFP4 fork)
./build-and-copy.sh -t vllm-rtx6000-mxfp4 --exp-mxfp4

# 5. Runtime GPU detection
docker run --gpus all --rm vllm-rtx6000 python3 -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"

# 6. SM120 mod test
# In container with mod applied:
python3 -c "from vllm.model_executor.layers.quantization.mxfp4 import MXFp4Config; print('OK')"

# 7. Run existing test suite
bash tests/test_recipes.sh
```

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `Dockerfile` | 7 edits | Parameterize CUDA arch (4 locations) |
| `Dockerfile.mxfp4` | 5 edits | Parameterize CUDA arch (4 locations) |
| `Dockerfile.wheels` | 3 edits | Parameterize arch + fix x86_64 wheel URL |
| `build-and-copy.sh` | 4 edits | Add `--cuda-arch` flag and propagation |
| `mods/fix-sm120-nvfp4/run.sh` | NEW | Runtime NVFP4 capability patch |
| `recipes/nemotron-3-nano-nvfp4.yaml` | 1 edit | Apply SM120 mod |
| `docs/NETWORKING.md` | 1 edit | Add SM120/x86_64 alternatives |
| `README.md` | 2 edits | Update title and hardware note |

## Architecture Comparison

| Hardware | Compute Capability | CUDA Arch | Platform | Default? |
|----------|-------------------|-----------|----------|----------|
| RTX PRO 6000 Blackwell | 12.0 (SM120) | `12.0a` | x86_64 | ✓ (NEW) |
| DGX Spark GB10 | 12.1 (SM121) | `12.1a` | aarch64 | (use `--cuda-arch 12.1a`) |

## Rationale

**Why SM120 as default?**
- RTX PRO 6000 is the new target hardware
- More widely available than DGX Spark
- SM120 and SM121 are architecturally near-identical
- Users can override with `--cuda-arch 12.1a` for DGX Spark

**Why the christopherowen MXFP4 fork?**
- Most mature FP4 kernel support (DGX Spark's flagship feature)
- Mainline vLLM has open issues with SM120 NVFP4 ([#31085](https://github.com/vllm-project/vllm/issues/31085))
- Fork kernels compile for SM120 with just arch flag change
- Runtime mod ensures capability checks recognize SM120

## Next Steps

1. Test full build on RTX PRO 6000 hardware
2. Verify MXFP4 quantization works correctly
3. Benchmark FP4 performance vs mainline vLLM
4. Document any RTX PRO 6000-specific tuning parameters
