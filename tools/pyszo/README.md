# pyszo

Python bindings for **SZo** — ultra-fast, error-bounded lossy compression for scientific data.

## Overview

`pyszo` provides a clean Python interface to SZo, built with Cython for high performance. It compresses NumPy arrays (`float32`, `float64`, `int32`, `int64`) under a user-specified error bound and recovers them to within that bound.

## Installation

### From PyPI (recommended)

```bash
pip install pyszo
```

Pre-built binary wheels are available for most platforms (Linux, macOS, Windows) and Python versions. They bundle SZo and Zstd — no build tools required.

### Building from source

If a wheel isn't available for your platform, you need:

- **CMake ≥ 3.13**
- a **C++17 compiler** (g++, clang++, or MSVC)
- **Git**
- **Python development headers** (`python3-dev` / `python3-devel`)

Then:

```bash
git clone https://github.com/BingluCS/SZo.git
cd SZo/tools/pyszo
pip install -e .
```

During a source install, `setup.py` builds SZo from source (with bundled Zstd), compiles the Cython bindings against it, and packages everything together.

### SIMD acceleration (AVX2 / SVE2)

SZo's fast paths are gated on the compiler's SIMD flags, so `pyszo` decides them **when it compiles the bindings** (the translation unit that includes the SZo headers). Without a flag, SZo falls back to the portable scalar path.

- **Default — auto-detect:** builds with `-march=native`, which enables **AVX2** on an AVX2-capable x86 and **SVE2** on an SVE2-capable ARM, and stays scalar on anything else. Ideal when you build and run on the same machine.
- **Explicit:** set the `PYSZO_SIMD` environment variable before installing:

  | `PYSZO_SIMD` | effect |
  |---|---|
  | *(unset)* / `auto` / `native` | `-march=native` (default) — the build machine's ISA |
  | `avx2` | `-mavx2 -mfma` — a portable AVX2 floor (x86) |
  | `sve2` | `-march=armv8.6-a+sve2` — ARM SVE2 |
  | `none` / `scalar` | no SIMD — portable everywhere, slowest |
  | *(anything else)* | passed verbatim, e.g. `PYSZO_SIMD="-march=znver4"` |

  ```bash
  PYSZO_SIMD=avx2 pip install -e .     # force a portable AVX2 floor
  PYSZO_SIMD=none pip install -e .     # scalar, maximum portability
  ```

> **Portability note:** `-march=native` (and `avx2` / `sve2`) bake those instructions into the `.so`, so it must run on a CPU that supports them — an unsupported CPU raises `SIGILL`. For a redistributable wheel, pick an explicit floor (`avx2`) or `none`; for a local `pip install -e .`, the default `native` is best. The chosen flags are printed during the build: `[pyszo] SIMD flags: …`.

## Quick Start

```python
import numpy as np
from pyszo import sz, szoConfig, szoErrorBoundMode, szoAlgorithm

# Create test data
data = np.random.rand(8, 8, 128).astype(np.float32)

# Configure
config = szoConfig(data.shape)
config.errorBoundMode = szoErrorBoundMode.ABS
config.absErrorBound  = 1e-3
# optional: config.cmprAlgo = szoAlgorithm.INTERP_LORENZO

# Compress (runs on a private copy — your input array is left unchanged)
compressed, ratio = sz.compress(data, config)
print(f"Compression ratio: {ratio:.2f}x")

# Decompress (the SZo config is recovered from the stream, so config is optional)
decompressed, config = sz.decompress(compressed, np.float32, data.shape)

# Verify
max_err, psnr, nrmse = sz.verify(data, decompressed)
print(f"Max error: {max_err:.2e}, PSNR: {psnr:.2f} dB, NRMSE: {nrmse:.2e}")
```

## API Reference

### `szoConfig`

Configuration object mirroring the C++ `Config` class.

```python
config = szoConfig(data.shape)       # shape tuple (recommended)
config = szoConfig((100, 200, 300))  # tuple
config = szoConfig([100, 200, 300])  # list
config = szoConfig(100, 200, 300)    # individual dimensions
```

Common fields:

- `errorBoundMode` — an `szoErrorBoundMode` (see below)
- `absErrorBound`, `relErrorBound` — the bound values
- `cmprAlgo` — an `szoAlgorithm` (see below)
- `interpAlgo` — an `szoInterpAlgorithm` (`LINEAR` or `CUBIC`)

Load settings from an INI config file:

```python
config = szoConfig(data.shape)
config.loadcfg('szo.config')
```

### Enums

```python
from pyszo import szoErrorBoundMode, szoAlgorithm, szoInterpAlgorithm

szoErrorBoundMode.ABS | REL | PSNR | L2NORM | ABS_AND_REL | ABS_OR_REL
szoAlgorithm.LORENZO_REG | INTERP_LORENZO | INTERP | NOPRED | LOSSLESS
szoInterpAlgorithm.LINEAR | CUBIC
```

### `sz.compress()`

```python
sz.compress(data, config) -> (compressed, ratio)
```

Compress a NumPy array. Dimensions are inferred from the array shape. Runs on a private copy, so `data` is left unchanged.

- `data` (ndarray): `float32`, `float64`, `int32`, or `int64`
- `config` (`szoConfig` or `str`): config object, or a path to a config file
- returns `compressed` (uint8 ndarray) and `ratio` (original / compressed size)

### `sz.decompress()`

```python
sz.decompress(compressed, dtype, shape, config=None) -> (data, config)
```

Decompress back to a NumPy array. The SZo configuration is recovered from the stream, so `config` is optional.

- `compressed` (ndarray): uint8 array from `compress()`
- `dtype` (type): `np.float32`, `np.float64`, `np.int32`, or `np.int64`
- `shape` (tuple): shape of the original data
- returns the decompressed `data` and the recovered `config`

### `sz.verify()`

```python
sz.verify(src_data, dec_data) -> (max_diff, psnr, nrmse)
```

Quality metrics between the original and decompressed arrays: maximum absolute difference, PSNR (dB), and NRMSE.

## Usage Examples

### Relative error bound

```python
config = szoConfig(data.shape)
config.errorBoundMode = szoErrorBoundMode.REL
config.relErrorBound  = 1e-4          # 0.01% relative error
compressed, ratio = sz.compress(data, config)
```

### Choosing an algorithm

```python
config.cmprAlgo = szoAlgorithm.INTERP_LORENZO   # default (best quality)
config.cmprAlgo = szoAlgorithm.INTERP           # interpolation only
config.cmprAlgo = szoAlgorithm.LORENZO_REG      # Lorenzo / regression
config.cmprAlgo = szoAlgorithm.LOSSLESS         # lossless only
```

### Double precision

```python
data = np.random.randn(50, 50, 50).astype(np.float64)
config = szoConfig(data.shape)
config.errorBoundMode = szoErrorBoundMode.ABS
config.absErrorBound  = 1e-6
compressed, ratio     = sz.compress(data, config)
decompressed, config  = sz.decompress(compressed, np.float64, data.shape)
```

### Save / load compressed data

```python
# Compress and save
compressed, ratio = sz.compress(data, config)
compressed.tofile('data.szo')

# Later: load and decompress
compressed = np.fromfile('data.szo', dtype=np.uint8)
decompressed, config = sz.decompress(compressed, np.float32, (8, 8, 128))
```

## Troubleshooting

**`ImportError: cannot import name 'sz' from 'pyszo'`** — reinstall the package:

```bash
cd tools/pyszo
pip install -e .
```

**`OSError: libzstd.so: cannot open shared object file`** — point the loader at the bundled Zstd:

```bash
export LD_LIBRARY_PATH="../../build/tools/zstd:$LD_LIBRARY_PATH"
```

## Links

- **Repository:** https://github.com/BingluCS/SZo
- **Issues:** https://github.com/BingluCS/SZo/issues

## License

See `../../copyright-and-BSD-license.txt`.
