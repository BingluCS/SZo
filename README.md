# SZo: Ultra-Fast and Scalable High-Ratio Scientific Error-Bounded Lossy Compression on CPUs

## Build & install

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] .. -DENABLE_AVX2=ON
make
make install
```

Executables land in `[INSTALL_DIR]/bin`, headers in `[INSTALL_DIR]/include`.

**SIMD / AVX2**
* `-DENABLE_AVX2=ON` turns on the AVX2 code paths (x86, needs CPU support) — a large speedup.
* On ARM use `-DENABLE_SVE2=ON` instead.
* With neither flag you get a portable scalar build. **Scalar, AVX2 and SVE2 all produce bit-identical output.**
* The SIMD flag is attached to the `SZo` CMake target, so **everything that links `SZo` inherits it automatically** — the `szo` binary, the `SZOc` C library, and the HDF5 filter. You only set it once, at configure time.

**Optional components**
* `-DBUILD_H5Z_FILTER=ON` also builds the HDF5 filter (requires the HDF5 development package).

## How to run

### SZo executable
`tools/szo/szo` — command-line compression / decompression. Run it with no arguments to see usage.

### C++ API
* Header-only: `#include "SZo/api/sz.hpp"`; everything lives in namespace `SZo`. Needs a C++17 compiler. (Different from the SZ2 API.)
```cpp
#include "SZo/api/sz.hpp"

SZo::Config conf(dim0, dim1, dim2);          // n-D dimensions
conf.errorBoundMode = SZo::EB_ABS;
conf.absErrorBound  = 1e-3;

size_t outSize;
char  *comp = SZ_compress(conf, data, outSize);          // T* data -> compressed bytes
T     *dec  = SZ_decompress<T>(conf, comp, outSize);     // -> reconstructed data
```
* **AVX2/SVE2:** compile the translation unit that includes the header with `-mavx2 -mfma` (x86) or `-march=armv8.6-a+sve2` (ARM). If you build through this project's CMake, linking the `SZo` target adds the flag for you.

### C API (SZOc)
* Header `tools/szoc/include/szo.h`, library target `SZOc`. **Compatible with the SZ2 C API.**
```c
#include "szo.h"

size_t outSize;
unsigned char *comp = SZ_compress_args(SZ_FLOAT, data, &outSize,
                                       ABS, /*absErr*/1e-3, 0, 0, 0, 0, r3, r2, r1);
void *dec = SZ_decompress(SZ_FLOAT, comp, outSize, 0, 0, r3, r2, r1);
free_buf(comp);
```
* **AVX2/SVE2:** `SZOc` picks up the SIMD flags automatically when the project is configured with `-DENABLE_AVX2=ON` (or `-DENABLE_SVE2=ON`).

### Python API (pyszo)
* `pip install pyszo`. [Source in `tools/pyszo`](https://github.com/BingluCS/SZo/tree/master/tools/pyszo).
```python
import numpy as np
from pyszo import sz, szoConfig, szoErrorBoundMode, szoAlgorithm

data   = np.random.rand(100, 200, 300).astype(np.float32)
config = szoConfig()
config.errorBoundMode = szoErrorBoundMode.ABS
config.absErrorBound  = 1e-3
# optional: config.cmprAlgo = szoAlgorithm.INTERP_LORENZO

compressed, ratio       = sz.compress(data, config)
decompressed, config    = sz.decompress(compressed, np.float32, data.shape)
max_err, psnr, nrmse    = sz.verify(data, decompressed.reshape(data.shape))
```
* `sz.compress` works on a private copy, so your input array is **left unchanged**. `sz.decompress(data, dtype, shape)` recovers the SZo configuration from the compressed stream, so passing the original `config` is optional.
* The wheel builds SZo from source; the bundled Zstd and (on x86) the SIMD build options are handled by `setup.py`.

### HDF5 filter (H5Z-SZo)
* Located in `tools/H5Z-SZo`; library target `hdf5szo`, HDF5 filter id **32024**. Build with `-DBUILD_H5Z_FILTER=ON` (requires HDF5).
* Add `-DENABLE_AVX2=ON` to vectorize the compression that runs *inside* the filter — the filter's CMake sets the SIMD flags explicitly, so it is AVX2/SVE2-accelerated just like the rest of SZo.
* `szoToHDF5` and `HDF5ToSzo` (under `tools/H5Z-SZo/test`) are provided for testing — e.g. `szoToHDF5 -f data.dat 100 100 100` writes `data.dat.szo.h5`, and `HDF5ToSzo data.dat.szo.h5` reconstructs it. The type flag is `-f` (float), `-d` (double), or `-i8/-u8/-i16/-u16/-i32/-u32/-i64/-u64` for integers.
* To use it as a dynamically-loaded HDF5 filter, point `HDF5_PLUGIN_PATH` at the built `libhdf5szo`, then request filter id `32024` on your dataset-creation property list.



<!-- ## Citations
[//]: # (**Kindly note**: If you mention SZ3 in your paper, the most appropriate citation is to include these three references &#40;**TBD22, ICDE21, Bigdata18**&#41; because they cover the design and implementation of the latest version of SZ.)
* QOZv2 (the enhanced interpolation-based algorithm): [High-performance Effective Scientific Error-bounded Lossy Compression with Auto-tuned Multi-component Interpolation](https://dl.acm.org/doi/10.1145/3639259).
* SZ3's interpolation-based algorithm: [Optimizing Error-Bounded Lossy Compression for Scientiﬁc Data by Dynamic Spline Interpolation](https://ieeexplore.ieee.org/document/9458791).
* The software engineering design of SZ3: [SZ3: A modular framework for composing prediction-based error-bounded lossy compressors](https://ieeexplore.ieee.org/abstract/document/9866018).


## Version history

Version New features

* SZ 3.0.0 SZ3 is the C++ version of SZ with a modular and composable design.
* SZ 3.0.1 Improve the build process.
* SZ 3.1.0 The default algorithm is now interpolation+Lorenzo.
* SZ 3.1.1 Add OpenMP support. Works for all algorithms. Please enable it using the config file. 
* SZ 3.1.2 Support configuration file (INI format). An example can be found in 'tools/szo/szo.config'.
* SZ 3.1.3 Support more error control mode: PSNR, L2Norm, ABS_AND_REL, ABS_OR_REL. Support INT32 and INT64 datatype.
* SZ 3.1.4 Support running on Windows natively with Visual Studio. Please use CMake to generate Visual Studio solution files.
* SZ 3.1.5 Support HDF5 by H5Z-SZ3. Please add "-DBUILD_H5Z_FILTER=ON" to enable this function for CMake.
* SZ 3.1.6 Support C API and Python API.
* SZ 3.1.7 Initial MDZ(https://github.com/szcompressor/SZ3/tree/master/tools/mdz) support.
* SZ 3.1.8 namespace changed from SZ to SZ3. H5Z-SZ3 supports configuration files now.
* SZ 3.2.0 API reconstructed for FZ. H5Z-SZ3 rewrite. Compression version checking.
* SZ 3.3.0 Add key QoZ v1 and v2 features to improve compression speed and data quality. The full QoZ is available from **a separate branch** (https://github.com/szcompressor/SZ3/tree/QoZ). 
* SZ 3.3.1: SZ3 Windows support for both Visual Studio and MinGW toolchains. pySZ v1 released and available via `pip install pysz`.

## 3rd party libraries/tools
* [Zstandard](https://facebook.github.io/zstd/) v1.4.5 will be fetched if libzstd can not be found by pkg-config.
* The source code of ska_hash is included in SZ3. -->
