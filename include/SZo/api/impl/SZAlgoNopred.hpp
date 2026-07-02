#ifndef SZo_SZALGO_NOPRED_HPP
#define SZo_SZALGO_NOPRED_HPP

#include "SZo/compressor/SZGenericCompressor.hpp"
#include "SZo/decomposition/NoPredictionDecomposition.hpp"
#include "SZo/encoder/HuffmanEncoder.hpp"
#include "SZo/lossless/Lossless_zstd.hpp"
#include "SZo/quantizer/LinearQuantizer.hpp"
#include "SZo/utils/Config.hpp"

namespace SZo {
template <class T, uint N>
size_t SZ_compress_nopred(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_NOPRED);
    calAbsErrorBound(conf, data);

    auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_noprediction<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
    return sz->compress(conf, data, cmpData, cmpCap);
    //        return cmpData;
}

template <class T, uint N>
void SZ_decompress_nopred(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_NOPRED);
    auto cmpDataPos = cmpData;
    auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_noprediction<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
    sz->decompress(conf, cmpDataPos, cmpSize, decData);
}

}  // namespace SZo
#endif
