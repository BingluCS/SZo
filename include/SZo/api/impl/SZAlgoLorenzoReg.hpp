#ifndef SZo_SZALGO_LORENZO_REG_HPP
#define SZo_SZALGO_LORENZO_REG_HPP

#include <cmath>
#include <memory>

#include "SZo/compressor/SZGenericCompressor.hpp"
#include "SZo/decomposition/BlockwiseDecomposition.hpp"
#include "SZo/def.hpp"
#include "SZo/lossless/Lossless_zstd.hpp"
#include "SZo/predictor/ComposedPredictor.hpp"
#include "SZo/predictor/LorenzoPredictor.hpp"
#include "SZo/predictor/RegressionPredictor.hpp"
#include "SZo/quantizer/LinearQuantizer.hpp"
#include "SZo/utils/Config.hpp"
#include "SZo/utils/Iterator.hpp"
#include "SZo/utils/QuantOptimizatioin.hpp"
#include "SZo/utils/Statistic.hpp"

namespace SZo {
template <class T, uint N, class Quantizer, class Encoder, class Lossless>
std::shared_ptr<concepts::CompressorInterface<T>> make_compressor_lorenzo_regression(const Config &conf,
                                                                                     Quantizer quantizer,
                                                                                     Encoder encoder,
                                                                                     Lossless lossless) {
    std::vector<std::shared_ptr<concepts::PredictorInterface<T, N>>> predictors;

    int methodCnt = (conf.lorenzo + conf.lorenzo2 + conf.regression);
    int use_single_predictor = (methodCnt == 1);
    if (methodCnt == 0) {
        throw std::invalid_argument("All lorenzo and regression methods are disabled.");
    }
    if (conf.lorenzo) {
        if (use_single_predictor) {
            return make_compressor_sz_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, LorenzoPredictor<T, N, 1>(conf.absErrorBound), quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 1>>(conf.absErrorBound));
        }
    }
    if (conf.lorenzo2) {
        if (use_single_predictor) {
            return make_compressor_sz_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, LorenzoPredictor<T, N, 2>(conf.absErrorBound), quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 2>>(conf.absErrorBound));
        }
    }
    if (conf.regression) {
        if (use_single_predictor) {
            return make_compressor_sz_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, RegressionPredictor<T, N>(conf.blockSize, conf.absErrorBound),
                                                   quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<RegressionPredictor<T, N>>(conf.blockSize, conf.absErrorBound));
        }
    }

    return make_compressor_sz_generic<T, N>(
        make_decomposition_blockwise<T, N>(conf, ComposedPredictor<T, N>(predictors), quantizer), encoder, lossless);
}

template <class T, uint N>
size_t SZ_compress_LorenzoReg(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_LORENZO_REG);
    calAbsErrorBound(conf, data);

    auto quantizer = LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
    auto sz = make_compressor_lorenzo_regression<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    return sz->compress(conf, data, cmpData, cmpCap);
}

template <class T, uint N>
void SZ_decompress_LorenzoReg(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_LORENZO_REG);
    auto cmpDataPos = cmpData;
    LinearQuantizer<T> quantizer;
    auto sz = make_compressor_lorenzo_regression<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    sz->decompress(conf, cmpDataPos, cmpSize, decData);
}
}  // namespace SZo
#endif
