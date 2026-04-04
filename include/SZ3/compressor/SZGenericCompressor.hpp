#ifndef SZ3_COMPRESSOR_TYPE_ONE_HPP
#define SZ3_COMPRESSOR_TYPE_ONE_HPP

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/decomposition/Decomposition.hpp"
#include "SZ3/def.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/quantizer/LinearQuantizer.hpp"
#include "SZ3/decomposition/InterpolationDecomposition.hpp"

namespace SZ3 {
/**
 * SZGenericCompressor glues together decomposition, encoder, and lossless modules to form the compressor.
 * It only takes Decomposition, not Predictor.
 * @tparam T original data type
 * @tparam N original data dimension
 * @tparam Decomposition decomposition module
 * @tparam Encoder encoder module
 * @tparam Lossless lossless module
 */
template <class T, uint N, class Decomposition, class Encoder, class Lossless>
class SZGenericCompressor : public concepts::CompressorInterface<T> {
   public:
    SZGenericCompressor(Decomposition decomposition, Encoder encoder, Lossless lossless)
        : decomposition(decomposition), encoder(encoder), lossless(lossless) {
        static_assert(std::is_base_of<concepts::DecompositionInterface<T, int, N>, Decomposition>::value,
                      "must implement the frontend interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }
    using TargetDecomposition =
    decltype(make_decomposition_interpolation<TUNING::DISABLED, T, N>(
        std::declval<Config>(),
        std::declval<LinearQuantizer<T>>()
    ));

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
#ifdef SZ3_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto [quant_inds, quant_inds_size] = decomposition.compress(conf, data);
#ifdef SZ3_PRINT_TIMINGS
        timer.stop("cmp interp");
#endif

    // std::ofstream out("quant.bin", std::ios::binary);
    // out.write(
    //     reinterpret_cast<const char*>(quant_inds),
    //     quant_inds_size * sizeof(int)
    // );

#ifdef SZ3_PRINT_TIMINGS
        timer.start();
#endif
        size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + sizeof(T) * quant_inds_size));
        auto buffer = static_cast<uchar *>(malloc(bufferSize));
        uchar *buffer_pos = buffer;
        auto cmpDataPos = cmpData;
        {
            decomposition.save(buffer_pos);
            write<size_t>(quant_inds_size, buffer_pos);
            if constexpr ((std::is_same_v<Decomposition, TargetDecomposition>)) {
               encoder.preprocess_encode(quant_inds, quant_inds_size, decomposition.get_out_range().second, decomposition.frequency);
            }
            else encoder.preprocess_encode(quant_inds, quant_inds_size, decomposition.get_out_range().second);
            
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, quant_inds_size, buffer_pos);
            encoder.postprocess_encode();

            auto currentSize = buffer_pos - buffer;
            cmpCap -= cmpDataPos-cmpData;

            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        }
#ifdef SZ3_PRINT_TIMINGS
        timer.stop("zstd");
#endif
        return cmpDataPos-cmpData;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {

#ifdef SZ3_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto cmpDataPos = cmpData;
        int* quant_inds;
        size_t quant_inds_size = 0;

        uchar *buffer = nullptr;
        size_t bufferSize = 0;

        cmpSize -= cmpDataPos - cmpData; 
        lossless.decompress(cmpDataPos, cmpSize, buffer, bufferSize);

        uchar const *bufferPos = buffer;
        decomposition.load(bufferPos, bufferSize);
        read(quant_inds_size, bufferPos);

        // std::cout << "quant_inds_size: " << quant_inds_size << std::endl;
        encoder.load(bufferPos, bufferSize);
        quant_inds = encoder.decode(bufferPos, quant_inds_size);
        

#ifdef SZ3_PRINT_TIMINGS
        timer.stop("huffman");
#endif
        free(buffer);
#ifdef SZ3_PRINT_TIMINGS
    timer.start();
#endif
        decomposition.decompress(conf, quant_inds, decData);
#ifdef SZ3_PRINT_TIMINGS
    timer.stop("decomposition decompress");
#endif
        return decData;
    }

   private:
    Decomposition decomposition;
    Encoder encoder;
    Lossless lossless;
};

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
std::shared_ptr<SZGenericCompressor<T, N, Decomposition, Encoder, Lossless>> make_compressor_sz_generic(
    Decomposition decomposition, Encoder encoder, Lossless lossless) {
    return std::make_shared<SZGenericCompressor<T, N, Decomposition, Encoder, Lossless>>(decomposition, encoder,
                                                                                         lossless);
}

}  // namespace SZ3
#endif