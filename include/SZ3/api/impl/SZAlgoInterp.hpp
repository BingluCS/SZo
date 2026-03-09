#ifndef SZ3_SZALGO_INTERP_HPP
#define SZ3_SZALGO_INTERP_HPP

#ifdef _OPENMP
#include "SZ3/decomposition/InterpolationDecomposition_Omp.hpp"
#include "SZ3/quantizer/LinearQuantizer_Omp.hpp"
#include "SZ3/compressor/SZGenericCompressor_Omp.hpp"
#include "SZ3/encoder/HuffmanEncode_OMP.hpp"
#endif

#include "SZ3/api/impl/SZAlgoLorenzoReg.hpp"
#include "SZ3/decomposition/BlockwiseDecomposition.hpp"
#include "SZ3/decomposition/InterpolationDecomposition.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/quantizer/LinearQuantizer.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Extraction.hpp"
#include "SZ3/utils/QuantOptimizatioin.hpp"
#include "SZ3/utils/Sample.hpp"
#include "SZ3/utils/Statistic.hpp"

namespace SZ3 {
template <class T, uint N>
size_t SZ_compress_Interp(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_INTERP);
    calAbsErrorBound(conf, data);
    if (conf.interpAnchorStride < 0) {  // set default anchor stride
        std::array<size_t, 4> anchor_strides = {4096, 128, 32, 16};
        conf.interpAnchorStride = anchor_strides[N - 1];
    }
    // conf.interpAlgo = 0;
    // conf.interpDirection = 0;
    // conf.interpAlpha = 1;
    // conf.interpBeta = 1;
#ifdef SZ3_PRINT_TIMINGS
    std::cout << "interpAlgo " << int(conf.interpAlgo) << "  interpDirection " << int(conf.interpDirection) << std::endl;
    std::cout << "interpAlpha " << conf.interpAlpha << " interpBeta " << conf.interpBeta << std::endl;
#endif
    #ifdef _OPENMP
    int threads = omp_get_max_threads();
    if (threads > 1) {
        auto sz = make_compressor_sz_generic_omp<T, N>(
        make_decomposition_interpolation_omp<T, N>(conf, LinearQuantizerOMP<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(),  Lossless_zstd());
        return sz->compress(conf, data, cmpData, cmpCap);
    }
    else {auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
    return sz->compress(conf, data, cmpData, cmpCap);}
    #else
        auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
            return sz->compress(conf, data, cmpData, cmpCap);
    #endif


}

template <class T, uint N>
void SZ_decompress_Interp(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_INTERP);
    auto cmpDataPos = cmpData;
   // std::cout<<"decomp started"<<std::endl;
    #ifdef _OPENMP
    int threads = omp_get_max_threads();
    if (threads > 1) {
    auto sz = make_compressor_sz_generic_omp<T, N>(
        make_decomposition_interpolation_omp<T, N>(conf, LinearQuantizerOMP<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
         sz->decompress(conf, cmpDataPos, cmpSize, decData);
    }
    else {
        auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
         sz->decompress(conf, cmpDataPos, cmpSize, decData);
    }
    #else
    auto sz = make_compressor_sz_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
         sz->decompress(conf, cmpDataPos, cmpSize, decData);
    #endif
   
}

template <class T, uint N>
double interp_compress_test(
    const std::vector<std::vector<T>> &sampled_blocks, const Config &conf, int block_size, uchar *cmpData,
    size_t cmpCap) {  // test interp cmp on a set of sampled data blocks and return the compression ratio
    /*
    #ifdef _OPENMP
     auto sz =
        make_decomposition_interpolation_omp<T, N>(conf, LinearQuantizerOMP<T>(conf.absErrorBound, conf.quantbinCnt / 2));
     #else
         auto sz =
        make_decomposition_interpolation<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
     #endif
   
    */

    // std::vector<std::vector<int> > quant_inds_vec(sampled_blocks.size());
    int** quant_inds = new int*[sampled_blocks.size()];
    int* quant_inds_size = new int[sampled_blocks.size()];
    std::vector<std::vector<T> > unpred_vec(sampled_blocks.size());
    // #ifdef _OPENMP
    // #pragma omp parallel for
    // #endif
    // for (size_t k = 0; k < sampled_blocks.size(); ++k) {
    //     unpred_buffer[k] = new uchar[sampled_blocks.size() * sizeof(T)
    //         * std::accumulate(conf.dims.begin(), conf.dims.end(), 1, std::multiplies<size_t>())];
    //     unpred_buffer_pos[k] = unpred_buffer[k];
    // }
    size_t sample_size = sampled_blocks.size();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t k = 0; k < sample_size; ++k) {
        auto sz =
            make_decomposition_interpolation<TUNING::ENABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
        auto cur_block = sampled_blocks[k];
        std::tie(quant_inds[k], quant_inds_size[k]) = sz.compress(conf, cur_block.data());
        unpred_vec[k] = sz.test_unpred();
    }
    std::vector<int> prefix_sum(sample_size << 1);
    int cur_size = 0;
    for (size_t k = 0; k < sample_size << 1; ++k) {
        prefix_sum[k] = cur_size;
        if(k < sample_size) {
            cur_size += unpred_vec[k].size() * sizeof(T);
        }
        else {
            cur_size += quant_inds_size[k - sample_size] * sizeof(int);
        }
    }
    uchar* buffer = new uchar[cur_size];
    for (size_t k = 0; k < sample_size << 1; ++k) {
        if(k < sample_size) {
            memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar*>(unpred_vec[k].data()), unpred_vec[k].size() * sizeof(T));
        }
        else {
            memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar*>(quant_inds[k - sample_size]), quant_inds_size[k - sample_size] * sizeof(int));
        }
    }
    auto lossless = Lossless_zstd();
    auto cmpSize = lossless.compress(buffer, cur_size, cmpData, cmpCap);

    auto compression_ratio = conf.num * sampled_blocks.size() * sizeof(T) * 1.0 / cmpSize;
    return compression_ratio;

}

template <class T, uint N>
double lorenzo_compress_test(
    const std::vector<std::vector<T>> sampled_blocks, const Config &conf, uchar *cmpData,
    size_t cmpCap) {  // test lorenzo cmp on a set of sampled data blocks and return the compression ratio
    std::vector<int> total_quant_bins;
    // if ((N == 3 && !conf.regression2) || (N == 1 && !conf.regression && !conf.regression2)) {
    std::vector<std::shared_ptr<concepts::PredictorInterface<T, N>>> predictors;
    predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 1>>(conf.absErrorBound));
    predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 2>>(conf.absErrorBound));
    auto sz = make_decomposition_blockwise<T, N>(conf, ComposedPredictor<T, N>(predictors),
                                                 LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
    // auto sz = make_decomposition_lorenzo_regression<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound,
    // conf.quantbinCnt / 2));
    for (size_t k = 0; k < sampled_blocks.size(); k++) {
        auto cur_block = sampled_blocks[k];
        // auto (quant_bins) = sz.compress(conf, cur_block.data());
        // total_quant_bins.insert(total_quant_bins.end(), quant_bins.begin(),
        //                         quant_bins.end());  // merge the quant bins. Lossless them together
        auto [quant_bins, quant_bins_size] = sz.compress(conf, cur_block.data());
        total_quant_bins.insert(total_quant_bins.end(), quant_bins, quant_bins + quant_bins_size);
    }
    auto encoder = HuffmanEncoder<int>();
    auto lossless = Lossless_zstd();
    encoder.preprocess_encode(total_quant_bins, conf.quantbinCnt);
    size_t bufferSize = std::max<size_t>(1000, 1.2 * (encoder.size_est() + sizeof(T) * total_quant_bins.size()));

    auto buffer = static_cast<uchar *>(malloc(bufferSize));
    uchar *buffer_pos = buffer;
    sz.save(buffer_pos);
    encoder.save(buffer_pos);

    // store the size of quant_inds is necessary as it is not always equal to conf.num
    write<size_t>(total_quant_bins.size(), buffer_pos);
    encoder.encode(total_quant_bins, buffer_pos);
    encoder.postprocess_encode();
    auto cmpSize = lossless.compress(buffer, buffer_pos - buffer, cmpData, cmpCap);
    free(buffer);
    auto compression_ratio = conf.num * sampled_blocks.size() * sizeof(T) * 1.0 / cmpSize;
    return compression_ratio;

    return 0;
}

template <class T, uint N>
size_t SZ_compress_Interp_lorenzo(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(conf.cmprAlgo == ALGO_INTERP_LORENZO);

#ifdef SZ3_PRINT_TIMINGS
    Timer timer(true);
#endif
    calAbsErrorBound(conf, data);

    if (conf.interpAnchorStride < 0) {  // set default anchor stride
        std::array<size_t, 4> anchor_strides = {4096, 128, 32, 16};
        conf.interpAnchorStride = anchor_strides[N - 1];
    }

    std::array<double, 4> sample_Rates = {0.005, 0.005, 0.005,
                                          0.005};  // default data sample rate. todo: add a config var to control
    auto sampleRate = sample_Rates[N - 1];
    std::array<size_t, 4> sampleBlock_Sizes = {4096, 128, 32,
                                               16};  // default sampled data block rate. Should better be no smaller
                                                     // than the anchor stride. todo: add a config var to control
    size_t sampleBlockSize = sampleBlock_Sizes[N - 1];
    size_t shortest_edge = conf.dims[0];
    for (size_t i = 0; i < N; i++) {
        shortest_edge = conf.dims[i] < shortest_edge ? conf.dims[i] : shortest_edge;
    }
    // Automatically adjust sampleblocksize.
    while (sampleBlockSize >= shortest_edge) sampleBlockSize /= 2;
    while (sampleBlockSize >= 16 && (pow(sampleBlockSize + 1, N) / conf.num) > 1.5 * sampleRate) sampleBlockSize /= 2;
    if (sampleBlockSize < 8) sampleBlockSize = 8;

    bool to_tune = pow(sampleBlockSize + 1, N) <= 0.05 * conf.num;  // to further revise
    for (auto &dim : conf.dims) {
        if (dim < sampleBlockSize) {
            to_tune = false;
            break;
        }
    }

    if (!to_tune) {  // if the sampled data would be too many (currently it is 5% of the input), skip the tuning
        conf.cmprAlgo = ALGO_INTERP;
        return SZ_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    }
#ifdef SZ3_PRINT_TIMINGS
    timer.stop("preparation");
    timer.start();
#endif
    std::vector<std::vector<T>> sampled_blocks;
    size_t per_block_ele_num = pow(sampleBlockSize + 1, N);
    size_t sampling_num;
    std::vector<std::vector<size_t>> starts;
    auto profStride = sampleBlockSize / 4;  // larger is faster, smaller is better
    profiling_block<T, N>(data, conf.dims, starts, sampleBlockSize, conf.absErrorBound,
                          profStride);  // filter out the non-constant data blocks
    size_t num_filtered_blocks = starts.size();
    bool profiling = num_filtered_blocks * per_block_ele_num >= 0.5 * sampleRate * conf.num;  // temp. to refine
    // bool profiling = false;
    sampleBlocks<T, N>(data, conf.dims, sampleBlockSize, sampled_blocks, sampleRate, profiling,
                       starts);  // sample out same data blocks
    sampling_num = sampled_blocks.size() * per_block_ele_num;

    if (sampling_num == 0 || sampling_num >= conf.num * 0.2) {
        conf.cmprAlgo = ALGO_INTERP;
        return SZ_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    }
#ifdef SZ3_PRINT_TIMINGS
    timer.stop("sampling");
    Timer timer_tuning(true);
    timer_tuning.start();
    timer.start();
#endif
    double best_lorenzo_ratio = 0, best_interp_ratio = 0, ratio;
    size_t bufferCap = conf.num * sizeof(T);
    auto buffer = static_cast<uchar *>(malloc(bufferCap));
    Config lorenzo_config = conf;

    {
        // tune interp
        conf.interpDirection = 0;
        conf.interpAlpha = 1.25;
        conf.interpBeta = 2.0;
        auto testConfig = conf;
        std::vector<size_t> dims(N, sampleBlockSize + 1);
        testConfig.setDims(dims.begin(), dims.end());
        uint8_t reverse = factorial(N) - 1;
        double ratios[4];

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i = 0; i < 4; i++) {
            size_t bufferCap = conf.num * sizeof(T);
            auto buffer = static_cast<uchar *>(malloc(bufferCap));
            auto test_config = testConfig;
            if (i / 2 == 0)
                test_config.interpAlgo = INTERP_ALGO_LINEAR;
            else
                test_config.interpAlgo = INTERP_ALGO_CUBIC;
            if (i % 2 == 0)
                test_config.interpDirection = 0; 
            else
                test_config.interpDirection = reverse;
            // std::cout<<"Testing interp algo "<<int(test_config.interpAlgo)<<" direction "<<int(test_config.interpDirection)<<std::endl;
            ratios[i] = interp_compress_test<T, N>(sampled_blocks, test_config, sampleBlockSize, buffer, bufferCap);
        }
        for(int i = 0; i < 4; i++) {
            if (ratios[i] > best_interp_ratio) {
                best_interp_ratio = ratios[i];
                if (i / 2 == 0)
                    conf.interpAlgo = INTERP_ALGO_LINEAR;
                else
                    conf.interpAlgo = INTERP_ALGO_CUBIC;
                if( i % 2 == 0)
                    conf.interpDirection = 0;
                else
                    conf.interpDirection = reverse;
            }
        }
        // for (auto &interp_op : {INTERP_ALGO_LINEAR,INTERP_ALGO_CUBIC}) {
        //     testConfig.interpAlgo = interp_op;
        //     ratio = interp_compress_test<T, N>(sampled_blocks, testConfig, sampleBlockSize, buffer, bufferCap);
        //     if (ratio > best_interp_ratio) {
        //         best_interp_ratio = ratio;
        //         conf.interpAlgo = interp_op;
        //     }
        // }
// #ifdef SZ3_PRINT_TIMINGS
//  timer.stop("interp tuning 0");
// #endif
// #ifdef SZ3_PRINT_TIMINGS
//     timer.start();
// #endif
//         testConfig.interpAlgo = conf.interpAlgo;
//         testConfig.interpDirection = factorial(N) - 1;
//         ratio = interp_compress_test<T, N>(sampled_blocks, testConfig, sampleBlockSize, buffer, bufferCap);
//         if (ratio > best_interp_ratio * 1.02) {
//             best_interp_ratio = ratio;
//             conf.interpDirection = testConfig.interpDirection;
//         }
// #ifdef SZ3_PRINT_TIMINGS
//         timer.stop("interp tuning 1");
//         timer.start();
// #endif
//         testConfig.interpDirection = conf.interpDirection;
//         // test more alpha-beta pairs for best compression ratio,
//         const int ablist_size = 3;
//        auto alphalist = std::array<double,ablist_size>{1.0, 1.5, 2.0};
//         auto betalist = std::array<double,ablist_size>{1.0, 2.5, 3.0};
//         std::array<double,ablist_size> ratios;
//         #ifdef _OPENMP
//         #pragma omp parallel for schedule(static)
//         #endif
//         for (size_t i = 0; i < ablist_size; i++) {
//             auto tempConfig = testConfig;
//             tempConfig.interpAlpha = alphalist[i];
//             tempConfig.interpBeta = betalist[i];
//             ratios[i] = interp_compress_test<T, N>(sampled_blocks, tempConfig, sampleBlockSize, buffer, bufferCap);
            
//         }
//         for (size_t i = 0; i < ablist_size; i++) {
//             auto ratio = ratios[i];
//             if (ratio > best_interp_ratio * 1.02) {
//                 best_interp_ratio = ratio;
//                 conf.interpAlpha = alphalist[i];
//                 conf.interpBeta = betalist[i];
//             }
//         }
// #ifdef SZ3_PRINT_TIMINGS
//         timer.stop("interp tuning 2");
// #endif
    }
#ifdef SZ3_PRINT_TIMINGS
 timer.stop("interp tuning");
#endif
    
#ifdef SZ3_PRINT_TIMINGS
    timer.start();
#endif
    {
        
        // only test lorenzo for 1D
        if (N == 1 && best_interp_ratio < 50) {
            std::vector<size_t> sample_dims(N, sampleBlockSize + 1);
            lorenzo_config.cmprAlgo = ALGO_LORENZO_REG;
            lorenzo_config.setDims(sample_dims.begin(), sample_dims.end());
            lorenzo_config.lorenzo = true;
            lorenzo_config.lorenzo2 = true;
            lorenzo_config.regression = false;
            lorenzo_config.regression2 = false;
            lorenzo_config.openmp = false;
            lorenzo_config.blockSize = 5;
            //        lorenzo_config.quantbinCnt = 65536 * 2;
            best_lorenzo_ratio = lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
            //            delete[]cmprData;
            //    printf("Lorenzo ratio = %.2f\n", ratio);
        }
    }


    bool useInterp = !(best_lorenzo_ratio >= best_interp_ratio * 1.1 && best_lorenzo_ratio < 50 &&
                       best_interp_ratio < 50);  // 1.1 is a fix coefficient. subject to revise
    size_t cmpSize = 0;
#ifdef SZ3_PRINT_TIMINGS
    timer.stop("lorenzo tuning");
    timer_tuning.stop("total tuning");
    timer.start();
#endif
    if (useInterp) {
        conf.cmprAlgo = ALGO_INTERP;
        cmpSize = SZ_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    } else {
        // no need to tune lorenzo for 3D anymore
        // if (N == 3) {
        //     float pred_freq, mean_freq;
        //     T mean_guess;
        //     lorenzo_config.quantbinCnt = optimize_quant_invl_3d<T>(
        //         data, conf.dims[0], conf.dims[1], conf.dims[2], conf.absErrorBound, pred_freq, mean_freq,
        //         mean_guess);
        //     lorenzo_config.pred_dim = 2;
        //     ratio  =
        //         lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
        //     if (ratio > best_lorenzo_ratio * 1.02) {
        //         best_lorenzo_ratio = ratio;
        //     } else {
        //         lorenzo_config.pred_dim = 3;
        //     }
        // }

        if (conf.relErrorBound < 1.01e-6 && best_lorenzo_ratio > 5 && lorenzo_config.quantbinCnt != 16384) {
            auto quant_num = lorenzo_config.quantbinCnt;
            lorenzo_config.quantbinCnt = 16384;
            ratio = lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
            if (ratio > best_lorenzo_ratio * 1.02) {
                best_lorenzo_ratio = ratio;
            } else {
                lorenzo_config.quantbinCnt = quant_num;
            }
        }
        lorenzo_config.setDims(conf.dims.begin(), conf.dims.end());
        conf = lorenzo_config;
        //            double tuning_time = timer.stop();
        cmpSize = SZ_compress_LorenzoReg<T, N>(conf, data, cmpData, cmpCap);
    }

    free(buffer);
    return cmpSize;
}
}  // namespace SZ3
#endif
