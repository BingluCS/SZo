#ifndef SZ3_COMPRESSOR_TYPE_ONE_OMP_HPP
#define SZ3_COMPRESSOR_TYPE_ONE_OMP_HPP

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/decomposition/Decomposition.hpp"
#include "SZ3/def.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/encoder/Encoder_OMP.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Timer.hpp"

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
class SZGenericCompressor_OMP : public concepts::CompressorInterface<T> {
   public:
    SZGenericCompressor_OMP(Decomposition decomposition, Encoder encoder, Lossless lossless)
        : decomposition(decomposition), encoder(encoder), lossless(lossless) {
        static_assert(std::is_base_of<concepts::DecompositionInterface_OMP<T, int, N>, Decomposition>::value,
                      "must implement the frontend interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
#ifdef SZ3_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto [local_quant_inds, local_quant_index] = decomposition.compress(conf, data);
#ifdef SZ3_PRINT_TIMINGS
        timer.stop("cmp interp");
        timer.start();
#endif
        int quant_inds_size;// = std::accumulate(conf.dims.begin(), conf.dims.end(), 1, std::multiplies<size_t>());
        // size_t nThreads = omp_get_max_threads();
//         std::vector<size_t> offset(nThreads);
        // size_t cur = 0;
        // for (size_t t = 0; t < nThreads; ++t) {
        //     cur += local_quant_index[t].value;
        //     // std::cout<<"thread "<<t<<" has "<< local_quant_index[t].value<<" quantization indices."<<std::endl;
        // }
        quant_inds_size = local_quant_index[0].value;
#ifdef SZ3_PRINT_TIMINGS
        timer.stop("cal quant_inds_size");
        timer.start();
#endif
        // std::ofstream out("quant1.bin", std::ios::binary);
        // out.write(
        //     reinterpret_cast<const char*>(local_quant_inds[0]),
        //     local_quant_index[0].value * sizeof(int)
        // );
        // if (decomposition.get_out_range().first != 0) {
        //     throw std::runtime_error("The output range of the decomposition must start from 0 for this compressor");
        // }

        auto cmpDataPos = cmpData;
        

        #ifdef _OPENMP
        auto default_nthreads = omp_get_max_threads();
        //std::cout<<default_nthreads<<" "<<quant_inds_size<<std::endl;
        auto best_num_threads = default_nthreads;// std::min(default_nthreads, static_cast<int>(quant_inds_size / (1u<<16)));
        //std::cout<<best_num_threads<<std::endl;
        if (best_num_threads > 1) {
            // size_t cur_bufferSize = std::max<size_t>(1000, 2 * 1.2 * sizeof(T) * local_quant_index[0].value * best_num_threads);
            // uchar* buffer = new (std::align_val_t(64)) uchar[cur_bufferSize * best_num_threads];
            omp_set_num_threads(best_num_threads);
            //uchar * offset_block_pos = buffer_pos + sizeof(int);
            //size_t bins_per_thread;
            // auto quant_inds_data = quant_inds;
            std::vector<size_t>block_byte_offsets;
            // size_t offset_quant_inds_size;
            // size_t total_huffman_size = 0;
            int nthreads = best_num_threads;
            size_t offset_chunk_size = nthreads * sizeof (size_t);
            size_t total_zstd_size = 0;
            block_byte_offsets.resize(nthreads);
            write<int>(nthreads, cmpDataPos);
            // Timer timer1(true);
            #pragma omp parallel
            {
                // #pragma omp single 
                // {
                //     Timer timer1(true);
                // }   
                Encoder cur_encoder;
                auto tid = omp_get_thread_num();
                // auto temp_buffer_pos0 = buffer_pos + tid * sizeof(int);
                // decomposition.save3(temp_buffer_pos0, tid);
                // write<int>(local_quant_index[tid].value, temp_buffer_pos0);
                size_t cur_bufferSize = std::max<size_t>(1000, 1.2 * sizeof(T) * local_quant_index[tid].value);
                auto cur_buffer = static_cast<uchar *>(malloc(cur_bufferSize)); 
                // uchar* cur_buffer = new (std::align_val_t(64)) uchar[cur_bufferSize];
                // auto cur_buffer = buffer + tid * cur_bufferSize;
                auto cur_buffer_pos = cur_buffer;
                if(tid == best_num_threads - 1)
                    decomposition.save2(cur_buffer_pos);

                decomposition.save3(cur_buffer_pos, tid);
                // #pragma omp single 
                // {
                //     timer1.stop("encode thread ");
                //     timer1.start();
                // }   
                write<int>(local_quant_index[tid].value, cur_buffer_pos);
                //  #pragma omp single 
                // {
                //     timer1.stop("write ");
                //     timer1.start();
                // }  
                cur_encoder.preprocess_encode(local_quant_inds[tid], local_quant_index[tid].value, 
                    decomposition.get_out_range().second, decomposition.frequencyList[tid]);
                // #pragma omp single 
                // {
                //     timer1.stop("preprocess ");
                //     timer1.start();
                // }   
                cur_encoder.save(cur_buffer_pos);
                //#pragma omp critical
                //std::cout<<tid<<" "<<cur_buffer_pos-cur_buffer <<std::endl;
                cur_encoder.encode(local_quant_inds[tid], local_quant_index[tid].value, cur_buffer_pos);
                //std::cout<<tid<<" "<<encode_length<<std::endl;
                cur_encoder.postprocess_encode();
                // #pragma omp single 
                // {
                //     timer1.stop("encode ");
                //     timer1.start();
                // }  
                auto cur_outSize = cur_buffer_pos - cur_buffer;

                size_t lossless_bufferSize = std::max<size_t>(1000, 1.2 * sizeof(uchar) * cur_outSize);
                auto lossless_buffer = static_cast<uchar *>(malloc(lossless_bufferSize)); 
                auto lossless_buffer_pos = lossless_buffer;
                // #pragma omp single 
                // {
                //     timer1.stop("lossless init ");
                //     timer1.start();
                // }  
                auto lossless_outSize = lossless.compress(cur_buffer, cur_outSize, lossless_buffer_pos, lossless_bufferSize);

                block_byte_offsets[tid] = lossless_outSize;
                // #pragma omp single 
                // {
                //     timer1.stop("lossless compress");
                //     timer1.start();
                // }  
                #pragma omp barrier
                #pragma omp single
                {
                    size_t prefix_sum = 0;
                    
                    for (int i = 0; i < nthreads; i++){
                        auto tmp = block_byte_offsets[i];
                        block_byte_offsets[i] = prefix_sum;
                        prefix_sum =  prefix_sum + tmp;
                        // std::cout << "block_byte_offsets[" << i << "] " << block_byte_offsets[i] << std::endl;

                    }
                    total_zstd_size =  prefix_sum;
                    // std::cout << "total_zstd_size " << total_zstd_size << std::endl;
                }
                // #pragma omp single 
                // {
                //     timer1.stop("lossless prefix");
                //     timer1.start();
                // } 
                auto temp_cmpData_pos = cmpDataPos + tid * sizeof(size_t);
                write<size_t>(block_byte_offsets[tid],temp_cmpData_pos);
                temp_cmpData_pos = cmpDataPos + offset_chunk_size + block_byte_offsets[tid];
                write<uchar>(lossless_buffer, lossless_outSize,  temp_cmpData_pos);

                free(lossless_buffer);
                free(cur_buffer);
                // #pragma omp single 
                // {
                //     timer1.stop("last");
                // } 

            }
            // free(buffer);
            omp_set_num_threads(default_nthreads);
            cmpDataPos += offset_chunk_size + total_zstd_size;
        }
            
        else{
            size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + sizeof(T) * quant_inds_size));

            auto buffer = static_cast<uchar *>(malloc(bufferSize));
            uchar *buffer_pos = buffer;

            write<int>(1, cmpDataPos); //1 thread
            decomposition.save2(buffer_pos);
            decomposition.save3(buffer_pos, 0);
            write<size_t>(quant_inds_size, buffer_pos);
            // Timer timer1(true);
            encoder.preprocess_encode(local_quant_inds[0], local_quant_index[0].value, 
                decomposition.get_out_range().second, decomposition.frequencyList[0]);
                    //                         timer1.stop("preprocess ");
                    // timer1.start();
            encoder.save(buffer_pos);
            encoder.encode(local_quant_inds[0], local_quant_index[0].value, buffer_pos);
            //   timer1.stop("encode ");
            encoder.postprocess_encode();
            auto currentSize = buffer_pos - buffer;
            cmpCap-=cmpDataPos-cmpData;

            write<size_t>(0, cmpDataPos); //offset = 0;
            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        }


        #else
            size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + sizeof(T) * quant_inds_size));

            auto buffer = static_cast<uchar *>(malloc(bufferSize));
            uchar *buffer_pos = buffer;

            write<int>(1, cmpDataPos); //1 thread
            decomposition.save2(buffer_pos);
            decomposition.save3(buffer_pos, 0);
            write<size_t>(quant_inds_size, buffer_pos);
            encoder.preprocess_encode(local_quant_inds[0], local_quant_index[0].value, decomposition.get_out_range().second, decomposition.frequencyList[0]);
            encoder.save(buffer_pos);
            encoder.encode(local_quant_inds[0], local_quant_index[0].value, buffer_pos);
            encoder.postprocess_encode();
            auto currentSize = buffer_pos - buffer;
            cmpCap-=cmpDataPos-cmpData;

            write<size_t>(0, cmpDataPos); //offset = 0;
            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        #endif
#ifdef SZ3_PRINT_TIMINGS
        timer.stop("huff");
        timer.start();
#endif

// #ifdef SZ3_PRINT_TIMINGS
//          timer.stop("zstd");
// #endif
        return cmpDataPos-cmpData;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {
#ifdef SZ3_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto cmpDataPos = cmpData;
        int compression_thread_num = 0;


        //lossless.decompress(cmpData, cmpSize, buffer, bufferSize);
        // read(huffSize, cmpDataPos);
        read(compression_thread_num, cmpDataPos);
        int** local_quant_inds = new int*[compression_thread_num];
        int* local_quant_index = new int[compression_thread_num];
        decomposition.init_local_unpred(compression_thread_num);
        if(compression_thread_num <= 1){
            uchar *buffer = nullptr;
            size_t bufferSize = 0;
            size_t offset;
            read(offset, cmpDataPos);
            cmpSize -= cmpDataPos - cmpData; 
            lossless.decompress(cmpDataPos, cmpSize, buffer, bufferSize);
            size_t quant_inds_size = 0;
            uchar const *bufferPos = buffer;
            decomposition.load2(bufferPos, bufferSize);
            decomposition.load3(bufferPos, bufferSize, 0);
            read(quant_inds_size, bufferPos);
            local_quant_index[0] = quant_inds_size;
            // std::cout << "quant_inds_size: " << quant_inds_size << std::endl;
            encoder.load(bufferPos, bufferSize);
            local_quant_inds[0] = encoder.decode(bufferPos, quant_inds_size);
            encoder.postprocess_decode();
            // decomposition.decompress(conf, local_quant_inds, decData);
            // return decData;
        }
        else{
            std::vector<size_t>block_byte_offsets(compression_thread_num);
            std::vector<size_t>output_block_byte_offsets(compression_thread_num);

            read(block_byte_offsets.data(),compression_thread_num,cmpDataPos);
            cmpSize -= cmpDataPos - cmpData;
            // #ifdef _OPENMP
            // #pragma omp parallel for 
            // #endif
            // for(int tid=0; tid < compression_thread_num;tid++){
            #pragma omp parallel num_threads(compression_thread_num)
            {
                auto tid = omp_get_thread_num();
                //size_t start_idx = (static_cast<size_t>(tid) * huffSize) / static_cast<size_t>(compression_thread_num), cur_len = (static_cast<size_t>(tid+1) * huffSize) / static_cast<size_t>(compression_thread_num) - start_idx;
                size_t block_byte_offset = block_byte_offsets[tid];
                //std::cout<<tid<<" "<<block_byte_offset<<std::endl;

                size_t block_byte_length = (tid != compression_thread_num-1)? block_byte_offsets[tid + 1] - block_byte_offset : cmpSize - block_byte_offset;
               // #pragma omp critical
                //std::cout<<"tid: "<<tid<<", prefix: "<<block_byte_offset<<std::endl;
                auto temp_cmp_pos = cmpDataPos + block_byte_offset;
                uchar* cur_buffer = nullptr;
                size_t cur_bufferSize = 0;
                lossless.decompress(temp_cmp_pos, block_byte_length, cur_buffer, cur_bufferSize);
                uchar const* cur_buffer_pos = cur_buffer;
                if(tid == compression_thread_num - 1)
                    decomposition.load2(cur_buffer_pos, cur_bufferSize);
                #pragma omp barrier
                decomposition.load3(cur_buffer_pos, cur_bufferSize, tid);
                read<int>(local_quant_index[tid], cur_buffer_pos);

                Encoder cur_encoder;
                cur_encoder.load(cur_buffer_pos, cur_bufferSize);
                // std::cout << "Thread " << tid << " encoded size: " << local_quant_index[tid] << std::endl;
                local_quant_inds[tid] = cur_encoder.decode(cur_buffer_pos, local_quant_index[tid]);   
                cur_encoder.postprocess_decode();   
                free(cur_buffer);
            }
        }

#ifdef SZ3_PRINT_TIMINGS
        timer.stop("decmp zstd");
        timer.start();
#endif

        decomposition.decompress(conf, local_quant_inds, decData);
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
std::shared_ptr<SZGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>> make_compressor_sz_generic_omp(
    Decomposition decomposition, Encoder encoder, Lossless lossless) {
    return std::make_shared<SZGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>>(decomposition, encoder,
                                                                                         lossless);
}

}  // namespace SZ3
#endif