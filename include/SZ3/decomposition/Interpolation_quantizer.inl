
namespace SZ3 {
#ifdef __AVX2__
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);
        
            for (; i + 1  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                // predict
                __m256 va = _mm256_loadu_ps(buf + i );
                __m256 vb = _mm256_loadu_ps(buf + i + 1);
                __m256 sum = _mm256_add_ps(va, vb);                        
                sum = _mm256_mul_ps(sum, factor);        

                // quantize
                size_t start = (i << 1) + 1;
                // i = k / 2;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[8];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];
                    ori[4] = data[base + (offsetx2 << 2)];
                    ori[5] = data[base + 5 * offsetx2];
                    ori[6] = data[base + 6 * offsetx2];
                    ori[7] = data[base + 7 * offsetx2];


                    __m256 ori_avx = _mm256_loadu_ps(ori);
                    __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
                    float tmp[8];
                    quantize_1D_float(sum, ori_avx, quant_avx, tmp);
                    int quant_vals[8];
                    __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(quant_inds + quant_index),
                        quant_avx_i
                    );
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m256i quant_avx_i = _mm256_loadu_si256(
                        reinterpret_cast<__m256i*>(quant_inds + quant_index));
                    int quant_vals[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
                    quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);
                    
                    __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
                    decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);
                    
                    __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
                    decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

                     __m256 decompressed = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                        _mm256_cvtpd_ps(decompressed_high), 1);
                    decompressed = _mm256_add_ps(decompressed, sum);
                    float tmp[8];
                    _mm256_storeu_ps(tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 1< odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);
            
            for (; i + 1 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                __m256d va = _mm256_loadu_pd(buf + i);
                __m256d vb = _mm256_loadu_pd(buf + i + 1);

                __m256d sum = _mm256_add_pd(va, vb);   
                sum = _mm256_mul_pd(sum, factor);    

                size_t start = (i << 1) + 1;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[4];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];

                    __m256d ori_avx = _mm256_loadu_pd(ori);
                    __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
                    T tmp[4];
                    quantize_1D_double(sum, ori_avx, quant_avx, tmp);

                    int quant_vals[4];
                    __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);
                    
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
                    size_t j = 0;
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(quant_inds + quant_index),
                        quant_avx_i
                    );
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m128i quant_avx_i = _mm_loadu_si128(
                        reinterpret_cast<__m128i*>(quant_inds + quant_index));
                    int quant_vals[4];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
                    quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

                    __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i), 
                                            ebx2_avx, sum);
                    T tmp[4];
                    _mm256_storeu_pd(tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;  
                }

            }
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else 
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_func(cur_ij_offset + last * offset , data[last * offset], pred_edge);
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        
        T pred_first; 
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else 
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_func(cur_ij_offset + offset , data[offset], pred_first);

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 nine  = _mm256_set1_ps(9.0f);
            const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

            for (; i + 3  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                __m256 va = _mm256_loadu_ps(buf + i);
                __m256 vb = _mm256_loadu_ps(buf + i + 1);
                __m256 vc = _mm256_loadu_ps(buf + i + 2);
                __m256 vd = _mm256_loadu_ps(buf + i + 3);

                 __m256 sum = _mm256_add_ps(vb, vc); 
                 sum = _mm256_mul_ps(sum, nine); 
                 sum = _mm256_sub_ps(sum, va); 
                sum = _mm256_sub_ps(sum, vd);                       
                sum = _mm256_mul_ps(sum, factor);        

                size_t start = (i << 1) + 3;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[8];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];
                    ori[4] = data[base + (offsetx2 << 2)];
                    ori[5] = data[base + 5 * offsetx2];
                    ori[6] = data[base + 6 * offsetx2];
                    ori[7] = data[base + 7 * offsetx2];


                    __m256 ori_avx = _mm256_loadu_ps(ori);
                    __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
                    float tmp[8];
                    quantize_1D_float(sum, ori_avx, quant_avx, tmp);

                    int quant_vals[8];
                    __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(quant_inds + quant_index),
                        quant_avx_i
                    );
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m256i quant_avx_i = _mm256_loadu_si256(
                        reinterpret_cast<__m256i*>(quant_inds + quant_index));
                    int quant_vals[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
                    quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);
                    
                    __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
                    decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);
                    
                    __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
                    decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

                     __m256 decompressed = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                        _mm256_cvtpd_ps(decompressed_high), 1);
                    decompressed = _mm256_add_ps(decompressed, sum);
                    float tmp[8];
                    _mm256_storeu_ps(tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;              
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d nine  = _mm256_set1_pd(9.0);
            const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

            for (; i + 3 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                __m256d va = _mm256_loadu_pd(buf + i);
                __m256d vb = _mm256_loadu_pd(buf + i + 1);
                __m256d vc = _mm256_loadu_pd(buf + i + 2);
                __m256d vd = _mm256_loadu_pd(buf + i + 3);

                __m256d sum = _mm256_add_pd(vb, vc); 
                 sum = _mm256_mul_pd(sum, nine); 
                 sum = _mm256_sub_pd(sum, va); 
                sum = _mm256_sub_pd(sum, vd); 
                sum = _mm256_mul_pd(sum, factor);    
                // _mm256_storeu_pd(p + i + 1, sum);
                size_t start = (i << 1) + 3;
                // T pred[4];
                // _mm256_storeu_pd(pred, sum);

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[4];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];

                    __m256d ori_avx = _mm256_loadu_pd(ori);
                    __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
                    T tmp[4];
                    quantize_1D_double(sum, ori_avx, quant_avx, tmp);

                    int quant_vals[4];
                    __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);
                    
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(quant_inds + quant_index),
                        quant_avx_i
                    );
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m128i quant_avx_i = _mm_loadu_si128(
                    reinterpret_cast<__m128i*>(quant_inds + quant_index));
                    int quant_vals[4];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
                    quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

                    __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i), 
                                            ebx2_avx, sum);
                    T tmp[4];
                    _mm256_storeu_pd(tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;  
                }
            }
        }

        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1] 
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                

            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);

            for (; i  < len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                
                __m256 sum = _mm256_add_ps(va, vb); 
                sum = _mm256_mul_ps(sum, factor);        
                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);

            for (; i  < len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);

                __m256d sum = _mm256_add_pd(va, vb);                       
                sum = _mm256_mul_pd(sum, factor);    
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }

    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 nine  = _mm256_set1_ps(9.0f);
            const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

            for (; i  < len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                __m256 vd = _mm256_loadu_ps(d + i);

                 __m256 sum = _mm256_add_ps(vb, vc); 
                 sum = _mm256_mul_ps(sum, nine); 
                 sum = _mm256_sub_ps(sum, va); 
                sum = _mm256_sub_ps(sum, vd); 
                sum = _mm256_mul_ps(sum, factor);        

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
            
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d nine  = _mm256_set1_pd(9.0);
            const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

            for (; i  < len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d vd = _mm256_loadu_pd(d + i);

                __m256d sum = _mm256_add_pd(vb, vc); 
                 sum = _mm256_mul_pd(sum, nine); 
                 sum = _mm256_sub_pd(sum, va); 
                sum = _mm256_sub_pd(sum, vd); 

                sum = _mm256_mul_pd(sum, factor);    
                // _mm256_storeu_pd(p + i, sum);
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }

    }
    
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_equal_and_quantize(const T * a, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            for (; i  < len; i += step) {
                __m256 sum = _mm256_loadu_ps(a + i);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
            
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            for (; i  < len; i += step) {
                __m256d sum = _mm256_loadu_pd(a + i); 
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);
            const __m256 three = _mm256_set1_ps(3.0f);
            for (; i  < len; i += step) {
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 va = _mm256_loadu_ps(a + i);
                vb= _mm256_mul_ps(vb, three);
                __m256 sum = _mm256_sub_ps(vb, va); 
                sum = _mm256_mul_ps(sum, factor);        
                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);
            const __m256d three = _mm256_set1_pd(3.0);
            for (; i  < len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                va = _mm256_mul_pd(va, three);
                __m256d sum = _mm256_sub_pd(vb, va);                       
                sum = _mm256_mul_pd(sum, factor);    
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.125f);
            const __m256 six = _mm256_set1_ps(6.0f);
            const __m256 three = _mm256_set1_ps(3.0f);

            for (; i  < len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                va = _mm256_mul_ps(va, three);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                vb = _mm256_fmsub_ps(vb, six, vc);
                __m256 sum = _mm256_add_ps(va, vb); 
                sum = _mm256_mul_ps(sum, factor);        
                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.125);
            const __m256d six = _mm256_set1_pd(6.0);
            const __m256d three = _mm256_set1_pd(3.0);

            for (; i  < len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                va = _mm256_mul_pd(va, three);
                __m256d vc = _mm256_loadu_pd(c + i);
                vb = _mm256_fmsub_pd(vb, six, vc);
                __m256d sum = _mm256_add_pd(va, vb); 
                sum = _mm256_mul_pd(sum, factor);    
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }
      
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.125f);
            const __m256 six = _mm256_set1_ps(6.0f);
            const __m256 three = _mm256_set1_ps(3.0f);

            for (; i  < len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                vc = _mm256_mul_ps(vc, three);
                __m256 vb = _mm256_loadu_ps(b + i);
                
                vb = _mm256_fmsub_ps(vb, six, va);
                __m256 sum = _mm256_add_ps(vc, vb); 
                sum = _mm256_mul_ps(sum, factor);        
                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step>(sum, i, data, offset, len);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.125);
            const __m256d six = _mm256_set1_pd(6.0);
            const __m256d three = _mm256_set1_pd(3.0);

            for (; i  < len; i += step) {
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d va = _mm256_loadu_pd(a + i);
                vc = _mm256_mul_pd(vc, three);
                __m256d vb = _mm256_loadu_pd(b + i);
                
                vb = _mm256_fmsub_pd(vb, six, va);
                __m256d sum = _mm256_add_pd(vc, vb); 
                sum = _mm256_mul_pd(sum, factor);     
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step>(sum, i, data, offset, len);
            }
        }
      
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_1D_float (__m256& sum, __m256& ori_avx, __m256& quant_avx, T tmp[8]) {

        __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
        quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        __m256d mask_low = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
        );
        quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

        __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
        quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d mask_high = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
        );
        quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);
        
        // dequantization for decompression
        __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
                                _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
        __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
                                _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

        quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
                        _mm256_cvtpd_ps(quant_avx_high), 1);

        __m256 decompressed = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
            _mm256_cvtpd_ps(decompressed_high), 1);

        __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);
        quant_avx = _mm256_add_ps(quant_avx, radius_avx_f);

        _mm256_storeu_ps(tmp, decompressed);
        
        __m256 mask = _mm256_and_ps(
                _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
                _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
        );
        
        quant_avx = _mm256_blendv_ps(zero_avx_f, quant_avx, mask);
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_1D_double (__m256d& sum, __m256d& ori_avx, __m256d& quant_avx, T tmp[4]) {
        quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        __m256d mask = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
        );
        quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

        __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
        _mm256_storeu_pd(tmp, decompressed);
        __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

        mask = _mm256_and_pd(
                _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
                _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
        );
        quant_avx = _mm256_add_pd(quant_avx,  radius_avx);
        quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);
    }
    
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, int step, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_float (__m256& sum, size_t& start, T*& data, size_t& offset, size_t& len) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[8] = {
                data[(start) * offset],
                data[(start + 1) * offset],
                data[(start + 2) * offset],
                data[(start + 3) * offset],
                data[(start + 4) * offset],
                data[(start + 5) * offset],
                data[(start + 6) * offset],
                data[(start + 7) * offset]
            };
            __m256 ori_avx = _mm256_loadu_ps(ori);
            __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
            // calculate quantization code
            __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
            quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            __m256d mask_low = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
            );
            quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

            __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
            quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d mask_high = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
            );
            quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);
            
            // dequantization for decompression
            __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
                                    _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
            __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
                                    _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

            quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
                            _mm256_cvtpd_ps(quant_avx_high), 1);

            __m256 decompressed = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                _mm256_cvtpd_ps(decompressed_high), 1);

            __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);
            quant_avx = _mm256_add_ps(quant_avx, radius_avx_f);
            float tmp[8];
            _mm256_storeu_ps(tmp, decompressed);
            
            __m256 mask = _mm256_and_ps(
                    _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
                    _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
            );
            
            quant_avx = _mm256_blendv_ps(zero_avx_f, quant_avx, mask);
            // float verify[8];
            // _mm256_storeu_ps(verify, quant_avx);
            int quant_vals[8];
            __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
            size_t j = 0;
            
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else
                    quantizer.force_save_unpred(ori[j]);
                ++frequency[quant_vals[j]];
            }

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(quant_inds + quant_index),
                quant_avx_i
            );
            quant_index += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            __m256i quant_avx_i = _mm256_loadu_si256(
                reinterpret_cast<__m256i*>(quant_inds + quant_index));
            int quant_vals[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
            quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);
            
            __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
            decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);
            
            __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
            decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

                __m256 decompressed = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                _mm256_cvtpd_ps(decompressed_high), 1);
            decompressed = _mm256_add_ps(decompressed, sum);
            float tmp[8];
            _mm256_storeu_ps(tmp, decompressed);
            
            size_t j = 0;
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else 
                    data[(start + j) * offset] = quantizer.recover_unpred();
            }
            quant_index += j;
        }
    }
    
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, int step, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_double (__m256d& sum, size_t& start, T*& data, size_t& offset, size_t& len) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[4] = {
                data[(start) * offset],
                data[(start + 1) * offset],
                data[(start + 2) * offset],
                data[(start + 3) * offset],
            };
            __m256d ori_avx = _mm256_loadu_pd(ori);
            __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
            quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            __m256d mask = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
            );
            quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

            __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
            T tmp[4];
            _mm256_storeu_pd(tmp, decompressed);
            __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

            mask = _mm256_and_pd(
                    _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
                    _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
            );
            quant_avx = _mm256_add_pd(quant_avx,  radius_avx);
            quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

            int quant_vals[4];
            __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);
            
            _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
            size_t j = 0;
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0)
                    data[(start + j) * offset] = tmp[j];
                else 
                    quantizer.force_save_unpred(ori[j]);
                ++frequency[quant_vals[j]];
            }
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(quant_inds + quant_index),
                quant_avx_i
            );
            quant_index += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            __m128i quant_avx_i = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(quant_inds + quant_index));
            int quant_vals[4];
            _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
            quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

            __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i), 
                                    ebx2_avx, sum);
            T tmp[4];
            _mm256_storeu_pd(tmp, decompressed);
            
            size_t j = 0;
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else
                    data[(start + j) * offset] = quantizer.recover_unpred();
            }
            quant_index += j;  
        }
    }
#elif defined(__ARM_FEATURE_SVE2) 
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            // const size_t step = AVX_256_parallelism;
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();
            for (; i + 1  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                // predict
                // svbool_t pg = svwhilelt_b32(i, even_len - 1);

                svfloat32_t va = svld1(pg, &buf[i]);
                svfloat32_t vb = svld1(pg, &buf[i + 1]);

                svfloat32_t sum = svadd_f32_x(pg, va, vb);
                sum = svmul_n_f32_x(pg, sum, 0.5f);
                // quantize
                size_t start = (i << 1) + 1;
                // i = k / 2;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat32_t ori_sve = svld1(pg, ori);
                    svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];

                    quantize_1D_float (sum, ori_sve, quant_sve, tmp, pg, pg64);

                    svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
                    svst1(pg, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    svst1(pg, quant_inds + quant_index, quant_sve_i);
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint32_t quant_sve_i = svld1_s32(pg, quant_inds + quant_index);
                    int quant_vals[step];
                    svst1(pg, quant_vals, quant_sve_i);
                    quant_sve_i = svsub_n_s32_x(pg, quant_sve_i, radius);
                    
                    svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
                    svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));
                    
                    decompressed_even_f64 = svmul_n_f64_x(pg64, decompressed_even_f64, real_ebx2);
                    decompressed_odd_f64 = svmul_n_f64_x(pg64, decompressed_odd_f64, real_ebx2);
                    
                    svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
                    decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);
                    
                    decompressed = svadd_f32_x(pg, decompressed, sum);

                    T tmp[step];
                    svst1_f32(pg, tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 1< odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();
            for (; i + 1 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                svfloat64_t va = svld1(pg64, &buf[i]);
                svfloat64_t vb = svld1(pg64, &buf[i + 1]);

                svfloat64_t sum = svadd_f64_x(pg64, va, vb);
                sum = svmul_n_f64_x(pg64, sum, 0.5);

                size_t start = (i << 1) + 1;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat64_t ori_sve = svld1(pg64, ori);
                    svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];
                    quantize_1D_double(sum, ori_sve, quant_sve, tmp, pg64);
                    
                    svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);
                    
                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    svst1w_s64(pg64, quant_inds + quant_index, quant_sve_i);
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint64_t quant_sve_i = svld1sw_s64(pg64, quant_inds + quant_index);
                    int quant_vals[step];
                    svst1w_s64(pg64, quant_vals, quant_sve_i);
                    quant_sve_i = svsub_n_s64_x(pg64, quant_sve_i, radius);

                    svfloat64_t decompressed = svmla_f64_x(pg64, sum, 
                            svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
                    T tmp[step];
                    svst1_f64(pg64, tmp, decompressed);
                    size_t j = 0;
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;  
                }

            }
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else 
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_func(cur_ij_offset + last * offset , data[last * offset], pred_edge);
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        
        T pred_first; 
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else 
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_func(cur_ij_offset + offset , data[offset], pred_first);

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();
            for (; i + 3  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!       

                svfloat32_t va = svld1(pg, &buf[i]);
                svfloat32_t vb = svld1(pg, &buf[i + 1]);
                svfloat32_t vc = svld1(pg, &buf[i + 2]);
                
                svfloat32_t sum = svadd_f32_x(pg, vb, vc);
                sum = svmul_n_f32_x(pg, sum, 9.0f);

                svfloat32_t vd = svld1(pg, &buf[i + 3]);
                sum = svsub_f32_x(pg, sum, va);
                sum = svsub_f32_x(pg, sum, vd);
                sum = svmul_n_f32_x(pg, sum, 0.0625f);

                size_t start = (i << 1) + 3;

                if constexpr (CompMode == COMPMODE::COMP) {
                     T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat32_t ori_sve = svld1(pg, ori);
                    svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];

                    quantize_1D_float (sum, ori_sve, quant_sve, tmp, pg, pg64);

                    svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
                    svst1(pg, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 3 < odd_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    svst1(pg, quant_inds + quant_index, quant_sve_i);
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint32_t quant_sve_i = svld1_s32(pg, quant_inds + quant_index);
                    int quant_vals[step];
                    svst1(pg, quant_vals, quant_sve_i);
                    quant_sve_i = svsub_n_s32_x(pg, quant_sve_i, radius);
                    
                    svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
                    svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));
                    
                    decompressed_even_f64 = svmul_n_f64_x(pg64, decompressed_even_f64, real_ebx2);
                    decompressed_odd_f64 = svmul_n_f64_x(pg64, decompressed_odd_f64, real_ebx2);
                    
                    svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
                    decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);
                    
                    decompressed = svadd_f32_x(pg, decompressed, sum);

                    T tmp[step];
                    svst1_f32(pg, tmp, decompressed);
                    
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else 
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();
            for (; i + 3 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!

                svfloat64_t va = svld1(pg64, &buf[i]);
                svfloat64_t vb = svld1(pg64, &buf[i + 1]);
                svfloat64_t vc = svld1(pg64, &buf[i + 2]);
                
                svfloat64_t sum = svadd_f64_x(pg64, vb, vc);
                sum = svmul_n_f64_x(pg64, sum, 9.0);

                svfloat64_t vd = svld1(pg64, &buf[i + 3]);
                sum = svsub_f64_x(pg64, sum, va);
                sum = svsub_f64_x(pg64, sum, vd);
                sum = svmul_n_f64_x(pg64, sum, 0.0625);

                size_t start = (i << 1) + 3;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat64_t ori_sve = svld1(pg64, ori);
                    svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];
                    quantize_1D_double(sum, ori_sve, quant_sve, tmp, pg64);
                    
                    svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);
                    
                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 3 < odd_len; ++j) {
                        if (quant_vals[j] != 0)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.force_save_unpred(ori[j]);
                        ++frequency[quant_vals[j]];
                    }
                    svst1w_s64(pg64, quant_inds + quant_index, quant_sve_i);
                    quant_index += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint64_t quant_sve_i = svld1sw_s64(pg64, quant_inds + quant_index);
                    int quant_vals[step];
                    svst1w_s64(pg64, quant_vals, quant_sve_i);
                    quant_sve_i = svsub_n_s64_x(pg64, quant_sve_i, radius);

                    svfloat64_t decompressed = svmla_f64_x(pg64, sum, 
                            svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
                    T tmp[step];
                    svst1_f64(pg64, tmp, decompressed);
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < odd_len; ++j) {
                        if (quant_vals[j] != 0) 
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred();
                    }
                    quant_index += j;  
                }

            }
        }
        
        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1] 
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                

            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();
            for (; i  < len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);

                svfloat32_t sum = svadd_f32_x(pg, va, vb);
                sum = svmul_n_f32_x(pg, sum, 0.5f);   
                
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);

                svfloat64_t sum = svadd_f64_x(pg64, va, vb);
                sum = svmul_n_f64_x(pg64, sum, 0.5);
                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);
                
                svfloat32_t sum = svadd_f32_x(pg, vb, vc);
                sum = svmul_n_f32_x(pg, sum, 9.0f);

                svfloat32_t vd = svld1(pg, &d[i]);
                
                sum = svsub_f32_x(pg, sum, va);
                sum = svsub_f32_x(pg, sum, vd);
                sum = svmul_n_f32_x(pg, sum, 0.0625f);
                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);
                
                svfloat64_t sum = svadd_f64_x(pg64, vb, vc);
                sum = svmul_n_f64_x(pg64, sum, 9.0);

                svfloat64_t vd = svld1(pg64, &d[i]);
                
                sum = svsub_f64_x(pg64, sum, va);
                sum = svsub_f64_x(pg64, sum, vd);
                sum = svmul_n_f64_x(pg64, sum, 0.0625);

                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }
        }
    }
    
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_equal_and_quantize(const T * a, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat32_t sum = svld1(pg, &a[i]);                
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat64_t sum = svld1(pg64, &a[i]);                
                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
  
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();
            for (; i  < len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);  
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svmul_n_f32_x(pg, vb, 1.5f);
                svfloat32_t sum = svmls_n_f32_x(pg, vb, va, 0.5f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();
            
            for (; i  < len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);  
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svmul_n_f64_x(pg64, vb, 1.5);
                svfloat64_t sum = svmls_n_f64_x(pg64, vb, va, 0.5);
                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);
                vb = svnmls_n_f32_x(pg, vc, vb, 6.0f);
                svfloat32_t va = svld1(pg, &a[i]);  
                svfloat32_t sum = svmla_n_f32_x(pg, vb, va, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();
            
            for (; i  < len; i += step) {
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);
                vb = svnmls_n_f64_x(pg64, vc, vb, 6.0);
                svfloat64_t va = svld1(pg64, &a[i]);  
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, va, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }

        }      
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad2_and_quantize (const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg = svptrue_b32();
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svnmls_n_f32_x(pg, va, vb, 6.0f);

                svfloat32_t vc = svld1(pg, &c[i]);
                svfloat32_t sum = svmla_n_f32_x(pg, vb, vc, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode>(sum, i, data, offset, len, step, pg, pg64);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            const svbool_t pg64 = svptrue_b64();

            for (; i  < len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svnmls_n_f64_x(pg64, va, vb, 6.0);

                svfloat64_t vc = svld1(pg64, &c[i]);
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, vc, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                // _mm256_storeu_ps(p + i, sum);
                quantize_double<CompMode>(sum, i, data, offset, len, step, pg64);
            }
        }   
      
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_1D_float (
        svfloat32_t& sum, svfloat32_t& ori_sve, svfloat32_t& quant_sve, T* tmp, const svbool_t& pg, const svbool_t& pg64) {
            
            svfloat64_t quant_even_f64 = svcvt_f64_f32_x(pg64, quant_sve);
            svfloat64_t quant_odd_f64  = svcvtlt_f64_f32_x(pg64, quant_sve);
            quant_even_f64 = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_even_f64, real_ebx2_r));
            quant_odd_f64  = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_odd_f64, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_even_f64, -radius); // val > -radius
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_even_f64,  radius); // val < +radius
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_even_f64 = svsel_f64(pg_in_range, quant_even_f64, svdup_n_f64(0.0));

            svbool_t pg_gt_neg_o = svcmpgt_n_f64(pg64, quant_odd_f64, -radius);
            svbool_t pg_lt_pos_o = svcmplt_n_f64(pg64, quant_odd_f64,  radius);
            svbool_t pg_in_range_o = svand_b_z(pg64, pg_gt_neg_o, pg_lt_pos_o);
            quant_odd_f64 = svsel_f64(pg_in_range_o, quant_odd_f64, svdup_n_f64(0.0));

            // dequantization for decompression

            svfloat64_t decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum),
                    quant_even_f64, svdup_f64(real_ebx2));
            svfloat64_t decompressed_odd_f64  = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum),
                    quant_odd_f64, svdup_f64(real_ebx2));

            // svfloat32_t even_f32 = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            // svfloat32_t odd_f32  = svcvtlt_f32_f64_x(pg64, decompressed_odd_f64);
            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);
            
            svst1_f32(pg, tmp, decompressed);

            // even_f32 = svcvt_f32_f64_x(pg64, quant_even_f64);
            // odd_f32  = svcvt_f32_f64_x(pg64, quant_odd_f64);
            // quant_sve = svzip1_f32(svuzp1_f32(even_f32, even_f32), svuzp1_f32(odd_f32, odd_f32));

            quant_sve = svcvt_f32_f64_x(pg64, quant_even_f64);
            quant_sve = svcvtnt_f32_f64_x(quant_sve, pg64, quant_odd_f64);

            svfloat32_t err_dequan = svsub_f32_x(pg, decompressed, ori_sve);
            quant_sve = svadd_n_f32_x(pg, quant_sve, radius);

            pg_in_range = svand_b_z(pg, svcmpge_n_f32(pg, err_dequan, -real_eb), svcmple_n_f32(pg, err_dequan, real_eb));
            quant_sve = svsel_f32(pg_in_range, quant_sve, svdup_n_f32(0.0));
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_1D_double (
        svfloat64_t& sum, svfloat64_t& ori_sve, svfloat64_t& quant_sve, T* tmp, const svbool_t& pg64) {
            
        quant_sve = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_sve, real_ebx2_r));

        svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_sve, -radius);
        svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_sve,  radius);
        svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
        quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));

        svfloat64_t decompressed = svmla_f64_x(pg64, sum, quant_sve, svdup_f64(real_ebx2));
        svst1_f64(pg64, tmp, decompressed);
        svfloat64_t err_dequan = svsub_f64_x(pg64, decompressed, ori_sve);

        quant_sve = svadd_n_f64_x(pg64, quant_sve, radius);
        pg_in_range = svand_b_z(pg64, svcmpge_n_f64(pg64, err_dequan, -real_eb), svcmple_n_f64(pg64, err_dequan, real_eb));
        quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_float (svfloat32_t& sum, size_t& start, T*& data, size_t& offset, 
        size_t& len, const size_t& step, const svbool_t& pg, const svbool_t& pg64) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[step];
            size_t base = start * offset;

            #pragma unroll
            for (size_t j = 0; j < step; ++j) {
                ori[j] = data[base + j * offset];
            }

            svfloat32_t ori_sve = svld1(pg, ori);
            svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error

            T tmp[step];
            int quant_vals[step];

            // calculate quantization code
            svfloat64_t quant_even_f64 = svcvt_f64_f32_x(pg64, quant_sve);
            svfloat64_t quant_odd_f64  = svcvtlt_f64_f32_x(pg64, quant_sve);
            quant_even_f64 = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_even_f64, real_ebx2_r));
            quant_odd_f64  = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_odd_f64, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_even_f64, -radius); // val > -radius
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_even_f64,  radius); // val < +radius
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_even_f64 = svsel_f64(pg_in_range, quant_even_f64, svdup_n_f64(0.0));

            svbool_t pg_gt_neg_o = svcmpgt_n_f64(pg64, quant_odd_f64, -radius);
            svbool_t pg_lt_pos_o = svcmplt_n_f64(pg64, quant_odd_f64,  radius);
            svbool_t pg_in_range_o = svand_b_z(pg64, pg_gt_neg_o, pg_lt_pos_o);
            quant_odd_f64 = svsel_f64(pg_in_range_o, quant_odd_f64, svdup_n_f64(0.0));
            
            // dequantization for decompression
            svfloat64_t decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum),
                    quant_even_f64, svdup_f64(real_ebx2));
            svfloat64_t decompressed_odd_f64  = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum),
                    quant_odd_f64, svdup_f64(real_ebx2));

            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);
            
            svst1_f32(pg, tmp, decompressed);

            quant_sve = svcvt_f32_f64_x(pg64, quant_even_f64);
            quant_sve = svcvtnt_f32_f64_x(quant_sve, pg64, quant_odd_f64);

            svfloat32_t err_dequan = svsub_f32_x(pg, decompressed, ori_sve);
            quant_sve = svadd_n_f32_x(pg, quant_sve, radius);

            pg_in_range = svand_b_z(pg, svcmpge_n_f32(pg, err_dequan, -real_eb), svcmple_n_f32(pg, err_dequan, real_eb));
            quant_sve = svsel_f32(pg_in_range, quant_sve, svdup_n_f32(0.0));

            svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
            svst1(pg, quant_vals, quant_sve_i);
            size_t j = 0;
            
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else
                    quantizer.force_save_unpred(ori[j]);
                ++frequency[quant_vals[j]];
            }

            svst1(pg, quant_inds + quant_index, quant_sve_i);
            quant_index += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            svint32_t quant_sve_i = svld1_s32(pg, quant_inds + quant_index);
            int quant_vals[step];
            svst1(pg, quant_vals, quant_sve_i);
            quant_sve_i = svsub_n_s32_x(pg, quant_sve_i, radius);
            
            svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
            svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));
            
            decompressed_even_f64 = svmul_n_f64_x(pg64, decompressed_even_f64, real_ebx2);
            decompressed_odd_f64 = svmul_n_f64_x(pg64, decompressed_odd_f64, real_ebx2);
            
            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);
            
            decompressed = svadd_f32_x(pg, decompressed, sum);

            T tmp[step];
            svst1_f32(pg, tmp, decompressed);
            
            size_t j = 0;
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else 
                    data[(start + j) * offset] = quantizer.recover_unpred();
            }
            quant_index += j;
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::quantize_double (svfloat64_t& sum, size_t& start, T*& data, size_t& offset, 
        size_t& len, const size_t& step, const svbool_t& pg64) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[step];
            size_t base = start * offset;

            #pragma unroll
            for (size_t j = 0; j < step; ++j) {
                ori[j] = data[base + j * offset];
            }

            svfloat64_t ori_sve = svld1(pg64, ori);
            svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
            T tmp[step];
            int quant_vals[step];
            
            quant_sve = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_sve, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_sve, -radius);
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_sve,  radius);
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));

            svfloat64_t decompressed = svmla_f64_x(pg64, sum, quant_sve, svdup_f64(real_ebx2));
            svst1_f64(pg64, tmp, decompressed);
            svfloat64_t err_dequan = svsub_f64_x(pg64, decompressed, ori_sve);

            quant_sve = svadd_n_f64_x(pg64, quant_sve, radius);
            pg_in_range = svand_b_z(pg64, svcmpge_n_f64(pg64, err_dequan, -real_eb), svcmple_n_f64(pg64, err_dequan, real_eb));
            quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));

            
            svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);
            
            svst1w_s64(pg64, quant_vals, quant_sve_i);

            size_t j = 0;
            #pragma unroll
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0)
                    data[(start + j) * offset] = tmp[j];
                else
                    quantizer.force_save_unpred(ori[j]);
                ++frequency[quant_vals[j]];
            }
            svst1w_s64(pg64, quant_inds + quant_index, quant_sve_i);
            quant_index += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { 
            svint64_t quant_sve_i = svld1sw_s64(pg64, quant_inds + quant_index);
            int quant_vals[step];
            svst1w_s64(pg64, quant_vals, quant_sve_i);
            quant_sve_i = svsub_n_s64_x(pg64, quant_sve_i, radius);

            svfloat64_t decompressed = svmla_f64_x(pg64, sum, 
                    svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
            T tmp[step];
            svst1_f64(pg64, tmp, decompressed);
            size_t j = 0;
            for ( ; j < step && start + j < len; ++j) {
                if (quant_vals[j] != 0) 
                    data[(start + j) * offset] = tmp[j];
                else
                    data[(start + j) * offset] = quantizer.recover_unpred();
            }
            quant_index += j;  
        }
    }
#else
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;

        for (; i + 1  < even_len; ++i) {
            size_t start = ((i << 1) + 1) * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_linear(buf[i], buf[i + 1]));
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else 
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_func(cur_ij_offset + last * offset , data[last * offset], pred_edge);
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data, 
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        
        T pred_first; 
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else 
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_func(cur_ij_offset + offset , data[offset], pred_first);

        size_t i = 0;
        for (; i + 3  < even_len; ++i) {
            size_t start = ((i << 1) + 3) * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_cubic(buf[i], buf[i + 1], buf[i + 2], buf[i + 3]));
        }
        
        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1] 
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_func(cur_ij_offset + last * offset, data[last * offset], edge_pred);
                

            }
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_linear(a[i], b[i]));
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_cubic(a[i], b[i], c[i], d[i]));
        }
    }
    
    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_equal_and_quantize(const T * a, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], a[i]);
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
  
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_linear1(a[i], b[i]));
        }
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_quad_1(a[i], b[i], c[i]));
        }
      
    }

    template <TUNING Tuning, class T, uint N, class Quantizer>
    template <COMPMODE CompMode, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition<Tuning, T, N, Quantizer>::interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data, 
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_func(cur_ij_offset + start,  data[start], interp_quad_2(a[i], b[i], c[i]));
        }
      
    }
#endif

}