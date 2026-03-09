//
// Created by Kai Zhao on 4/20/20.
//

#ifndef SZ3_STATISTIC_HPP
#define SZ3_STATISTIC_HPP

#include "Config.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

namespace SZ3 {
template <class T>
T data_range(const T *data, size_t num) {

    
    if( num <= 16){
        T max = data[0];
        T min = data[0];
        for (size_t i = 1; i < num; ++i) {
            if (max < data[i]) max = data[i];
            if (min > data[i]) min = data[i];
        }
        return max - min;

    }

    T max_val = data[0];
    T min_val = data[0];
#ifdef __ARM_FEATURE_SVE2
    constexpr bool is_float_sve  = std::is_same_v<T, float>;
    constexpr bool is_double_sve = std::is_same_v<T, double>;
    if constexpr (is_float_sve) {
#ifdef _OPENMP
        #pragma omp parallel reduction(min:min_val) reduction(max:max_val)
        {
            svbool_t pg = svptrue_b32();
            uint64_t vl = svcntw();  // number of floats per SVE vector
            svfloat32_t vmax = svdup_f32(data[0]);
            svfloat32_t vmin = svdup_f32(data[0]);
            #pragma omp for nowait
            for (size_t i = 0; i + vl <= num; i += vl) {
                svfloat32_t v = svld1_f32(pg, data + i);
                vmax = svmax_f32_x(pg, vmax, v);
                vmin = svmin_f32_x(pg, vmin, v);
            }
            max_val = std::max(max_val, (T)svmaxv_f32(pg, vmax));
            min_val = std::min(min_val, (T)svminv_f32(pg, vmin));
        }
        uint64_t vl = svcntw();
        size_t tail_start = (num / vl) * vl;
        for (size_t k = tail_start; k < num; ++k) {
            max_val = std::max(max_val, data[k]);
            min_val = std::min(min_val, data[k]);
        }
        return max_val - min_val;
#else
        svbool_t pg = svptrue_b32();
        uint64_t vl = svcntw();
        svfloat32_t vmax = svdup_f32(data[0]);
        svfloat32_t vmin = svdup_f32(data[0]);
        size_t i = 0;
        for (; i + vl <= num; i += vl) {
            svfloat32_t v = svld1_f32(pg, data + i);
            vmax = svmax_f32_x(pg, vmax, v);
            vmin = svmin_f32_x(pg, vmin, v);
        }
        T maxval = (T)svmaxv_f32(pg, vmax);
        T minval = (T)svminv_f32(pg, vmin);
        for (; i < num; ++i) {
            maxval = std::max(maxval, data[i]);
            minval = std::min(minval, data[i]);
        }
        return maxval - minval;
#endif
    } else if constexpr (is_double_sve) {
#ifdef _OPENMP
        #pragma omp parallel reduction(min:min_val) reduction(max:max_val)
        {
            svbool_t pg = svptrue_b64();
            uint64_t vl = svcntd();  // number of doubles per SVE vector
            svfloat64_t vmax = svdup_f64(data[0]);
            svfloat64_t vmin = svdup_f64(data[0]);
            #pragma omp for nowait
            for (size_t i = 0; i + vl <= num; i += vl) {
                svfloat64_t v = svld1_f64(pg, data + i);
                vmax = svmax_f64_x(pg, vmax, v);
                vmin = svmin_f64_x(pg, vmin, v);
            }
            max_val = std::max(max_val, (T)svmaxv_f64(pg, vmax));
            min_val = std::min(min_val, (T)svminv_f64(pg, vmin));
        }
        uint64_t vl = svcntd();
        size_t tail_start = (num / vl) * vl;
        for (size_t k = tail_start; k < num; ++k) {
            max_val = std::max(max_val, data[k]);
            min_val = std::min(min_val, data[k]);
        }
        return max_val - min_val;
#else
        svbool_t pg = svptrue_b64();
        uint64_t vl = svcntd();
        svfloat64_t vmax = svdup_f64(data[0]);
        svfloat64_t vmin = svdup_f64(data[0]);
        size_t i = 0;
        for (; i + vl <= num; i += vl) {
            svfloat64_t v = svld1_f64(pg, data + i);
            vmax = svmax_f64_x(pg, vmax, v);
            vmin = svmin_f64_x(pg, vmin, v);
        }
        T maxval = (T)svmaxv_f64(pg, vmax);
        T minval = (T)svminv_f64(pg, vmin);
        for (; i < num; ++i) {
            maxval = std::max(maxval, data[i]);
            minval = std::min(minval, data[i]);
        }
        return maxval - minval;
#endif
    } else {
        for (size_t i = 1; i < num; ++i) {
            if (data[i] > max_val) max_val = data[i];
            if (data[i] < min_val) min_val = data[i];
        }
        return max_val - min_val;
    }
#elif defined(__AVX2__)
    constexpr bool is_float  = std::is_same_v<T, float>;
    constexpr bool is_double = std::is_same_v<T, double>;
    if constexpr (is_float){
#ifdef _OPENMP
    int res = num % 8;
    #pragma omp parallel reduction(min:min_val) reduction(max:max_val)
    {
        size_t i = 0;
        __m256 vmax = _mm256_set1_ps(data[0]);
        __m256 vmin = _mm256_set1_ps(data[0]);

        #pragma omp for nowait
        for (i = 0; i < num - res; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            vmax = _mm256_max_ps(vmax, v);
            vmin = _mm256_min_ps(vmin, v);
        }

        float tmp_max[8], tmp_min[8];
        _mm256_storeu_ps(tmp_max, vmax);
        _mm256_storeu_ps(tmp_min, vmin);
        
        for (int k = 0; k < 8; ++k) {
            max_val = std::max(max_val, tmp_max[k]);
            min_val = std::min(min_val, tmp_min[k]);
        }
    }
    for (size_t k = num - res; k < num; ++k){
        max_val = std::max(max_val, data[k]);
        min_val = std::min(min_val, data[k]);
    }
    return max_val - min_val;
#else
        __m256 vmax = _mm256_set1_ps(data[0]);
        __m256 vmin = _mm256_set1_ps(data[0]);
        for (; i + 7 < num; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            vmax = _mm256_max_ps(vmax, v);
            vmin = _mm256_min_ps(vmin, v);
        }

        float tmp_max[8];
        float tmp_min[8];
        _mm256_storeu_ps(tmp_max, vmax);
        _mm256_storeu_ps(tmp_min, vmin);

        float maxval = tmp_max[0], minval = tmp_min[0];
        for (int k = 1; k < 8; ++k){
            maxval = std::max(maxval, tmp_max[k]);
            minval = std::min(minval, tmp_min[k]);
        }

        for (; i < num; ++i){
            maxval = std::max(maxval, data[i]);
            minval = std::min(minval, data[i]);
        }
        return maxval - minval;
#endif   
    }
    else if constexpr (is_double){
#ifdef _OPENMP
    int res = num % 4;
    double max_val = data[0];
    double min_val = data[0];
    #pragma omp parallel reduction(min:min_val) reduction(max:max_val)
    {
        size_t i = 0;
        __m256d vmax = _mm256_set1_pd(data[0]);
        __m256d vmin = _mm256_set1_pd(data[0]);

        #pragma omp for nowait
        for (i = 0; i < num - res; i += 4) {
            __m256d v = _mm256_loadu_pd(data + i);
            vmax = _mm256_max_pd(vmax, v);
            vmin = _mm256_min_pd(vmin, v);
        }

        // 将向量化结果存到临时数组，归约成标量
        double tmp_max[4], tmp_min[4];
        _mm256_storeu_pd(tmp_max, vmax);
        _mm256_storeu_pd(tmp_min, vmin);
        
        for (int k = 0; k < 4; ++k) {
            max_val = std::max(max_val, tmp_max[k]);
            min_val = std::min(min_val, tmp_min[k]);
        }
    }
    for (size_t k = num - res; k < num; ++k){
        max_val = std::max(max_val, data[k]);
        min_val = std::min(min_val, data[k]);
    }
    return max_val - min_val;
#else
        size_t i = 0;
        __m256d vmax = _mm256_set1_pd(data[0]);
        __m256d vmin = _mm256_set1_pd(data[0]);

        for (; i + 3 < num; i += 4) {
            __m256d v = _mm256_loadu_pd(data + i);
            vmax = _mm256_max_pd(vmax, v);
            vmin = _mm256_min_pd(vmin, v);
        }

        double tmp_max[4];
        double tmp_min[4];
        _mm256_storeu_pd(tmp_max, vmax);
        _mm256_storeu_pd(tmp_min, vmin);

        double maxval = tmp_max[0], minval = tmp_min[0];
        for (int k = 1; k < 4; ++k){
            maxval = std::max(maxval, tmp_max[k]);
            minval = std::min(minval, tmp_min[k]);
        }

        for (; i < num; ++i){
            maxval = std::max(maxval, data[i]);
            minval = std::min(minval, data[i]);
        }

        return maxval  - minval;
#endif
    }
    else{
        T max = data[0];
        T min = data[0];
        for (size_t i = 1; i < num; ++i) {
            if (max < data[i]) max = data[i];
            if (min > data[i]) min = data[i];
        }
        return max - min;
    }
#else
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
#endif
    for (size_t i = 1; i < num; ++i) {
        if (data[i] > max_val) max_val = data[i];
        if (data[i] < min_val) min_val = data[i];
    }
    return max_val - min_val;
#endif
}

inline int factorial(int n) { return (n == 0) || (n == 1) ? 1 : n * factorial(n - 1); }

inline double computeABSErrBoundFromPSNR(double psnr, double threshold, double value_range) {
    double v1 = psnr + 10 * log10(1 - 2.0 / 3.0 * threshold);
    double v2 = v1 / (-20);
    double v3 = pow(10, v2);
    return value_range * v3;
}

template <class T>
void calAbsErrorBound(Config &conf, const T *data, T range = 0) {
    if (conf.errorBoundMode != EB_ABS) {
        if (conf.errorBoundMode == EB_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num));
        } else if (conf.errorBoundMode == EB_PSNR) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = computeABSErrBoundFromPSNR(conf.psnrErrorBound, 0.99,
                                                            ((range > 0) ? range : data_range(data, conf.num)));
        } else if (conf.errorBoundMode == EB_L2NORM) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = sqrt(3.0 / conf.num) * conf.l2normErrorBound;
        } else if (conf.errorBoundMode == EB_ABS_AND_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound =
                std::min(conf.absErrorBound, conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num)));
        } else if (conf.errorBoundMode == EB_ABS_OR_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound =
                std::max(conf.absErrorBound, conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num)));
        } else {
            throw std::invalid_argument("Error bound mode not supported");
        }
    }
}

template <typename Type>
double autocorrelation1DLag1(const Type *data, size_t numOfElem, Type avg) {
    double cov = 0;
    for (size_t i = 0; i < numOfElem; i++) {
        cov += (data[i] - avg) * (data[i] - avg);
    }
    cov = cov / numOfElem;

    if (cov == 0) {
        return 0;
    } else {
        int delta = 1;
        double sum = 0;

        for (size_t i = 0; i < numOfElem - delta; i++) {
            sum += (data[i] - avg) * (data[i + delta] - avg);
        }
        return sum / (numOfElem - delta) / cov;
    }
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse, double &max_diff) {
    size_t i = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    max_diff = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0, l2sum = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
        l2sum += data[i] * data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double *diff = static_cast<double *>(malloc(num_elements * sizeof(double)));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr) maxpw_relerr = relerr;
        }

        if (max_diff < err) max_diff = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = sum / num_elements;
    double range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(l2sum);

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.2G\n", max_diff);
    printf("Max relative error = %.2G\n", max_diff / (Max - Min));
    printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.10G\n", psnr, nrmse);
    printf("normError = %f, normErr_norm = %f\n", normErr, normErr_norm);
    printf("acEff=%f\n", acEff);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements) {
    double psnr, nrmse, max_diff;
    verify(ori_data, data, num_elements, psnr, nrmse, max_diff);
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse) {
    double max_diff;
    verify(ori_data, data, num_elements, psnr, nrmse, max_diff);
}
}  // namespace SZ3

#endif
