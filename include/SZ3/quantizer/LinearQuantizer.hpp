#ifndef SZ3_LINEAR_QUANTIZER_HPP
#define SZ3_LINEAR_QUANTIZER_HPP

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/MemoryUtil.hpp"

namespace SZ3 {

template <class T>
class LinearQuantizer : public concepts::QuantizerInterface<T, int> {
   public:
    LinearQuantizer() : error_bound(1), double_error_bound(2), double_error_bound_reciprocal(0.5), radius(32768) {}

    LinearQuantizer(double eb, int r = 32768) : error_bound(eb),double_error_bound(2*eb), double_error_bound_reciprocal(0.5 / eb), radius(r) {
        assert(eb != 0);
    }

    double get_eb() const { return error_bound; }

    ALWAYS_INLINE std::tuple<double, double, double> get_all_eb() const { return {error_bound, double_error_bound, double_error_bound_reciprocal}; }

    void set_eb(double eb) {
        error_bound = eb;
        double_error_bound = 2 * eb;
        double_error_bound_reciprocal = 1.0 / double_error_bound;
    }

    std::pair<int, int> get_out_range() const override { return std::make_pair(0, radius * 2); }

    // quantize the data with a prediction value, and returns the quantization index and the decompressed data
    // int quantize(T data, T pred, T& dec_data);
     ALWAYS_INLINE int quantize_and_overwrite(T &data, T pred) override {
        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            //if (diff < 0) 
            //    quant_index = -quant_index;
            //auto quant_index_shifted = this->radius + quant_index;
            // fused, f64-intermediate dequant to match the SIMD (svmla / fmadd) paths bit-for-bit
            T decompressed_data = static_cast<T>(std::fma(static_cast<double>(quant_index), this->double_error_bound, static_cast<double>(pred)));
            // if data is NaN, the error is NaN, and NaN <= error_bound is false
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                data = decompressed_data;
                return quant_index;                 // int16: store signed quant_index directly (no +radius)
            } else {
                unpred.emplace_back(data);
                return -this->radius;               // escape sentinel = -radius (tracks radius; default -32768)
            }
        } else {
            unpred.emplace_back(data);
            return -this->radius;
        }
    }

    // recover the data using the quantization index
    ALWAYS_INLINE T recover(T pred, int quant_index) override {
        if (quant_index != -this->radius) {                // escape = -radius; other codes are signed quant errors
            return recover_pred(pred, quant_index);
        } else {
            return recover_unpred();
        }
    }

    ALWAYS_INLINE T recover_pred(T pred, int quant_index) {
        // must match quantize_and_overwrite's dequant exactly (fused, f64)
        return static_cast<T>(std::fma(static_cast<double>(quant_index), this->double_error_bound, static_cast<double>(pred)));
    }

    ALWAYS_INLINE T recover_unpred() { return unpred[index++]; }

    ALWAYS_INLINE int force_save_unpred(T ori) override {
        unpred.emplace_back(ori);
        return -this->radius;
    }

    size_t size_est() { return unpred.size() * sizeof(T); }

    void save(unsigned char *&c) const override {
        write(uid, c);
        write(this->error_bound, c);
        write(this->radius, c);
        int unpred_size = unpred.size();
        write(unpred_size, c);
        if (unpred_size > 0) {
            write(unpred.data(), unpred.size(), c);
        }
    }

    void load(const unsigned char *&c, size_t &remaining_length) override {
        uchar uid_read;
        read(uid_read, c, remaining_length);
        if (uid_read != uid) {
            throw std::invalid_argument("LinearQuantizer uid mismatch");
        }
        double eb;
        read(eb, c, remaining_length);
        set_eb(eb);
        read(this->radius, c, remaining_length);
        int unpred_size = 0;
        read(unpred_size, c, remaining_length);
        if (unpred_size > 0) {
            unpred.resize(unpred_size);
            read(unpred.data(), unpred_size, c, remaining_length);
        }
        index = 0;
    }
    std::vector<T>& test_unpred() {
        return unpred;
    }

    void print() override {
        printf("[LinearQuantizer] error_bound = %.8G, radius = %d, unpred = %zu\n", error_bound, radius, unpred.size());
    }

   private:
    std::vector<T> unpred;
    size_t index = 0;  // used in decompression only
    uchar uid = 0b10;

    double error_bound;
    double double_error_bound;
    double double_error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace SZ3
#endif