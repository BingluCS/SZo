#include <cmath>
#include <cstdint>

#include "SZo/encoder/ArithmeticEncoder.hpp"
#include "SZo/encoder/BypassEncoder.hpp"
#include "SZo/encoder/HuffmanEncoder.hpp"
#include "SZo/encoder/RunlengthEncoder.hpp"
#include "gtest/gtest.h"

template <typename Encoder, typename T>
void runFunctionalTest() {
    int N = 1000;
    std::vector<SZo::uchar> buffer_data(N * sizeof(T) * 2);
    std::vector<SZo::uchar> buffer_conf(N * sizeof(T) * 2);
    std::vector<T> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i % 100;
    }

    size_t data_len = 0, conf_len = 0;
    {
        SZo::uchar *buffer_data_pos = buffer_data.data();
        SZo::uchar *buffer_conf_pos = buffer_conf.data();
        Encoder coder;
        coder.preprocess_encode(data, 100);
        coder.encode(data, buffer_data_pos);
        coder.save(buffer_conf_pos);
        data_len = buffer_data_pos - buffer_data.data();
        conf_len = buffer_conf_pos - buffer_conf.data();
    }
    {
        const SZo::uchar *buffer_data_pos = buffer_data.data();
        const SZo::uchar *buffer_conf_pos = buffer_conf.data();
        Encoder coder;
        coder.load(buffer_conf_pos, conf_len);
        auto dataDecoded = coder.decode(buffer_data_pos, N);
        for (int i = 0; i < N; i++) {
            EXPECT_EQ(data[i], dataDecoded[i]);
        }
    }
}

template <typename Encoder, typename T>
void runAllTest() {
    runFunctionalTest<Encoder, T>();
}

TEST(EncoderTest, HuffmanEncoder) { runAllTest<SZo::HuffmanEncoder<int>, int>(); }

TEST(EncoderTest, RunlengthEncoder) { runAllTest<SZo::RunlengthEncoder<int>, int>(); }

TEST(EncoderTest, ArithmeticEncoder) { runAllTest<SZo::ArithmeticEncoder<int>, int>(); }

TEST(EncoderTest, BypassEncoder) { runAllTest<SZo::BypassEncoder<int>, int>(); }
