//
// Created by Kai Zhao on 1/29/21.
//

#ifndef SZo_PREFILTER_HPP
#define SZo_PREFILTER_HPP

#include "SZo/preprocessor/PreProcessor.hpp"

namespace SZo {
    template<class T, uint N>

    class PreFilter : public concepts::PreprocessorInterface<T, N> {

        void preprocess(T *data, std::array<size_t, N> dims, std::pair<T, T> range, T defaultValue) {
            for (T &d : data) {
                if (d > range.second || d < range.first) {
                    d = defaultValue;
                }
            }
        }
    };
}
#endif //SZo_PRETRANSPOSE_H
