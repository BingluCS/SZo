//
// Created by Kai Zhao on 4/21/20.
//

#ifndef SZo_LOSSLESS_BYPASS_HPP
#define SZo_LOSSLESS_BYPASS_HPP

#include <cstring>
#include "SZo/def.hpp"
#include "SZo/lossless/Lossless.hpp"

namespace SZo {
class Lossless_bypass : public concepts::LosslessInterface {
public:
    size_t compress(const uchar *src, size_t srcLen, uchar *dst, size_t dstCap) override {
        std::memcpy(dst, src, srcLen);
        // dst = src;
        return srcLen;
    }

    size_t decompress(const uchar *src, const size_t srcLen, uchar *&dst, size_t &dstLen) override {
        dstLen = srcLen;
        if (dst == nullptr) {
            dst = static_cast<uchar *>(malloc(dstLen));
        }
        std::memcpy(dst, src, dstLen);
        return dstLen;
    }
};
}  // namespace SZo
#endif  // SZ_LOSSLESS_BYPASS_HPP
