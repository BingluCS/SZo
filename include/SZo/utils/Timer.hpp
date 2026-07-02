//
// Created by Kai Zhao on 10/30/20.
//

#ifndef SZo_TIMER_HPP
#define SZo_TIMER_HPP

#include <chrono>
#include <iostream>
#include <string>

namespace SZo {
class Timer {
   public:
    Timer() = default;

    Timer(bool initstart) {
        if (initstart) {
            start();
        }
    }

    void start() { begin = std::chrono::steady_clock::now(); }

    double stop() {
        end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - begin).count();
    }

    double stop(const std::string &msg) {
        double seconds = stop();
//#if SZo_DEBUG_TIMINGS
        std::cout << msg << " time = " << seconds << "s" << std::endl;
//#endif
        return seconds;
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
};
}  // namespace SZo

#endif  // SZ_TIMER_HPP
