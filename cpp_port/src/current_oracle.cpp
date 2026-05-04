#include "synth/core.hpp"

namespace synth {

std::uint8_t current_oracle(std::uint8_t x) {
    return static_cast<std::uint8_t>((1/(sin8_value(x))*cos8_value(x)) & MASK);
}

}  // namespace synth
