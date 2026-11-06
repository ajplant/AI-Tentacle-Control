#pragma once

// HOST TEST STUB ONLY. This is not the Arduino/ESP32 SDK and must never be
// copied into the sketch directory or used for an ESP32 firmware upload.
// It simulates just enough time, serial I/O, and PWM to test controller logic.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>

using std::isfinite;
using std::size_t;

extern uint32_t fakeNow;
extern uint32_t fakePwmOutputs[40];
extern bool fakeFailAttach;

inline uint32_t millis() { return fakeNow; }
inline void delay(uint32_t amount) { fakeNow += amount; }

template <class T>
T constrain(T value, T low, T high) {
  return std::max(low, std::min(value, high));
}

inline bool ledcAttach(uint8_t pin, uint32_t frequency, uint8_t resolution) {
  assert(pin < 40);
  assert(frequency == 50);
  assert(resolution == 16);
  return !(fakeFailAttach && pin == 19);
}

inline bool ledcWrite(uint8_t pin, uint32_t duty) {
  assert(pin < 40);
  fakePwmOutputs[pin] = duty;
  return true;
}

struct FakeSerial {
  std::deque<char> input;
  std::string output;
  int writable = 10000;

  void begin(uint32_t baudrate) { assert(baudrate == 115200); }
  int availableForWrite() { return writable; }
  size_t write(const uint8_t *bytes, size_t count) {
    output.append(reinterpret_cast<const char *>(bytes), count);
    return count;
  }
  int available() { return static_cast<int>(input.size()); }
  int read() {
    assert(!input.empty());
    const char value = input.front();
    input.pop_front();
    return static_cast<unsigned char>(value);
  }
};

extern FakeSerial Serial;
