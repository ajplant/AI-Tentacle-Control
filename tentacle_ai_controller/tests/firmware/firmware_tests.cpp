// HOST-STUB TESTS ONLY: these do not build or simulate real ESP32 hardware.
// From the tentacle_ai_controller project root on Linux/WSL:
//
// g++ -std=c++11 -Wall -Wextra -Werror -I tests/firmware tests/firmware/firmware_tests.cpp -o /tmp/tentacle_firmware_tests
// /tmp/tentacle_firmware_tests
//
// Use Arduino CLI / the ESP32 Arduino SDK separately for an actual core build.
// In particular, these tests cannot verify USB timing, motor current, neutral
// calibration, watchdog scheduling on hardware, or actual PWM waveforms.

#include "Arduino.h"

#ifdef NDEBUG
#error "Build these tests with assertions enabled (without -DNDEBUG)."
#endif

uint32_t fakeNow = 0;
uint32_t fakePwmOutputs[40] = {};
bool fakeFailAttach = false;
FakeSerial Serial;

#include "../../firmware/esp32_tentacle/esp32_tentacle.ino"

void issueCommand(const std::string &line) {
  char buffer[200];
  assert(line.size() < sizeof(buffer));
  strcpy(buffer, line.c_str());
  processCommand(buffer);
}

void feedSerial(const std::string &data) {
  for (char value : data) {
    Serial.input.push_back(value);
  }
  while (Serial.available()) {
    loop();
  }
}

int main() {
  // Boot: hardware attached, but outputs remain disabled until explicit ARM.
  setup();
  assert(hardwareReady && !armed);
  assert(fakePwmOutputs[18] == 0);
  assert(fakePwmOutputs[19] == 0);
  assert(fakePwmOutputs[23] == 0);

  // Commands while disarmed are rejected. Three distinct velocities produce
  // correspondingly distinct simulated pulse widths, without hidden ramping.
  issueCommand("V,1,0.1,0,0");
  assert(!armed);
  issueCommand("ARM");
  assert(armed);
  issueCommand("V,1,0.2,-0.1,0");
  assert(armed && haveSequence && lastSequence == 1);
  assert(fakePwmOutputs[18] > fakePwmOutputs[23]);
  assert(fakePwmOutputs[19] < fakePwmOutputs[23]);
  assert(Serial.output.find("OK,V,1\r\n") != std::string::npos);

  // A replayed sequence must stop and disarm, not renew the motor lease.
  issueCommand("V,1,0.2,-0.1,0");
  assert(!armed);

  // Strict field parsing, finite values, and conservative firmware speed cap.
  const char *invalidCommands[] = {
      "V,-1,0,0,0", "V,4294967296,0,0,0", "V,,0,0,0",
      "V,1,0,,0", "V,1,0,0,0,", "V,1,0,0,0x",
      "V,1,nan,0,0", "V,1,inf,0,0", "V,1,0.351,0,0",
      "V,1,-0.351,0,0", "V,1,0,0,1e-1000",
  };
  for (const char *invalid : invalidCommands) {
    issueCommand("ARM");
    issueCommand(invalid);
    assert(!armed);
  }

  // Sequence wrap is valid; an ordinary stale sequence is not.
  issueCommand("ARM");
  issueCommand("V,4294967295,0.35,0,-0.35");
  assert(armed);
  issueCommand("V,0,0,0,0");
  assert(armed);
  issueCommand("V,4294967295,0,0,0");
  assert(!armed);

  // Exact watchdog boundary and stop-to-neutral response.
  issueCommand("ARM");
  issueCommand("V,1,0,0,0");
  const uint32_t neutral = fakePwmOutputs[18];
  issueCommand("V,2,0.2,0,0");
  assert(fakePwmOutputs[18] != neutral);
  fakeNow += 249;
  loop();
  assert(armed);
  fakeNow += 1;
  loop();
  assert(!armed && fakePwmOutputs[18] == neutral);

  // Full serial TX buffer must not prevent the safety stop.
  issueCommand("ARM");
  issueCommand("V,1,0.2,0,0");
  Serial.writable = 0;
  fakeNow += 250;
  loop();
  assert(!armed && fakePwmOutputs[18] == neutral);
  Serial.writable = 10000;

  // Overflow and invalid-byte faults discard the entire remainder of a line;
  // a suffix resembling ARM cannot accidentally restart the controller.
  feedSerial(std::string(100, 'X') + "ARM\n");
  assert(!armed && !discardingLine);
  feedSerial("ARM\n");
  assert(armed);
  feedSerial(std::string("V,1,0,0,0") + std::string(1, '\0') + "ARM\n");
  assert(!armed);

  // A hardware attachment error disables all available outputs and latches
  // the initialization fault, preventing subsequent ARM commands.
  fakeFailAttach = true;
  setup();
  assert(!hardwareReady && !armed);
  issueCommand("ARM");
  assert(!armed);

  std::puts("Firmware host-stub safety tests passed (not an Arduino build).");
}
