#include <Arduino.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ESP32 Arduino Core 3.x firmware for three continuous-rotation MG90 servos.
// Commands are newline terminated:
//   ARM
//   V,<sequence>,<servo1>,<servo2>,<servo3>   speeds: -0.35 through +0.35
//   STOP
//   PING

namespace Config {
constexpr uint32_t kSerialBaud = 115200;
constexpr uint32_t kPwmFrequencyHz = 50;
constexpr uint8_t kPwmResolutionBits = 16;
constexpr uint32_t kCommandWatchdogMs = 250;
constexpr float kMaximumSpeed = 0.35f;
constexpr size_t kCommandBufferSize = 96;
constexpr size_t kMaximumBytesPerLoop = 64;
}  // namespace Config

class IContinuousServo {
 public:
  virtual ~IContinuousServo() = default;
  virtual bool begin() = 0;
  virtual void setSpeed(float normalizedSpeed) = 0;
  virtual void stop() = 0;
  virtual void disable() = 0;
};

struct ContinuousServoCalibration {
  uint8_t pin;
  uint16_t reverseLimitUs;
  uint16_t neutralUs;
  uint16_t forwardLimitUs;
  bool inverted;
};

class MG90ContinuousServo final : public IContinuousServo {
 public:
  explicit MG90ContinuousServo(ContinuousServoCalibration calibration)
      : calibration_(calibration) {}

  bool begin() override {
    if (!(calibration_.reverseLimitUs < calibration_.neutralUs &&
          calibration_.neutralUs < calibration_.forwardLimitUs &&
          calibration_.reverseLimitUs >= 1000 &&
          calibration_.forwardLimitUs <= 2000)) {
      return false;
    }
    attached_ = ledcAttach(calibration_.pin, Config::kPwmFrequencyHz,
                           Config::kPwmResolutionBits);
    if (attached_) {
      disable();
    }
    return attached_;
  }

  void setSpeed(float normalizedSpeed) override {
    normalizedSpeed = constrain(normalizedSpeed, -1.0f, 1.0f);
    if (calibration_.inverted) {
      normalizedSpeed = -normalizedSpeed;
    }

    float pulseUs = calibration_.neutralUs;
    if (normalizedSpeed >= 0.0f) {
      pulseUs += normalizedSpeed *
                 (calibration_.forwardLimitUs - calibration_.neutralUs);
    } else {
      pulseUs += normalizedSpeed *
                 (calibration_.neutralUs - calibration_.reverseLimitUs);
    }
    writePulseUs(static_cast<uint16_t>(lroundf(pulseUs)));
  }

  void stop() override { writePulseUs(calibration_.neutralUs); }

  void disable() override {
    if (attached_) {
      ledcWrite(calibration_.pin, 0);
    }
  }

 private:
  void writePulseUs(uint16_t pulseUs) {
    if (!attached_) {
      return;
    }
    constexpr uint32_t periodUs = 1000000UL / Config::kPwmFrequencyHz;
    constexpr uint32_t maximumDuty =
        (1UL << Config::kPwmResolutionBits) - 1UL;
    const uint32_t duty = static_cast<uint32_t>(
        (static_cast<uint64_t>(pulseUs) * maximumDuty) / periodUs);
    ledcWrite(calibration_.pin, duty);
  }

  ContinuousServoCalibration calibration_;
  bool attached_ = false;
};

class TentacleDrive {
 public:
  TentacleDrive(IContinuousServo &first, IContinuousServo &second,
                IContinuousServo &third)
      : servos_{&first, &second, &third} {}

  bool begin() {
    bool allStarted = true;
    for (IContinuousServo *servo : servos_) {
      allStarted = servo->begin() && allStarted;
    }
    disableAll();
    return allStarted;
  }

  void setSpeeds(float first, float second, float third) {
    const float requested[] = {first, second, third};
    for (size_t i = 0; i < 3; ++i) {
      // No hidden ramp: the host applies its acceleration limit and integrates
      // the commands it actually sent. Calibration pulses must be reproducible.
      servos_[i]->setSpeed(requested[i]);
    }
  }

  void stopAll() {
    for (size_t i = 0; i < 3; ++i) {
      servos_[i]->stop();
    }
  }

  void disableAll() {
    for (size_t i = 0; i < 3; ++i) {
      servos_[i]->disable();
    }
  }

 private:
  IContinuousServo *servos_[3];
};

// Tune neutralUs separately for each physical servo. If a servo creeps at
// speed 0, adjust only its neutralUs by a few microseconds and re-upload.
MG90ContinuousServo servo1({18, 1200, 1500, 1800, false});
MG90ContinuousServo servo2({19, 1200, 1500, 1800, false});
MG90ContinuousServo servo3({23, 1200, 1500, 1800, false});
TentacleDrive tentacleDrive(servo1, servo2, servo3);

char commandBuffer[Config::kCommandBufferSize] = {};
size_t commandLength = 0;
uint32_t lastValidVelocityMs = 0;
bool armed = false;
bool hardwareReady = false;
bool discardingLine = false;
bool haveSequence = false;
uint32_t lastSequence = 0;

// Responses must not block the independent safety loop if the host stops
// reading its serial input. The host treats missing acknowledgements as a fault.
void sendResponse(const char *response) {
  const size_t length = strlen(response);
  if (Serial.availableForWrite() >= static_cast<int>(length + 2)) {
    Serial.write(reinterpret_cast<const uint8_t *>(response), length);
    Serial.write(reinterpret_cast<const uint8_t *>("\r\n"), 2);
  }
}

void stopAndDisarm(const char *response) {
  if (hardwareReady) {
    tentacleDrive.stopAll();
  } else {
    tentacleDrive.disableAll();
  }
  armed = false;
  haveSequence = false;
  sendResponse(response);
}

void checkWatchdog() {
  if (armed &&
      static_cast<uint32_t>(millis() - lastValidVelocityMs) >=
          Config::kCommandWatchdogMs) {
    stopAndDisarm("ERR,WATCHDOG");
  }
}

bool parseSequence(const char *token, uint32_t &sequence) {
  if (*token == '\0') {
    return false;
  }
  uint32_t value = 0;
  for (const char *cursor = token; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
    const uint32_t digit = static_cast<uint32_t>(*cursor - '0');
    if (value > (UINT32_MAX - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
  }
  sequence = value;
  return true;
}

bool parseSpeed(const char *token, float &speed) {
  if (*token == '\0') {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  speed = strtof(token, &end);
  return end != token && *end == '\0' && errno != ERANGE &&
         isfinite(speed) && speed >= -Config::kMaximumSpeed &&
         speed <= Config::kMaximumSpeed;
}

void processCommand(char *command) {
  checkWatchdog();
  while (*command == ' ' || *command == '\t') {
    ++command;
  }
  size_t length = strlen(command);
  while (length > 0 &&
         (command[length - 1] == ' ' || command[length - 1] == '\t')) {
    command[--length] = '\0';
  }

  if (strcmp(command, "ARM") == 0) {
    if (!hardwareReady) {
      stopAndDisarm("ERR,PWM_ATTACH");
      return;
    }
    tentacleDrive.stopAll();
    armed = true;
    haveSequence = false;
    lastValidVelocityMs = millis();
    sendResponse("OK,ARM");
    return;
  }

  if (strcmp(command, "STOP") == 0) {
    stopAndDisarm("OK,STOP");
    return;
  }

  if (strcmp(command, "PING") == 0) {
    sendResponse("PONG");
    return;
  }

  // Split without strtok: empty fields must not be silently discarded.
  char *tokens[5] = {};
  size_t tokenCount = 1;
  tokens[0] = command;
  for (char *cursor = command; *cursor != '\0'; ++cursor) {
    if (*cursor == ',') {
      if (tokenCount >= 5) {
        stopAndDisarm("ERR,BAD_COMMAND");
        return;
      }
      *cursor = '\0';
      tokens[tokenCount++] = cursor + 1;
    }
  }
  uint32_t sequence = 0;
  float speeds[3] = {};
  if (tokenCount != 5 || strcmp(tokens[0], "V") != 0 ||
      !parseSequence(tokens[1], sequence) ||
      !parseSpeed(tokens[2], speeds[0]) ||
      !parseSpeed(tokens[3], speeds[1]) ||
      !parseSpeed(tokens[4], speeds[2])) {
    stopAndDisarm("ERR,BAD_COMMAND");
    return;
  }

  if (!armed) {
    stopAndDisarm("ERR,NOT_ARMED");
    return;
  }

  // Reject duplicate or backwards sequence numbers, including replayed packets.
  // Unsigned subtraction also supports normal wrap from UINT32_MAX to zero.
  const uint32_t advance = sequence - lastSequence;
  if (haveSequence && (advance == 0 || advance >= 0x80000000UL)) {
    stopAndDisarm("ERR,STALE_SEQUENCE");
    return;
  }

  tentacleDrive.setSpeeds(speeds[0], speeds[1], speeds[2]);
  lastSequence = sequence;
  haveSequence = true;
  lastValidVelocityMs = millis();
  char response[32] = {};
  snprintf(response, sizeof(response), "OK,V,%lu",
           static_cast<unsigned long>(sequence));
  sendResponse(response);
}

void readSerialCommands() {
  size_t processed = 0;
  while (Serial.available() > 0 && processed < Config::kMaximumBytesPerLoop) {
    checkWatchdog();
    const char incoming = static_cast<char>(Serial.read());
    ++processed;
    if (incoming == '\n' || incoming == '\r') {
      if (!discardingLine && commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        processCommand(commandBuffer);
      }
      commandLength = 0;
      discardingLine = false;
    } else if (discardingLine) {
      continue;
    } else if (incoming != '\t' &&
               (static_cast<unsigned char>(incoming) < 32 ||
                static_cast<unsigned char>(incoming) > 126)) {
      commandLength = 0;
      discardingLine = true;
      stopAndDisarm("ERR,BAD_CHARACTER");
    } else if (commandLength < Config::kCommandBufferSize - 1) {
      commandBuffer[commandLength++] = incoming;
    } else {
      commandLength = 0;
      discardingLine = true;
      stopAndDisarm("ERR,COMMAND_TOO_LONG");
    }
  }
}

void setup() {
  Serial.begin(Config::kSerialBaud);
  delay(250);
  hardwareReady = tentacleDrive.begin();
  if (!hardwareReady) {
    sendResponse("ERR,PWM_ATTACH");
  } else {
    sendResponse("READY,MG90_TENTACLE,GPIO18,GPIO19,GPIO23");
  }
}

void loop() {
  checkWatchdog();
  readSerialCommands();
  checkWatchdog();
}
