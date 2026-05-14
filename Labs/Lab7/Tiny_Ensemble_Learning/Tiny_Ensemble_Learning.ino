#include "TensorFlowLite.h"

// Model headers
extern const unsigned char encoder_clf_raw_pruned_qat_int8_tflite[];
extern const int encoder_clf_raw_pruned_qat_int8_tflite_len;

extern const unsigned char encoder_clf_std_pruned_qat_int8_tflite[];
extern const int encoder_clf_std_pruned_qat_int8_tflite_len;

extern const unsigned char encoder_clf_minmax_pruned_qat_int8_tflite[];
extern const int encoder_clf_minmax_pruned_qat_int8_tflite_len;

extern const unsigned char stacked_meta_clf_pruned_qat_int8_tflite[];
extern const int stacked_meta_clf_pruned_qat_int8_tflite_len;

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

//#include <Arduino_LSM9DS1.h>
#include <Arduino_BMI270_BMM150.h>

// Label map
const char* label_map[12] = {
  "Standing still", "Sitting and relaxing", "Lying down", "Walking",
  "Climbing stairs", "Waist bends forward", "Frontal elevation of arms",
  "Knees bending (crouching)", "Cycling", "Jogging", "Running", "Jump front & back"
};

// Parameters
const int kWindowSize = 100;
const int kFeatureCount = 6;
const int kInputSize = kWindowSize * kFeatureCount;
const int kNumClasses = 12;
const int kMetaInputSize = 36;
const int kSharedArenaSize = 48 * 1024;

// Memory
__attribute__((aligned(32), section(".noinit"))) static uint8_t shared_arena[kSharedArenaSize];
__attribute__((aligned(32), section(".noinit"))) static float window_buffer[kWindowSize][kFeatureCount];

// Normalization
float standard_means[6] = {3.268805, -9.153526, 0.700608, -0.020571, -0.520382, -0.299817};
float standard_stds[6]  = {4.253270, 5.397648, 6.433731, 0.325434, 0.571107, 0.542112};
float minmax_mins[6]    = {-22.091, -19.571, -19.363, -0.977740, -1.611600, -1.206300};
float minmax_maxs[6]    = { 20.003,  20.909,  24.599,  1.463800,  1.557200,  1.326100};

// State
int sample_index = 0;
bool window_ready = false;

// TF Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

void run_model(const tflite::Model* model_data,
               float* input_data,
               int input_length,
               float* output_data,
               const char* name) {
  static tflite::AllOpsResolver resolver;

  if (interpreter) {
    delete interpreter;
    interpreter = nullptr;
  }

  interpreter = new tflite::MicroInterpreter(model_data, resolver, shared_arena, kSharedArenaSize, error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.print("❌ AllocateTensors failed for "); Serial.println(name);
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  for (int i = 0; i < input_length; i++) {
    input->data.int8[i] = round(input_data[i] / input->params.scale) + input->params.zero_point;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.print("❌ Inference failed for "); Serial.println(name);
    while (1);
  }

  for (int i = 0; i < output->dims->data[1]; i++) {
    output_data[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
  }
}

void run_meta_model(float* stacked_input, float* final_output) {
  static tflite::AllOpsResolver resolver;

  if (interpreter) {
    delete interpreter;
    interpreter = nullptr;
  }

  interpreter = new tflite::MicroInterpreter(
    tflite::GetModel(stacked_meta_clf_pruned_qat_int8_tflite),
    resolver, shared_arena, kSharedArenaSize, error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Meta model AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  for (int i = 0; i < kMetaInputSize; i++) {
    input->data.int8[i] = round(stacked_input[i] / input->params.scale) + input->params.zero_point;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Meta model inference failed");
    while (1);
  }

  for (int i = 0; i < kNumClasses; i++) {
    final_output[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("❌ IMU not found");
    while (1);
  }

  Serial.println("✅ IMU initialized. Ready to collect data.");
}

void loop() {
  float ax, ay, az, gx, gy, gz;
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() &&
      IMU.readAcceleration(ax, ay, az) && IMU.readGyroscope(gx, gy, gz)) {

    window_buffer[sample_index][0] = ax;
    window_buffer[sample_index][1] = ay;
    window_buffer[sample_index][2] = az;
    window_buffer[sample_index][3] = gx;
    window_buffer[sample_index][4] = gy;
    window_buffer[sample_index][5] = gz;

    sample_index++;
    if (sample_index >= kWindowSize) {
      sample_index = 0;
      window_ready = true;
    }

    if (window_ready) {
      static float raw_input[kInputSize], std_input[kInputSize], minmax_input[kInputSize];
      static float raw_out[kNumClasses], std_out[kNumClasses], minmax_out[kNumClasses];
      static float stacked_input[kMetaInputSize], meta_out[kNumClasses];

      for (int i = 0; i < kWindowSize; i++) {
        for (int j = 0; j < kFeatureCount; j++) {
          int idx = i * kFeatureCount + j;
          float val = window_buffer[i][j];
          raw_input[idx] = val;
          std_input[idx] = (val - standard_means[j]) / standard_stds[j];
          minmax_input[idx] = (val - minmax_mins[j]) / (minmax_maxs[j] - minmax_mins[j]);
        }
      }

      run_model(tflite::GetModel(encoder_clf_raw_pruned_qat_int8_tflite), raw_input, kInputSize, raw_out, "raw");
      run_model(tflite::GetModel(encoder_clf_std_pruned_qat_int8_tflite), std_input, kInputSize, std_out, "std");
      run_model(tflite::GetModel(encoder_clf_minmax_pruned_qat_int8_tflite), minmax_input, kInputSize, minmax_out, "minmax");

      // Debug print base model outputs
      Serial.println("🔍 Base model outputs:");
      for (int i = 0; i < kNumClasses; i++) { Serial.print(raw_out[i], 4); Serial.print(" "); } Serial.println("← Raw");
      for (int i = 0; i < kNumClasses; i++) { Serial.print(std_out[i], 4); Serial.print(" "); } Serial.println("← Std");
      for (int i = 0; i < kNumClasses; i++) { Serial.print(minmax_out[i], 4); Serial.print(" "); } Serial.println("← MinMax");

      // Stack inputs for meta
      for (int i = 0; i < kNumClasses; i++) {
        stacked_input[i] = raw_out[i];
        stacked_input[i + kNumClasses] = std_out[i];
        stacked_input[i + 2 * kNumClasses] = minmax_out[i];
      }

      // Debug print stacked input
      Serial.println("🧩 Stacked input to meta model:");
      for (int i = 0; i < kMetaInputSize; i++) {
        Serial.print(stacked_input[i], 4); Serial.print(" ");
      }
      Serial.println();

      // Run meta model
      run_meta_model(stacked_input, meta_out);

      // Final prediction
      int best_class = -1;
      float best_score = -1.0;
      for (int i = 0; i < kNumClasses; i++) {
        if (meta_out[i] > best_score) {
          best_score = meta_out[i];
          best_class = i;
        }
      }

      Serial.print("🌟 Final Prediction (Meta): ");
      Serial.print(label_map[best_class]);
      Serial.print(" | Class Index: ");
      Serial.print(best_class);
      Serial.print(" | Confidence: ");
      Serial.println(best_score, 4);

      window_ready = false;
    }
  }

  delay(20);  // ~50 Hz
}
