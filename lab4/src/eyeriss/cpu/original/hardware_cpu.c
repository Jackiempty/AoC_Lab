#include "hardware_cpu.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))
void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W,
                     uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
                     uint32_t filter_H, uint32_t filter_W, int8_t* filter,
                     int32_t* bias, uint32_t padding, uint8_t* output,
                     uint32_t scale) {

    /*! <<<========= Implement here =========>>>*/
    //opsum
    int32_t* opsum = (int32_t*)malloc(sizeof(int32_t) * input_H * input_W * filter_N);
    memset(opsum, 0, sizeof(int32_t) * input_H * input_W * filter_N);
    //padding
    int8_t ifmap_pad[input_C][input_H + 2*padding][input_W + 2*padding];
    for (int c = 0; c < input_C; c++) {
        for (int h = 0; h < input_H + 2*padding; h++) {
            for (int w = 0; w < input_W + 2*padding; w++) {
                if (h < padding || h >= input_H + padding || w < padding || w >= input_W + padding) {
                    ifmap_pad[c][h][w] = 0;
                } else {
                    ifmap_pad[c][h][w] = activation[c * input_H * input_W + (h - padding) * input_W + (w - padding)] - 128;
                }
            }
        }
    }
    //conv
    for (int m = 0; m < filter_N; m++) {  // output channel
        for (int h = 0; h < input_H; h++) {
          for (int w = 0; w < input_W; w++) {
            for (int c = 0; c < input_C; c++) {
              for (int r = 0; r < filter_H; r++) {
                for (int s = 0; s < filter_W; s++) {
                  int input_val = ifmap_pad[c][h + r][w + s];
                  int filter_val = filter[m * filter_C * filter_H * filter_W + 
                                          c * filter_H * filter_W + 
                                          r * filter_W + s];
                  opsum[m * input_H * input_W + h * input_W + w] += input_val * filter_val;
                }
              }
            }
          }
        }
      }
    // bias
    for(int i = 0; i < input_H * input_W * filter_N; i++){
        opsum[i] += bias[i / (input_H * input_W)];
    }
    
    // relu post quant
    for(int i = 0; i < input_H * input_W * filter_N; i++){
        opsum[i] = (opsum[i] < 0)? 128 : 
                    ((opsum[i] / (1U << scale)) >= 128)? 255 : opsum[i] / (1U << scale) + 128;
    }
    // Maxpool
    for(int i = 0; i < input_H * input_W * filter_N / 4; i++){
        int start = (i * 2) + input_W * (i / (input_W / 2));
        output[i] = MAX(
            MAX(opsum[start], opsum[start + 1]),
            MAX(opsum[start + input_W], opsum[start + input_W + 1])
        );
    }
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W,
          uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
          uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
          uint32_t padding, uint8_t* output, uint32_t scale) {

    /*! <<<========= Implement here =========>>>*/
    //opsum
    int32_t* opsum = (int32_t*)malloc(sizeof(int32_t) * input_H * input_W * filter_N);
    memset(opsum, 0, sizeof(int32_t) * input_H * input_W * filter_N);
    //padding
    int8_t ifmap_pad[input_C][input_H + 2*padding][input_W + 2*padding];
    for (int c = 0; c < input_C; c++) {
        for (int h = 0; h < input_H + 2*padding; h++) {
            for (int w = 0; w < input_W + 2*padding; w++) {
                if (h < padding || h >= input_H + padding || w < padding || w >= input_W + padding) {
                    ifmap_pad[c][h][w] = 0;
                } else {
                    ifmap_pad[c][h][w] = activation[c * input_H * input_W + (h - padding) * input_W + (w - padding)] - 128;
                }
            }
        }
    }
    //conv
    for (int m = 0; m < filter_N; m++) {  // output channel
        for (int h = 0; h < input_H; h++) {
          for (int w = 0; w < input_W; w++) {
            for (int c = 0; c < input_C; c++) {
              for (int r = 0; r < filter_H; r++) {
                for (int s = 0; s < filter_W; s++) {
                  int input_val = ifmap_pad[c][h + r][w + s];
                  int filter_val = filter[m * filter_C * filter_H * filter_W + 
                                          c * filter_H * filter_W + 
                                          r * filter_W + s];
                  opsum[m * input_H * input_W + h * input_W + w] += input_val * filter_val;
                }
              }
            }
          }
        }
      }
    // bias
    for(int i = 0; i < input_H * input_W * filter_N; i++){
        opsum[i] += bias[i / (input_H * input_W)];
    }
    
    // relu post quant
    for(int i = 0; i < input_H * input_W * filter_N; i++){
        output[i] = (opsum[i] < 0)? 128 : 
                    ((opsum[i] / (1U << scale)) >= 128)? 255 : opsum[i] / (1U << scale) + 128;
    }
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias,
                 uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    int32_t* output_tmp = (int32_t*)malloc(sizeof(int32_t) * output_size);
    memset(output_tmp, 0, sizeof(int32_t) * output_size);
    for(int output_idx = 0; output_idx < output_size; output_idx++){
        for(int input_idx = 0; input_idx < input_size; input_idx++){
            output_tmp[output_idx] += (activation[input_idx] - 128) * filter[output_idx * input_size + input_idx];
        }
        output_tmp[output_idx] += bias[output_idx];
        output_tmp[output_idx] = (output_tmp[output_idx] < 0)? 0 : output_tmp[output_idx];
        output[output_idx] = (output_tmp[output_idx] / (1U << scale)) + 128;
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    int32_t* output_tmp = (int32_t*)malloc(sizeof(int32_t) * output_size);
    memset(output_tmp, 0, sizeof(int32_t) * output_size);
    for(int output_idx = 0; output_idx < output_size; output_idx++){
        for(int input_idx = 0; input_idx < input_size; input_idx++){
            output_tmp[output_idx] += (activation[input_idx] - 128) * filter[output_idx * input_size + input_idx];
        }
        output_tmp[output_idx] += bias[output_idx];
        output[output_idx] = (output_tmp[output_idx] / (1U << scale)) + 128;
    }
};

void quantize(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size,
              uint32_t scale) {
    float fp_scale = 1;
    for (uint32_t i = 0; i < scale; i++) {
        fp_scale *= 2;
    }
    for (uint32_t i = 0; i < size; i++) {
        float t = input_in_DRAM[i] * fp_scale;
        int32_t temp = (int32_t)t + 128;
        // clamp to 0 ~ 255
        if (temp < 0) {
            output_in_DRAM[i] = 0;
        } else if (temp > 255)
            output_in_DRAM[i] = 255;
        else
            output_in_DRAM[i] = (uint8_t)temp;
    }
};

void dequantize(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size,
                uint32_t scale) {
    float fp_scale = 1;
    for (uint32_t i = 0; i < scale; i++) {
        fp_scale *= 2;
    }
    for (uint32_t i = 0; i < size; i++) {
        float temp = *(input_in_DRAM + i) - 128;
        *(output_in_DRAM + i) = temp / fp_scale;
    }
};
