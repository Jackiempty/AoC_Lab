#include "hardware_cpu.h"

#include <stdio.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
                     uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
                     uint32_t padding, uint8_t* output, uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    int padded_H = input_H + 2 * padding;
    int padded_W = input_W + 2 * padding;
    int8_t* ifmap_pad = (int8_t*)malloc(input_C * padded_H * padded_W);
    for (int c = 0; c < input_C; c++) {
        for (int h = 0; h < padded_H; h++) {
            for (int w = 0; w < padded_W; w++) {
                int idx = c * padded_H * padded_W + h * padded_W + w;
                if (h < padding || h >= input_H + padding || w < padding || w >= input_W + padding) {
                    ifmap_pad[idx] = 0;
                } else {
                    int src_idx = c * input_H * input_W + (h - padding) * input_W + (w - padding);
                    ifmap_pad[idx] = activation[src_idx] - 128;
                }
            }
        }
    }
    int8_t* re_filter = (int8_t*)malloc(filter_N * filter_C * filter_H * filter_W);
    for (int m = 0; m < filter_N; m++) {
        for (int r = 0; r < filter_H; r++) {
            for (int s = 0; s < filter_W; s++) {
                for (int c = 0; c < filter_C; c++) {
                    re_filter[m * filter_H * filter_W * filter_C + r * filter_W * filter_C + s * filter_C + c] =
                        filter[m * filter_C * filter_H * filter_W + c * filter_H * filter_W + r * filter_W + s];
                }
            }
        }
    }
    // opsum
    int32_t* opsum = (int32_t*)malloc(sizeof(int32_t) * input_H * input_W * filter_N);
    memset(opsum, 0, sizeof(int32_t) * input_H * input_W * filter_N);
    for (int m = 0; m < filter_N; m++) {
        for (int h = 0; h < input_H; h++) {
            for (int w = 0; w < input_W; w++) {
                int32_t acc = 0;
                for (int r = 0; r < filter_H; r++) {
                    for (int s = 0; s < filter_W; s++) {
                        for (int c = 0; c < filter_C; c++) {
                            int act_val = ifmap_pad[c * padded_H * padded_W + (h + r) * padded_W + (w + s)];
                            int filt_val = re_filter[m * filter_H * filter_W * filter_C + r * filter_W * filter_C +
                                                     s * filter_C + c];
                            acc += act_val * filt_val;
                        }
                    }
                }
                opsum[m * input_H * input_W + h * input_W + w] = acc + bias[m];
            }
        }
    }

    for (int i = 0; i < input_H * input_W * filter_N; i++) {
        int32_t sum = opsum[i];
        sum = (sum < 0) ? 0 : sum;
        sum = sum >> scale;
        sum = (sum >= 128) ? 255 : sum + 128;
        opsum[i] = sum;
    }
    int pool_H = input_H / 2;
    int pool_W = input_W / 2;

    for (int m = 0; m < filter_N; m++) {
        for (int h = 0; h < pool_H; h++) {
            for (int w = 0; w < pool_W; w++) {
                int base = m * input_H * input_W + (h * 2) * input_W + (w * 2);
                int a = opsum[base];
                int b = opsum[base + 1];
                int c = opsum[base + input_W];
                int d = opsum[base + input_W + 1];
                output[m * pool_H * pool_W + h * pool_W + w] = MAX(MAX(a, b), MAX(c, d));
            }
        }
    }
    free(ifmap_pad);
    free(re_filter);
    free(opsum);
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
          uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias, uint32_t padding,
          uint8_t* output, uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    int padded_H = input_H + 2 * padding;
    int padded_W = input_W + 2 * padding;
    int8_t* ifmap_pad = (int8_t*)malloc(input_C * padded_H * padded_W);
    for (int c = 0; c < input_C; c++) {
        for (int h = 0; h < padded_H; h++) {
            for (int w = 0; w < padded_W; w++) {
                int idx = c * padded_H * padded_W + h * padded_W + w;
                if (h < padding || h >= input_H + padding || w < padding || w >= input_W + padding) {
                    ifmap_pad[idx] = 0;
                } else {
                    int src_idx = c * input_H * input_W + (h - padding) * input_W + (w - padding);
                    ifmap_pad[idx] = activation[src_idx] - 128;
                }
            }
        }
    }
    int8_t* re_filter = (int8_t*)malloc(filter_N * filter_C * filter_H * filter_W);
    for (int m = 0; m < filter_N; m++) {
        for (int r = 0; r < filter_H; r++) {
            for (int s = 0; s < filter_W; s++) {
                for (int c = 0; c < filter_C; c++) {
                    re_filter[m * filter_H * filter_W * filter_C + r * filter_W * filter_C + s * filter_C + c] =
                        filter[m * filter_C * filter_H * filter_W + c * filter_H * filter_W + r * filter_W + s];
                }
            }
        }
    }
    // opsum
    int32_t* opsum = (int32_t*)malloc(sizeof(int32_t) * input_H * input_W * filter_N);
    memset(opsum, 0, sizeof(int32_t) * input_H * input_W * filter_N);
    for (int m = 0; m < filter_N; m++) {
        for (int h = 0; h < input_H; h++) {
            for (int w = 0; w < input_W; w++) {
                int32_t acc = 0;
                for (int r = 0; r < filter_H; r++) {
                    for (int s = 0; s < filter_W; s++) {
                        for (int c = 0; c < filter_C; c++) {
                            int act_val = ifmap_pad[c * padded_H * padded_W + (h + r) * padded_W + (w + s)];
                            int filt_val = re_filter[m * filter_H * filter_W * filter_C + r * filter_W * filter_C +
                                                     s * filter_C + c];
                            acc += act_val * filt_val;
                        }
                    }
                }
                opsum[m * input_H * input_W + h * input_W + w] = acc + bias[m];
            }
        }
    }

    for (int i = 0; i < input_H * input_W * filter_N; i++) {
        int32_t sum = opsum[i];
        sum = (sum < 0) ? 0 : sum;
        sum = sum >> scale;
        sum = (sum >= 128) ? 255 : sum + 128;
        output[i] = sum;
    }

    free(ifmap_pad);
    free(re_filter);
    free(opsum);
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
                 int32_t* bias, uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    const int BATCH = 64;
    for (int output_idx = 0; output_idx < output_size; ++output_idx) {
        int32_t output_tmp = bias[output_idx];
        for (int ii = 0; ii < input_size; ii += BATCH) {
            int i_end = (ii + BATCH > input_size) ? input_size : ii + BATCH;
            for (int batch_idx = ii; batch_idx < i_end; ++batch_idx) {
                output_tmp += filter[output_idx * input_size + batch_idx] * (activation[batch_idx] - 128);
            }
        }
        output_tmp = (output_tmp < 0) ? 0 : output_tmp;
        output[output_idx] = (output_tmp >> scale) + 128;
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
            int32_t* bias, uint32_t scale) {
    /*! <<<========= Implement here =========>>>*/
    const int BATCH = 64;
    for (int output_idx = 0; output_idx < output_size; ++output_idx) {
        int32_t output_tmp = bias[output_idx];
        for (int ii = 0; ii < input_size; ii += BATCH) {
            int i_end = (ii + BATCH > input_size) ? input_size : ii + BATCH;
            for (int batch_idx = ii; batch_idx < i_end; ++batch_idx) {
                output_tmp += filter[output_idx * input_size + batch_idx] * (activation[batch_idx] - 128);
            }
        }
        output[output_idx] = (output_tmp >> scale) + 128;
    }
};

void quantize(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size, uint32_t scale) {
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

void dequantize(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size, uint32_t scale) {
    float fp_scale = 1;
    for (uint32_t i = 0; i < scale; i++) {
        fp_scale *= 2;
    }
    for (uint32_t i = 0; i < size; i++) {
        float temp = *(input_in_DRAM + i) - 128;
        *(output_in_DRAM + i) = temp / fp_scale;
    }
};