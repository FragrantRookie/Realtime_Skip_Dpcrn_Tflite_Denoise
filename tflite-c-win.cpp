#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#define DR_WAV_IMPLEMENTATION
#include <tensorflow\lite\c\c_api.h>

#ifndef MIN
#define MIN(A, B)        ((A) < (B) ? (A) : (B))
#endif

#include "pocketfft_hdronly.h"
#include <iostream>
#include "sin.h"
#include <cmath>


#define block_len		512
#define block_shift		256
#define fft_out_size    (block_len / 2 + 1)
#define BLOCK_SIZE      256
#define gru_size       128 * 32 * 2

using namespace pocketfft;
using namespace std;
typedef complex<double> cpx_type;

typedef unsigned char BYTE;
using namespace std;
typedef std::complex<double> Complex;

struct trg_engine {
    float in_buffer[block_len] = { 0 };
    float out_buffer[block_len] = { 0 };
    float states_1[gru_size] = { 0 };

    TfLiteTensor* input_details_1[2];
    const TfLiteTensor* output_details_1[4];

    TfLiteInterpreter* interpreter_dpcrn;
    TfLiteModel* model_dpcrn;
};

void audio_denoise(char* in_file, char* out_file);
#define S16_INPUT_RAW


void f32_16khz_to_s16_16khz(float* in, short* out, int count)
{
    for (int i = 0; i < count / BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
            out[j] = in[j] * 32767.f;

        in += BLOCK_SIZE;
        out += BLOCK_SIZE;
    }
}

void calc_mag_phase(vector<cpx_type> fft_res, float* in_mag, float* in_phase, float* inp, int count)
{
    for (int i = 0; i < count; i++)
    {
        in_mag[i] = fft_res[i].real();
        in_phase[i] = fft_res[i].imag();
        inp[i * 3] = in_mag[i];
        inp[i * 3 + 1] = in_phase[i];
        inp[i * 3 + 2] = 2 * log(sqrtf(fft_res[i].real() * fft_res[i].real() + fft_res[i].imag() * fft_res[i].imag()));
    }
}

void tflite_create(trg_engine* engine)
{

    engine->model_dpcrn = TfLiteModelCreateFromFile("model/dpcrn_quant.tflite");

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    engine->interpreter_dpcrn = TfLiteInterpreterCreate(engine->model_dpcrn, options);
    if (engine->interpreter_dpcrn == nullptr) {
        printf("Failed to create interpreter");
        return;
    }

    if (TfLiteInterpreterAllocateTensors(engine->interpreter_dpcrn) != kTfLiteOk) {
        printf("Failed to allocate tensors!");
        return;
    }
    engine->input_details_1[0] = TfLiteInterpreterGetInputTensor(engine->interpreter_dpcrn, 0);
    engine->input_details_1[1] = TfLiteInterpreterGetInputTensor(engine->interpreter_dpcrn, 1);
    engine->output_details_1[0] = TfLiteInterpreterGetOutputTensor(engine->interpreter_dpcrn, 0);
    engine->output_details_1[1] = TfLiteInterpreterGetOutputTensor(engine->interpreter_dpcrn, 1);
    engine->output_details_1[2] = TfLiteInterpreterGetOutputTensor(engine->interpreter_dpcrn, 2);
    engine->output_details_1[3] = TfLiteInterpreterGetOutputTensor(engine->interpreter_dpcrn, 3);
}

void tflite_destroy(trg_engine* engine)
{
    TfLiteModelDelete(engine->model_dpcrn);
}

void tflite_infer(trg_engine* engine)
{

    float in_mag[block_len / 2 + 1] = { 0 };
    float in_phase[block_len / 2 + 1] = { 0 };
    float inp[(block_len / 2 + 1) * 3] = { 0 };
    float estimated_block[block_len];

    double fft_in[block_len];
    vector<cpx_type> fft_res(block_len);
    vector<cpx_type> spec(block_len);

    shape_t shape;
    shape.push_back(block_len);
    shape_t axes;
    axes.push_back(0);
    stride_t stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));


    for (int i = 0; i < block_len; i++)
    {
        fft_in[i] = engine->in_buffer[i] * win_sin[i];
    }

    r2c(shape, stridel, strideo, axes, FORWARD, fft_in, fft_res.data(), 1.0);

    calc_mag_phase(fft_res, in_mag, in_phase, inp, fft_out_size);

    memcpy(engine->input_details_1[0]->data.f, inp, fft_out_size * 3 * sizeof(float));

    memcpy(engine->input_details_1[1]->data.f, engine->states_1, gru_size * sizeof(float));


    if (TfLiteInterpreterInvoke(engine->interpreter_dpcrn) != kTfLiteOk) {
        printf("Error invoking detection model");
    }

    float* out_mask = engine->output_details_1[0]->data.f;
    float* out_cos = engine->output_details_1[1]->data.f;
    float* out_sin = engine->output_details_1[2]->data.f;
    memcpy(engine->states_1, engine->output_details_1[3]->data.f, gru_size * sizeof(float));


    for (int i = 0; i < fft_out_size; i++) {
        fft_res[i] = complex<double>{ fft_res[i].real() * out_mask[i] * out_cos[i] - fft_res[i].imag() * out_mask[i] * out_sin[i], fft_res[i].real() * out_mask[i] * out_sin[i] + fft_res[i].imag() * out_mask[i] * out_cos[i] };
    }


    c2r(shape, strideo, stridel, axes, BACKWARD, fft_res.data(), fft_in, 1.0);
    for (int i = 0; i < block_len; i++)
        estimated_block[i] = (fft_in[i] / block_len) * win_sin[i];

    memmove(engine->out_buffer, engine->out_buffer + block_shift, (block_len - block_shift) * sizeof(float));
    memset(engine->out_buffer + (block_len - block_shift), 0, block_shift * sizeof(float));
    for (int i = 0; i < block_len; i++)
        engine->out_buffer[i] += estimated_block[i];
}

void trg_denoise(trg_engine* engine, float* samples, float* out, int sampleCount)
{
    int num_blocks = sampleCount / block_shift;

    for (int idx = 0; idx < num_blocks; idx++)
    {
        memmove(engine->in_buffer, engine->in_buffer + block_shift, (block_len - block_shift) * sizeof(float));
        memcpy(engine->in_buffer + (block_len - block_shift), samples, block_shift * sizeof(float));
        tflite_infer(engine);
        memcpy(out, engine->out_buffer, block_shift * sizeof(float));
        samples += block_shift;
        out += block_shift;
    }
}

void s16_16khz_to_f32_16khz(short* in, float* out, int count)
{
    for (int j = 0; j < count; j++)
        out[j] = in[j] / 32767.f;
}

void floatTobytes(float* data, BYTE* bytes, int dataLength)
{
    int i;
    size_t length = sizeof(float) * dataLength;
    BYTE* pdata = (BYTE*)data;
    for (i = 0; i < length; i++)
    {
        bytes[i] = *pdata++;
    }
    return;
}

float bytesToFloat(BYTE* bytes)
{
    return *((float*)bytes);
}

void shortToByte(short* data, BYTE* bytes, int dataLength)
{
    for (int i = 0; i < dataLength; i++) {
        bytes[i * 2] = (BYTE)(0xff & data[i]);
        bytes[i * 2 + 1] = (BYTE)((0xff00 & data[i]) >> 8);
    }
    return;
}

short bytesToShort(BYTE* bytes)
{
    short addr = bytes[0] & 0xFF;
    addr |= ((bytes[1] << 8) & 0xFF00);
    return addr;
}

void ByteToChar(BYTE* bytes, char* chars, unsigned int count) {
    for (unsigned int i = 0; i < count; i++)
        chars[i] = (char)bytes[i];
}

void audio_denoise(char* in_file, char* out_file, trg_engine* eng1)
{
    uint32_t sampleRate = 16000;
    uint64_t inSampleCount = 0;

#ifdef S16_INPUT_RAW
    inSampleCount = BLOCK_SIZE * 2;
    short inBuffer_s16_16k[BLOCK_SIZE];
    FILE* fp = fopen(in_file, "rb");
    FILE* fpf_out = fopen(out_file, "wb");
    if (!fp)
    {
        printf("Please change input file path.\n");
        return;
    }
    while (!feof(fp))
    {
        BYTE* inBuffer_byte_16k = (BYTE*)malloc(inSampleCount * sizeof(BYTE));
        fread(inBuffer_byte_16k, inSampleCount, 1, fp);

        for (int i = 0, j = 0; i < inSampleCount; i = i + 2)
        {
            inBuffer_s16_16k[j] = bytesToShort(inBuffer_byte_16k);
            inBuffer_byte_16k = inBuffer_byte_16k + 2;
            j++;
        }
        float f32_sample[BLOCK_SIZE];
        float outBuffer_f32_16khz[BLOCK_SIZE];
        short out_s16_16khz[BLOCK_SIZE];
        BYTE out_bytes[BLOCK_SIZE * 2];
        char out_chars[BLOCK_SIZE * 2];

        s16_16khz_to_f32_16khz(inBuffer_s16_16k, f32_sample, BLOCK_SIZE);
        trg_denoise(eng1, f32_sample, outBuffer_f32_16khz, BLOCK_SIZE);
        f32_16khz_to_s16_16khz(outBuffer_f32_16khz, out_s16_16khz, BLOCK_SIZE);
        shortToByte(out_s16_16khz, out_bytes, BLOCK_SIZE);

        fwrite(out_bytes, BLOCK_SIZE * 2, 1, fpf_out);
        ByteToChar(out_bytes, out_chars, BLOCK_SIZE * 2);
        inBuffer_byte_16k++;
    }
    tflite_destroy(eng1);
    fclose(fp);
#else
    float* inBuffer = wavRead_scalar(in_file, &sampleRate, &inSampleCount);
#endif
}


int main() {

    printf("audio denoise by dpcrn_tflite model.\n");
    char* in_file = (char*)"440C020A_mix.wav";
    char* out_file = (char*)"440C020A_mix.pcm";
    trg_engine eng1;
    tflite_create(&eng1);
    audio_denoise(in_file, out_file, &eng1);
    printf("press any key to exit.\n");

    return 0;
}