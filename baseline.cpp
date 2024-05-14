#include <chrono>
#include <cstring>
#include <iostream>
#include <random>

#define NTIMES 1000

void copy(float* dst, float* src, int size) {
    for (int i = 0; i < size; i++)
        dst[i] = src[i];
}

void fma(float* res, float* a, float* b, float scalar, int size) {
    for (int i = 0; i < size; i++)
        res[i] = a[i] + scalar * b[i];
}

float norm2(float* mas, size_t size) {
    float res = 0.0;
    for (size_t i = 0; i < size; i++)
        res += mas[i] * mas[i];
    return res;
}

void test(float* a, float* b, float* c, float d, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        copy(c, a, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double time_copy = elapsed_seconds.count();


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        fma(c, a, b, d, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_fma = elapsed_seconds.count();


    float norm;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        norm = norm2(c, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_norm = elapsed_seconds.count();


    std::cout << "COPY: " << 8.0 * size * NTIMES / time_copy / 1024.0 / 1024.0 << "MB/s" << std::endl;
    std::cout << "FMA: " << 12.0 * size * NTIMES / time_fma / 1024.0 / 1024.0 << "MB/s" << std::endl;
    std::cout << "NORM: " << time_norm / NTIMES << "s" << std::endl;
    std::cout << "norm2: " << norm << std::endl;
}

void init_rand(float* arr, int size, unsigned int seed) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis(1.0, 2.0);
    gen.seed(seed);
    for (int i = 0; i < size; i++)
        arr[i] = dis(gen);
}

int main(int argc, char* argv[]) {
    int size_arr = 100000;
    float* a = new float[size_arr];
    float* b = new float[size_arr];
    float* c = new float[size_arr];
    float d = 2.0;

    init_rand(a, size_arr, 111);
    init_rand(b, size_arr, 222);

    test(a, b, c, d, size_arr);
}
