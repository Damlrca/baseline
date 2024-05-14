#include <chrono>
#include <iostream>
#include <random>

#include <algorithm>
#include <cmath>

#include <riscv-vector.h>

#define NTIMES 10000
#define V_SIZE 100009

void copy(float* dst, float* src, int size) {
    for (int i = 0; i < size; i++)
        dst[i] = src[i];
}

void copy_v(float* dst, float* src, int size) {
	size_t vl = vsetvlmax_e32m4();
	vfloat32m4_t v_src;
	while (size > 0) {
		vl = vsetvl_e32m4(size);
		v_src = vle_v_f32m4(src, vl);
		vse_v_f32m4(dst, v_src, vl);
		dst += vl;
		src += vl;
		size -= vl;
	}
}

void fma(float* res, float* a, float* b, float scalar, int size) {
    for (int i = 0; i < size; i++)
        res[i] = a[i] + scalar * b[i];
}

void fma_v(float* res, float* a, float* b, float scalar, int size) {
	size_t vl = vsetvlmax_e32m4();
	vfloat32m4_t v_a, v_b, v_res;
	while (size > 0) {
		vl = vsetvl_e32m4(size);
		v_a = vle_v_f32m4(a, vl);
		v_b = vle_v_f32m4(b, vl);
		v_res = vfmacc_vf_f32m4(v_a, scalar, v_b, vl);
		vse_v_f32m4(res, v_res, vl);
		a += vl;
		b += vl;
		res += vl;
		size -= vl;
	}
}

float norm2(float* mas, size_t size) {
    float res = 0.0;
    for (size_t i = 0; i < size; i++)
        res += mas[i] * mas[i];
    return res;
}

float norm2_v(float* mas, size_t size) {
	float res = 0.0f;
	size_t vl = vsetvlmax_e32m4();
	vfloat32m4_t v_mas;
	// vfloat32m4_t v_mul;
	vfloat32m4_t v_summ = vfmv_v_f_f32m4(0.0f, vl);
	while (size > vl) {
		v_mas = vle_v_f32m4(mas, vl);
		
		// v_mul = vfmul_vv_f32m4(v_mas, v_mas, vl);
		// v_summ = vfadd_vv_f32m4(v_summ, v_mul, vl);
		v_summ = vfmacc_vv_f32m4(v_summ, v_mas, v_mas, vl);
		
		mas += vl;
		size -= vl;
	}
	
	vfloat32m1_t v_res = vfmv_v_f_f32m1(0.0f, vsetvlmax_e32m1());
	v_res = vfredsum_vs_f32m4_f32m1(v_res, v_summ, v_res, vl);
	
	vl = vsetvl_e32m4(size);
	v_mas = vle_v_f32m4(mas, vl);
	v_mas = vfmul_vv_f32m4(v_mas, v_mas, vl);
	v_res = vfredsum_vs_f32m4_f32m1(v_res, v_mas, v_res, vl);
	vse_v_f32m1(&res, v_res, 1);
	return res;
}

void test_copy(float* a, float* b, float* b_v, int size) {
	auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        copy(b, a, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double time_copy = elapsed_seconds.count();
	
	start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        copy_v(b_v, a, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_copy_v = elapsed_seconds.count();
	
	float mx_diff = 0.0f;
	for (int i = 0; i < size; i++)
		mx_diff = std::max(mx_diff, std::abs(b[i] - b_v[i]));
	
	std::cout << "  COPY: " << 8.0 * size * NTIMES / time_copy / 1024.0 / 1024.0 << "MB/s" << std::endl;
	std::cout << "COPY_V: " << 8.0 * size * NTIMES / time_copy_v / 1024.0 / 1024.0 << "MB/s" << std::endl;
	std::cout << "copy mx diff: " << mx_diff << std::endl;
}

void test_fma(float* a, float* b, float* c, float* c_v, float d, int size) {
	auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        fma(c, a, b, d, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double time_fma = elapsed_seconds.count();
	
	start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        fma_v(c_v, a, b, d, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_fma_v = elapsed_seconds.count();
	
	float mx_diff = 0.0f;
	for (int i = 0; i < size; i++)
		mx_diff = std::max(mx_diff, std::abs(c[i] - c_v[i]));
	
	std::cout << "   FMA: " << 12.0 * size * NTIMES / time_fma / 1024.0 / 1024.0 << "MB/s" << std::endl;
	std::cout << " FMA_V: " << 12.0 * size * NTIMES / time_fma_v / 1024.0 / 1024.0 << "MB/s" << std::endl;
	std::cout << " fma mx diff: " << mx_diff << std::endl;
}

void test_norm2(float* c, int size) {
	float norm;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        norm = norm2(c, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double time_norm = elapsed_seconds.count();
	
	float norm_v;
	start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        norm_v = norm2_v(c, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_norm_v = elapsed_seconds.count();
	
	std::cout << "  NORM: " << time_norm / NTIMES << "s" << std::endl;
	std::cout << "NORM_V: " << time_norm_v / NTIMES << "s" << std::endl;
	std::cout << "norm  : " << norm << std::endl;
	std::cout << "norm_v: " << norm << std::endl;
	std::cout << "norm diff: " << std::abs(norm - norm_v) << std::endl;
}

void init_rand(float* arr, int size, unsigned int seed) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis(1.0, 2.0);
    gen.seed(seed);
    for (int i = 0; i < size; i++)
        arr[i] = dis(gen);
}

int main(int argc, char* argv[]) {
	int threads_num = omp_get_max_threads();
	omp_set_num_threads(threads_num);
	std::cout << "threads_num: " << threads_num << std::endl;
	
    int size_arr = V_SIZE;
    float* a = new float[size_arr];
    float* b = new float[size_arr];
    float* b_v = new float[size_arr];
    float* c = new float[size_arr];
    float* c_v = new float[size_arr];
    float d = 2.0;
	
    init_rand(a, size_arr, 111);
	test_copy(a, b, b_v, size_arr);
	
	init_rand(a, size_arr, 222);
	init_rand(b, size_arr, 555);
	test_fma(a, b, c, c_v, d, size_arr);
	
	test_norm2(c, size_arr);
}
