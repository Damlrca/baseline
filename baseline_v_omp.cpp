#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <omp.h>

#include <riscv-vector.h>

#define NTIMES 1000

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

void copy_omp(float* dst, float* src, int size) {
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int thread_num = omp_get_num_threads();
		int part_size = size / thread_num;
		int offset = part_size * id;
		if (id == thread_num - 1)
			part_size = size - offset;
		copy_v(dst + offset, src + offeset, part_size);
	}
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

void test(float* a, float* b, float* c, float d, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        copy_omp(c, a, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double time_copy = elapsed_seconds.count();


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        fma_v(c, a, b, d, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    double time_fma = elapsed_seconds.count();


    float norm;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NTIMES; i++)
        norm = norm2_v(c, size);
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
