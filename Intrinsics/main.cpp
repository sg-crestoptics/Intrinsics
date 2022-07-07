#include <iostream>
#include <immintrin.h>
#include <stdlib.h>
#include <thread>

const int ARRAY_SIZE = 5120000000;
const int NUMBER_OF_INT_PER_REGISTER = 8;
const int NUMBER_OF_CORES = 8;

void multithreadedOpt(int* const arr1, int* const arr2, int* const vecResult, const int& nIterations)
{
	__m256i a, b, add, st_m2, a_p1, b_p1, add_m1;

	a = _mm256_loadu_epi32(arr1);
	b = _mm256_loadu_epi32(arr2);
	add_m1 = _mm256_add_epi32(a, b);
	a = _mm256_loadu_epi32(arr1 + 8);
	b = _mm256_loadu_epi32(arr2 + 8);

	for (int i = 1; i < nIterations - 1; i++)
	{
		a_p1 = _mm256_loadu_epi32(arr1 + (i + 1) * NUMBER_OF_INT_PER_REGISTER);
		b_p1 = _mm256_loadu_epi32(arr2 + (i + 1) * NUMBER_OF_INT_PER_REGISTER);

		add = _mm256_add_epi32(a, b);
		a = a_p1;
		b = b_p1;

		_mm256_storeu_epi32(vecResult + (i - 1) * NUMBER_OF_INT_PER_REGISTER, add_m1);
		add_m1 = add;
	}
	_mm256_storeu_epi32(vecResult + (nIterations - 2) * NUMBER_OF_INT_PER_REGISTER, add);
	add = _mm256_add_epi32(a, b);
	_mm256_storeu_epi32(vecResult + (nIterations - 1) * NUMBER_OF_INT_PER_REGISTER, add);

}


int main()
{
#if _DEBUG
	std::cout << "Debug mode" << std::endl;
#else
	std::cout << "Release mode" << std::endl;
#endif
	const int nIterations = ARRAY_SIZE / NUMBER_OF_INT_PER_REGISTER;
	const int nIterationPerThread = ARRAY_SIZE / (NUMBER_OF_CORES * NUMBER_OF_INT_PER_REGISTER);
	int* arr1 = new int[ARRAY_SIZE];
	int* arr2 = new int[ARRAY_SIZE];
	int* vecResult = new int[ARRAY_SIZE];
	int* stdResult = new int[ARRAY_SIZE];

	std::thread calculationThreads[NUMBER_OF_CORES];

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		*(arr1 + i) = std::rand();
		*(arr2 + i) = std::rand();
	}


	std::clock_t clockStart, clockEnd, duration;

	// VECTORIAL V1 - BEGIN
	__m256i a1, a2, c;
	clockStart = clock();

	for (int i = 0; i < nIterations; i++)
	{
		a1 = _mm256_loadu_epi32(arr1 + i * 8);
		a2 = _mm256_loadu_epi32(arr2 + i * 8);
		c = _mm256_add_epi32(a1, a2);
		_mm256_storeu_epi32(vecResult + i * 8, c);
	}

	clockEnd = clock();
	duration = clockEnd - clockStart;
	std::cout << "Execution time using intrinsics: " << (float)duration / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
	// VECTORIAL V1- END


	// VECTORIAL V2- BEGIN
	__m256i a, b, add, st_m2, a_p1, b_p1, add_m1;
	clockStart = clock();
	a = _mm256_loadu_epi32(arr1);
	b = _mm256_loadu_epi32(arr2);
	add_m1 = _mm256_add_epi32(a, b);
	a = _mm256_loadu_epi32(arr1 + 8);
	b = _mm256_loadu_epi32(arr2 + 8);

	for (int i = 1; i < nIterations - 1; i++)
	{
		a_p1 = _mm256_loadu_epi32(arr1 + (i + 1) * 8);
		b_p1 = _mm256_loadu_epi32(arr2 + (i + 1) * 8);

		add = _mm256_add_epi32(a, b);
		a = a_p1;
		b = b_p1;

		_mm256_storeu_epi32(vecResult + (i - 1) * 8, add_m1);
		add_m1 = add;
	}
	_mm256_storeu_epi32(vecResult + (nIterations - 2) * 8, add);
	add = _mm256_add_epi32(a, b);
	_mm256_storeu_epi32(vecResult + (nIterations - 1) * 8, add);
	clockEnd = clock();
	duration = clockEnd - clockStart;

	std::cout << "Execution time using intrinsics with LSU and ALU optimization: " << (float)duration / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
	// VECTORIAL V2- END
	// 
	// VECTORIAL V3- BEGIN
	clockStart = clock();
	for (int i = 0; i < NUMBER_OF_CORES; i++)
	{
		calculationThreads[i] = std::thread(multithreadedOpt, arr1 + (i * nIterationPerThread * NUMBER_OF_INT_PER_REGISTER), arr2 + (i * nIterationPerThread * NUMBER_OF_INT_PER_REGISTER), vecResult + (i * nIterationPerThread * NUMBER_OF_INT_PER_REGISTER), nIterationPerThread);
	}
	for (int i = 0; i < NUMBER_OF_CORES; i++)
	{
		calculationThreads[i].join();
	}
	clockEnd = clock();
	duration = clockEnd - clockStart;
	std::cout << "Execution time using multithreading and intrinsics with LSU and ALU optimization: " << (float)duration / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
	// VECTORIAL V3- END

	// STANDARD - BEGIN
	clockStart = clock();
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		stdResult[i] = arr1[i] + arr2[i];
		//*(stdResult + i) = *(arr1 + i) + *(arr2 + i);
	}
	clockEnd = clock();
	duration = clockEnd - clockStart;

	std::cout << "Execution time with standard loop: " << (float)duration / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
	// STANDARD - END

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		if (*(stdResult + i) != *(vecResult + i))
			std::cout << "Errore elemento: " << i << std::endl;
	}
	delete[] arr1, arr2, vecResult, stdResult;

}