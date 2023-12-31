#include "stdio.h"
#include <iostream>
/*
#include <mmintrin.h>  //MMX
#include <xmmintrin.h> //SSE(include mmintrin.h)
#include <emmintrin.h> //SSE2(include xmmintrin.h)
#include <pmmintrin.h> //SSE3(include emmintrin.h)
#include <tmmintrin.h> //SSSE3(include pmmintrin.h)
#include <smmintrin.h> //SSE4.1(include tmmintrin.h)
#include <nmmintrin.h> //SSE4.2(include smmintrin.h)
#include <wmmintrin.h> //AES(include nmmintrin.h)
#include <immintrin.h> //AVX(include wmmintrin.h)
#include <intrin.h>    //(include immintrin.h)*/

//#include <emmintrin.h>
#ifdef LINUX
#include <immintrin.h>
#endif

#ifdef ARM_NEON
#include <arm_neon.h>
#endif

#include <opencv2/opencv.hpp>
#include <future>
using namespace std;
using namespace cv;

//origin
void RGB2Y(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		for (int X = 0; X < Width; X++, LinePS += 3) {
			LinePD[X] = int(0.114 * LinePS[0] + 0.587 * LinePS[1] + 0.299 * LinePS[2]);
		}
	}
}

//int
void RGB2Y_1(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		for (int X = 0; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}

//4路并行
void RGB2Y_2(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 4; X += 4, LinePS += 12) {
			LinePD[X + 0] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
			LinePD[X + 1] = (B_WT * LinePS[3] + G_WT * LinePS[4] + R_WT * LinePS[5]) >> 8;
			LinePD[X + 2] = (B_WT * LinePS[6] + G_WT * LinePS[7] + R_WT * LinePS[8]) >> 8;
			LinePD[X + 3] = (B_WT * LinePS[9] + G_WT * LinePS[10] + R_WT * LinePS[11]) >> 8;
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}

//openmp
void RGB2Y_3(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
#pragma omp parallel for num_threads(4)
		for (int X = 0; X < Width; X++) {
			LinePD[X] = (B_WT * LinePS[0 + X*3] + G_WT * LinePS[1 + X*3] + R_WT * LinePS[2 + X*3]) >> 8;
		}
	}
}

#ifdef LINUX
//sse 一次处理12个
void RGB2Y_4(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)

	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 12; X += 12, LinePS += 36) {
			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3

			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//4
			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//5
			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//6

			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//7
			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//8
			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//9

			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//10
			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//11
			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//12

			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));//13
			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));//14
			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));//15
			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));//16
			__m128i sclaL = _mm_srli_epi16(sumaL, 8);//17 移位操作，相当于普通算法中的>>8
			__m128i sclaH = _mm_srli_epi16(sumaH, 8);//18
			__m128i sclbL = _mm_srli_epi16(sumbL, 8);//19
			__m128i sclbH = _mm_srli_epi16(sumbH, 8);//20
			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//21 当掩模值大于1时，该值除以16求余，如17即为1
			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//22
			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));//23
			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));//24
			__m128i accumL = _mm_or_si128(shftaL, shftbL);//25
			__m128i accumH = _mm_or_si128(shftaH, shftbH);//26
			__m128i h3 = _mm_or_si128(accumL, accumH);//27
													  //__m128i h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}

//sse 一次处理15个
void RGB2Y_5(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)

	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 15; X += 15, LinePS += 45)
		{
			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3

			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));

			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));

			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));

			__m128i p1cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 36))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p2cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 37))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
			__m128i p3cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 38))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));

			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));
			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));
			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));
			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));
			__m128i sumcH = _mm_add_epi16(p3cH, _mm_add_epi16(p1cH, p2cH));

			__m128i sclaL = _mm_srli_epi16(sumaL, 8);
			__m128i sclaH = _mm_srli_epi16(sumaH, 8);
			__m128i sclbL = _mm_srli_epi16(sumbL, 8);
			__m128i sclbH = _mm_srli_epi16(sumbH, 8);
			__m128i sclcH = _mm_srli_epi16(sumcH, 8);

			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 2, 8, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 8, 14, -1, -1, -1, -1));
			__m128i shftcH = _mm_shuffle_epi8(sclcH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 6, 12, -1));
			__m128i accumL = _mm_or_si128(shftaL, shftbL);
			__m128i accumH = _mm_or_si128(shftaH, shftbH);
			__m128i h3 = _mm_or_si128(accumL, accumH);
			h3 = _mm_or_si128(h3, shftcH);
			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}

void debug(__m128i var) {
	uint8_t *val = (uint8_t*)&var;//can also use uint32_t instead of 16_t 
	printf("Numerical: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n",
		val[0], val[1], val[2], val[3], val[4], val[5],
		val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13],
		val[14], val[15]);
}

void debug2(__m256i var) {
	uint8_t *val = (uint8_t*)&var;//can also use uint32_t instead of 16_t 
	printf("Numerical: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n",
		val[0], val[1], val[2], val[3], val[4], val[5],
		val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13],
		val[14], val[15], val[16], val[17], val[18], val[19], val[20], val[21], val[22], val[23], val[24], val[25], val[26], val[27],
		val[28], val[29], val[30], val[31]);
}


// AVX2
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;
constexpr uint16_t B_WT = static_cast<uint16_t>(32768.0 * B_WEIGHT + 0.5);
constexpr uint16_t G_WT = static_cast<uint16_t>(32768.0 * G_WEIGHT + 0.5);
constexpr uint16_t R_WT = static_cast<uint16_t>(32768.0 * R_WEIGHT + 0.5);
static const __m256i weight_vec = _mm256_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT);

void  _RGB2Y(unsigned char* Src, const int32_t Width, const int32_t start_row, const int32_t thread_stride, const int32_t Stride, unsigned char* Dest)
{
	for (int Y = start_row; Y < start_row + thread_stride; Y++)
	{
		//Sleep(1);
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 10; X += 10, LinePS += 30)
		{
			//B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5 G5 R5 B6 
			__m256i temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS + 0)));
			__m256i in1 = _mm256_mulhrs_epi16(temp, weight_vec);

			//B6 G6 R6 B7 G7 R7 B8 G8 R8 B9 G9 R9 B10 G10 R10 B11
			temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS + 15)));
			__m256i in2 = _mm256_mulhrs_epi16(temp, weight_vec);


			//0  1  2  3   4  5  6  7  8  9  10 11 12 13 14 15    16 17 18 19 20 21 22 23 24 25 26 27 28   29 30  31       
			//B1 G1 R1 B2 G2 R2 B3 G3  B6 G6 R6 B7 G7 R7 B8 G8    R3 B4 G4 R4 B5 G5 R5 B6 R8 B9 G9 R9 B10 G10 R10 B11
			__m256i mul = _mm256_packus_epi16(in1, in2); // 2*64位A 2*64位B 2*64位A 2*64位B

			__m256i b1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(
				//  B1 B2 B3 -1, -1, -1  B7  B8  -1, -1, -1, -1, -1, -1, -1, -1,
				0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1, B4 B5 B6 -1, -1  B9 B10 -1, -1, -1, -1, -1, -1
				-1, -1, -1, 1, 4, 7, -1, -1, 9, 12, -1, -1, -1, -1, -1, -1));

			__m256i g1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				// G1 G2 G3 -1, -1  G6 G7  G8  -1, -1, -1, -1, -1, -1, -1, -1, 
				1, 4, 7, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1  G4 G5 -1, -1, -1  G9  G10 -1, -1, -1, -1, -1, -1	
				-1, -1, -1, 2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1));

			__m256i r1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				//  R1 R2 -1  -1  -1  R6  R7  -1, -1, -1, -1, -1, -1, -1, -1, -1,	
				2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, R3 R4 R5 -1, -1, R8 R9  R10 -1, -1, -1, -1, -1, -1
				-1, -1, 0, 3, 6, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1));



			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m256i accum = _mm256_adds_epu8(r1, _mm256_adds_epu8(b1, g1));


			// _mm256_castsi256_si128(accum)
			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0

			// _mm256_extracti128_si256(accum, 1)
			// 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m128i h3 = _mm_adds_epu8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));

			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			int tmpB = (B_WT * LinePS[0]) >> 14 + 1;
			tmpB = max(min(255, tmpB), 0);

			int tmpG = (G_WT * LinePS[1]) >> 14 + 1;
			tmpG = max(min(255, tmpG), 0);

			int tmpR = (R_WT * LinePS[2]) >> 14 + 1;
			tmpR = max(min(255, tmpR), 0);

			int tmp = tmpB + tmpG + tmpR;
			LinePD[X] = max(min(255, tmp), 0);
		}
	}
}

//avx2 
void RGB2Y_6(unsigned char *Src, unsigned char *Dest, int width, int height, int stride)
{
	_RGB2Y(Src, width, 0, height, stride, Dest);
}

//avx2 + std::async异步编程
void RGB2Y_7(unsigned char *Src, unsigned char *Dest, int width, int height, int stride) {
	const int32_t hw_concur = std::min(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
	std::vector<std::future<void>> fut(hw_concur);
	const int thread_stride = (height - 1) / hw_concur + 1;
	int i = 0, start = 0;
	for (; i < std::min(height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, _RGB2Y, Src, width, start, thread_stride, stride, Dest);
	}
	for (int j = 0; j < i; ++j)
		fut[j].wait();
}
#endif

#ifdef ARM_NEON
//ARM NEON加速
void RGB2Y_8(uint8_t *Src, uint8_t *Dest, int Width, int Height, int Stride) {
	uint8_t B_WT = uint8_t(0.114 * 256 + 0.5);
	uint8_t G_WT = uint8_t(0.587 * 256 + 0.5);
	uint8_t R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)
	uint8x8_t rfac = vdup_n_u8(B_WT);  // 转换权值 R
    uint8x8_t gfac = vdup_n_u8(G_WT); // 转换权值 G
    uint8x8_t bfac = vdup_n_u8(R_WT);  // 转换权值 B
	const int size = Width * Height;
	int i = 0;
	uint8_t *pstSrc = Src;
	uint8_t *pstDest = Dest;
	for (i = 0; i < size - 8*3; i += 8*3)
	{
		uint16x8_t temp;
        // uint8x8 表示将64bit寄存器 分成 8 个 8bit
        uint8x8x3_t rgb = vld3_u8(pstSrc); //一次读取3个unit8x8到3个64位寄存器
        uint8x8_t result;
 
        temp = vmull_u8(rgb.val[0], rfac);       // temp=rgb.val[0]*rfac
        temp = vmlal_u8(temp, rgb.val[1], gfac); // temp=temp+rgb.val[1]*gfac
        temp = vmlal_u8(temp, rgb.val[2], bfac); // temp=temp+rgb.val[2]*bfac
 
        result = vshrn_n_u16(temp, 8); // vshrn_n_u16 会在temp做右移8 位的同时将2字节无符号型转成1字节无符号型
        vst1_u8(pstDest, result);      // 转存运算结果到dest
        pstSrc += 8 * 3;
        pstDest += 8;
	}
	for (; i < size; i += 3) 
	{
		*pstDest = (B_WT * pstSrc[0] + G_WT * pstSrc[1] + R_WT * pstSrc[2]) >> 8;
		pstSrc += 3;
        pstDest++;
	}
}

void RGB2Y_9(uint8_t *Src, uint8_t *Dest, int Width, int Height, int Stride)
//void rgb2grayAssembly(const unsigned char *rgbImage,
//                      const int            height,
//                      const int            width,
//                      unsigned char       *grayImage)
{
    int size = Height * Width;
    const unsigned char *src = Src;
    unsigned char *dst = Dest;

    int neonLen = size >> 4;
	uint8_t B_WT = uint8_t(0.114 * 256 + 0.5);
	uint8_t G_WT = uint8_t(0.587 * 256 + 0.5);
	uint8_t R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)
    //int remain  = size - (neonLen << 4); 

#ifdef __aarch64__  // armv8
    // pass
#else   // armv7
     __asm__ volatile(
        // [rW, rW, rW, rW, rW, rW, rW, rW]
        "vmov.u8   d0, #77                  \n"
        // [gW, gW, gW, gW, gW, gW, gW, gW]
        "vmov.u8   d1, #150                 \n"
        // [bW, bW, bW, bW, bW, bW, bW, bW]
        "vmov.u8   d2, #29                  \n"

        "0:                                 \n"

        "pld        [%[src], #512]          \n"

        // load [rgb,rgb,rgb,rgb,rgb,rgb,rgb,rgb] into 
        // [b,b,b,b,b,b,b,b], [g,g,g,g,g,g,g,g], [r,r,r,r,r,r,r,r]
        "vld3.u8   {d3,d4,d5}, [%[src]]!    \n"
        // load next [rgb,rgb,rgb,rgb,rgb,rgb,rgb,rgb] into 
        // [b,b,b,b,b,b,b,b], [g,g,g,g,g,g,g,g], [r,r,r,r,r,r,r,r]
        "vld3.u8   {d6,d7,d8}, [%[src]]!    \n"

        // [r,r,r,r,r,r,r,r] * [rW,rW,rW,rW,rW,rW,rW,rW]
        // bitwidth of results(uint16_t) are 
        // wider than elements in inputs(uint8_t)
        //"vmull.u8	q5, d3, d0              \n"
        //"vmull.u8	q6, d6, d0              \n"
		"vmull.u8	q5, d5, d0              \n"
		"vmull.u8	q6, d8, d0              \n"

        // [g,g,g,g,g,g,g,g] * [gW,gW,gW,gW,gW,gW,gW,gW]
        // bitwidth of results(uint16_t) are 
        // wider than elements in inputs(uint8_t)
        "vmull.u8	q7, d4, d1              \n"
        "vmull.u8	q8, d7, d1              \n"

        // r*rW + [b,b,b,b,b,b,b,b] * [bW,bW,bW,bW,bW,bW,bW,bW]
        //"vmlal.u8	q5, d5, d2              \n"
        //"vmlal.u8	q6, d8, d2              \n"
		"vmlal.u8	q5, d3, d2              \n"
        "vmlal.u8	q6, d6, d2              \n"

        // (r*rW + g*gW + b*bW) >> 8
        "vaddhn.u16	d18, q5, q7             \n"
        "vaddhn.u16	d19, q6, q8             \n"

        "subs       %[neonLen], #1          \n"

        "vst1.u8   {d18-d19}, [%[dst]]!     \n"

        "bgt        0b                      \n"

        :[src]        "+r"(src),
         [dst]        "+r"(dst),
         [neonLen]    "+r"(neonLen)
        :
        :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"
    );
#endif
	int i = neonLen << 4;
	uint8_t *pstSrc = Src + 3 * i;
	uint8_t *pstDest = Dest + i;
	for (; i < size; i += 3) 
	{
		*pstDest = (B_WT * pstSrc[0] + G_WT * pstSrc[1] + R_WT * pstSrc[2]) >> 8;
		pstSrc += 3;
        pstDest++;
	}
}
#endif

bool compare(unsigned char *Src, unsigned char *Dest, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pos = i * width + j;
			if (Src[pos] - Dest[pos] > 1 || Src[pos] - Dest[pos] < -1) return false;
		}
	}
	return true;
}

int main() {
    std::cout << "--------------------start speed_rgb2gray--------------------" << std::endl;
	Mat src = imread("../image/car.jpg");
	int Height = src.rows;
	int Width = src.cols;
    printf("Height:%d,Width:%d\n", Height, Width);
	unsigned char *Src = src.data;
	unsigned char *Base = (unsigned char *)malloc(Height * Width);
	unsigned char *Dest = (unsigned char *)malloc(Height * Width);
	int Stride = Width * 3;
	int Radius = 11;
	int64 st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y(Src, Base, Width, Height, Stride);
	}
	double duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("origin RGB2Y %.5f ms \n", duration);

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_1(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("int RGB2Y_1 %.5f ms \n", duration);
	bool flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_1 correct \n");
	else printf("RGB2Y_1 error \n");

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_2(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("4路并行 RGB2Y_2 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	printf("int RGB2Y_2 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_2 correct \n");
	else printf("RGB2Y_2 error \n");

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_3(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("int RGB2Y_3 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_3 correct \n");
	else printf("RGB2Y_3 error \n");

#ifdef LINUX
    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_4(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("sse 一次处理12个 RGB2Y_4 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_4 correct \n");
	else printf("RGB2Y_4 error \n");

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_5(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("sse 一次处理15个 RGB2Y_5 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_5 correct \n");
	else printf("RGB2Y_5 error \n");

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_6(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("avx2 RGB2Y_6 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_6 correct \n");
	else printf("RGB2Y_6 error \n");

    st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_7(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("avx2 + std::async异步编程 RGB2Y_7 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_7 correct \n");
	else printf("RGB2Y_7 error \n");
#endif

#ifdef ARM_NEON
	st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_8(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("ARM NEON加速 RGB2Y_8 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_8 correct \n");
	else printf("RGB2Y_8 error \n");

	st = cv::getTickCount();
	for (int i = 0; i < 100; i++) {
		RGB2Y_9(Src, Dest, Width, Height, Stride);
	}
	duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 10;
	printf("内嵌汇编 RGB2Y_9 %.5f ms \n", duration);
	flg = compare(Base, Dest, Width, Height);
	if (flg) printf("RGB2Y_9 correct \n");
	else printf("RGB2Y_9 error \n");
#endif

	Mat dst(Height, Width, CV_8UC1, Dest);
	imwrite("../image/res.jpg", dst);
	free(Base);
	free(Dest);
	Base = NULL;
	Dest = NULL;
    std::cout << "--------------------end speed_rgb2gray--------------------" << std::endl;
	return 0;
}