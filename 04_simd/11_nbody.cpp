#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xvec = _mm512_load_ps(x);
    __m512 yvec = _mm512_load_ps(y);
    __m512i ivec = _mm512_set1_epi32(i);
    __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __mmask16 mask = _mm512_cmp_epi32_mask(idx, ivec, _MM_CMPINT_NE);
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 rxvec = _mm512_sub_ps(xivec, xvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yvec);
    __m512 rx2vec = _mm512_mul_ps(rxvec, rxvec);
    __m512 ry2vec = _mm512_mul_ps(ryvec, ryvec);
    __m512 d2vec = _mm512_add_ps(rx2vec, ry2vec);
    __m512 rinvvect = _mm512_rsqrt14_ps(d2vec);
    __m512 rinvvect2 = _mm512_mul_ps(rinvvect, rinvvect);
    __m512 rinvvect3 = _mm512_mul_ps(rinvvect2, rinvvect);
    __m512 mvec = _mm512_load_ps(m);
    __m512 rinvvect3m = _mm512_mul_ps(rinvvect3, mvec);
    __m512 fxposvec = _mm512_mul_ps(rinvvect3m, rxvec);
    __m512 fyposvec = _mm512_mul_ps(rinvvect3m, ryvec);
    __m512 zerovec = _mm512_setzero_ps();
    fxposvec = _mm512_mask_blend_ps(mask, zerovec, fxposvec);
    fyposvec = _mm512_mask_blend_ps(mask, zerovec, fyposvec);
    
    
    fx[i] -= _mm512_reduce_add_ps(fxposvec);
    fy[i] -= _mm512_reduce_add_ps(fyposvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
