#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_sort_init(int* bucket, int* overheads){
  bucket[threadIdx.x] = 0;
  overheads[threadIdx.x] = 0;
}

__global__ void bucket_sort_compute_bucket(int*key, int* bucket, int* overheads, int range){
  atomicAdd(&bucket[key[threadIdx.x]], 1);
  for(int i = key[threadIdx.x]+1; i<range; i++){
    atomicAdd(&overheads[i], 1);
  }
}


__global__ void bucket_sort(int*key, int* bucket, int* overheads){
  for(int i=0; i<bucket[threadIdx.x]; i++){
    key[i + overheads[threadIdx.x]] = threadIdx.x;
  }
}


int main() {
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  int* key;
  cudaMallocManaged(&key, n * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* bucket;
  cudaMallocManaged(&bucket, range * sizeof(int));
  int* overheads;
  cudaMallocManaged(&overheads, range * sizeof(int));
  bucket_sort_init<<<1, range>>>(bucket, overheads);
  bucket_sort_compute_bucket<<<1, n>>>(key, bucket, overheads, range);
  bucket_sort<<<1, range>>>(key, bucket, overheads);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
