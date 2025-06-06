#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;
typedef vector<vector<float>> matrix;

__global__ void compute_pressure(float* p, float* b, int nx, int ny, double dx, double dy, float* pn) {
  int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIndex >= nx * ny) return;
  int i = threadIndex % nx;
  int j = threadIndex / nx;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    float pn_i_plus = pn[j*nx+i+1];
    float pn_i_minus = pn[j*nx+i-1];
    float pn_j_plus = pn[(j+1)*nx+i];
    float pn_j_minus = pn[(j-1)*nx+i];
    p[j*nx+i] = (dy*dy * (pn_i_plus + pn_i_minus) +\
                           dx*dx * (pn_j_plus + pn_j_minus) -\
                           b[j*nx+i] * dx*dx * dy*dy)\
                          / (2 * (dx*dx + dy*dy));
  }
}

__global__ void compute_velocity(float* u, float* v, float* un, float* vn, float* p, int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
  int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIndex >= nx * ny) return;
  int i = threadIndex % nx;
  int j = threadIndex / nx;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i - 1])\
                               - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j - 1)*nx+i])\
                               - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])\
                               + nu * dt / (dx*dx) * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])\
                               + nu * dt / (dy*dy) * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
    v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i - 1])\
                            - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j - 1)*nx+i])\
                            - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])\
                            + nu * dt / (dx*dx) * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])\
                            + nu * dt / (dy*dy) * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
  }
}



int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  cudaError_t err;
  float* u;
  err = cudaMallocManaged(&u, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for u: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  float* v;
  err = cudaMallocManaged(&v, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for u: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }  float* p;
  err = cudaMallocManaged(&p, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for p: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  float* b;
  cudaMallocManaged(&b, ny*nx*sizeof(float));
    if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for b: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  float* un;
  err = cudaMallocManaged(&un, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for un: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  float* vn;
  err = cudaMallocManaged(&vn, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for vn: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  float* pn;
  err = cudaMallocManaged(&pn, ny*nx*sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "cudaMallocManaged failed for pn: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j*nx+i] = 0;
      v[j*nx+i] = 0;
      p[j*nx+i] = 0;
      b[j*nx+i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        b[j*nx+i] = rho * (1 / dt *\
                    ((u[(j)*nx+(i+1)] - u[(j)*nx+(i-1)]) / (2 * dx) + (v[(j+1)*nx+(i)] - v[(j-1)*nx+(i)]) / (2 * dy)) -\
                    ((u[(j)*nx+(i+1)] - u[(j)*nx+(i-1)]) / (2 * dx))*((u[(j)*nx+(i+1)] - u[(j)*nx+(i-1)]) / (2 * dx)) - 2 * ((u[(j+1)*nx+(i)] - u[(j-1)*nx+(i)]) / (2 * dy) *\
                     (v[(j)*nx+(i+1)] - v[(j)*nx+(i-1)]) / (2 * dx)) - ((v[(j+1)*nx+(i)] - v[(j-1)*nx+(i)]) / (2 * dy))*((v[(j+1)*nx+(i)] - v[(j-1)*nx+(i)]) / (2 * dy)));
      }
    }
    for (int it=0; it<nit; it++) {
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          pn[j*nx+i] = p[j*nx+i];
        }
      }
      compute_pressure<<<(ny*nx + 1025) / 1024, 1024>>>(p, b, nx, ny, dx, dy, pn);
      cudaDeviceSynchronize();
      for (int j=0; j<ny; j++) {
        // Compute p[(j)*nx+(0)] and p[(j)*nx+(nx-1)]
        p[j*nx] = p[j*nx+1];
        p[j*nx+nx-1] = p[j*nx + nx-2];
      }
      for (int i=0; i<nx; i++) {
        // Compute p[(0)*nx+(i)] and p[(ny-1)*nx+(i)]
        p[0*nx+i] = p[1*nx+i];
        p[(ny-1)*nx+i] = 0;
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[(j)*nx+(i)] = u[(j)*nx+(i)];
	      vn[(j)*nx+(i)] = v[(j)*nx+(i)];
      }
    }
    compute_velocity<<<(ny*nx + 1025) / 1024, 1024>>>(u, v, un, vn, p, nx, ny, dx, dy, dt, rho, nu);
    cudaDeviceSynchronize();
    for (int j=0; j<ny; j++) {
      // Compute u[(j)*nx+(0)], u[(j)*nx+(nx-1)], v[(j)*nx+(0)], v[(j)*nx+(nx-1)]
      u[(j)*nx+(0)] = 0;
      u[(j)*nx+(nx-1)] = 0;
      v[(j)*nx+(0)] = 0;
      v[(j)*nx+(nx-1)] = 0;
    }
    for (int i=0; i<nx; i++) {
      // Compute u[(0)*nx+(i)], u[(ny-1)*nx+(i)], v[(0)*nx+(i)], v[(ny-1)*nx+(i)]
      u[(0)*nx+(i)] = 0;
      u[(ny-1)*nx+(i)] = 1;
      v[(0)*nx+(i)] = 0;
      v[(ny-1)*nx+(i)] = 0;
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[(j)*nx+(i)] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[(j)*nx+(i)] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j*nx+i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
