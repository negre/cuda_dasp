#pragma once

#include <cuda_dasp/dasp.hpp>
#include <cuda_dasp/cuda_utils_dev.cuh>

namespace cuda_dasp
{

template<typename T>
struct Queue
{
    T* buf;
    uint32_t bufSize;
    uint32_t head;
    int nbElt;
    
    __device__ bool enqueue(T a)
	{
	    int n = atomicAggInc(&nbElt);
	    if(n<=bufSize)
	    {
		buf[(head+n)%bufSize] = a;
		return true;
	    }else{
		atomicAggDec(&nbElt);
	    }
	    return false;
	}
	
    __device__ bool dequeue(T* a)
	{
	    if(nbElt<=0)
		return false;
	    
	    int n = atomicAggDec(&nbElt);
	    if(n>0)
	    {
		uint32_t idx = atomicAggInc(&head);
		*a = buf[idx%bufSize];
		return true;
	    }
	    else
	    {		
		atomicAggInc(&nbElt);
	    }
	    return false;
	}
};


__global__ void computeImages_kernel(cudaTextureObject_t tex_rgb,
                                     cudaTextureObject_t tex_depth,
                                     float4* __restrict__ im_positions,
                                     float4* __restrict__ im_normals,
                                     float4* __restrict__ im_lab,
                                     float2* __restrict__ im_gradient,
                                     float* __restrict__ im_density,
                                     const float radius_meter,
                                     const float fx,
                                     const float fy,
                                     const float cx,
                                     const float cy,
                                     const int rows,
                                     const int cols,
                                     const int step_ch4,
                                     const int step_ch2,
                                     const int step_ch1);

__global__ void initClusters_kernel(float4* __restrict__ positions,
                                    float4* __restrict__ colors,
                                    float4* __restrict__ normals,
                                    float4* __restrict__ crd,
                                    const int2* __restrict__ seeds,
                                    cudaTextureObject_t tex_positions,
                                    cudaTextureObject_t tex_normals,
                                    cudaTextureObject_t tex_lab,
                                    cudaTextureObject_t tex_density,
                                    const int nb_clusters);

__global__ void initClusters2_kernel(float4* __restrict__ positions,
				     float4* __restrict__ colors,
				     float4* __restrict__ normals,
				     float4* __restrict__ crd,
				     ushort2* __restrict__ seeds_queue_buffer,
				     const int2* __restrict__ seeds,
				     cudaTextureObject_t tex_positions,
				     cudaTextureObject_t tex_normals,
				     cudaTextureObject_t tex_lab,
				     cudaTextureObject_t tex_density,
				     float4* __restrict__ acc_positions_sizes,
				     float2* __restrict__ acc_centers,
				     float4* __restrict__ acc_normals,
				     float4* __restrict__ acc_colors_densities,
				     Cov3* __restrict__ acc_shapes,
				     int* __restrict__  im_labels,
				     const int nb_clusters,
				     const int step);
    
__global__ void assignClustersV1_kernel(const float4* __restrict__ positions,
                                        float4* __restrict__ colors,
                                        const float4* __restrict__ normals,
                                        const float4* __restrict__ crd,
                                        cudaTextureObject_t tex_positions,
                                        cudaTextureObject_t tex_normals,
                                        cudaTextureObject_t tex_lab,
                                        cudaTextureObject_t tex_density,
                                        float4* __restrict__ acc_positions_sizes,
                                        float2* __restrict__ acc_centers,
                                        float4* __restrict__ acc_normals,
                                        float4* __restrict__ acc_colors_densities,
                                        Cov3* __restrict__ acc_shapes,
                                        int* __restrict__  im_labels,
                                        const float compactness,
                                        const float normal_weight,
                                        const float lambda,
                                        const float radius_meter,
                                        const int nb_clusters,
                                        const int step,
                                        const int rows,
                                        const int cols); // one thread per pixel and examines a max of nine clusters centers ? or one thread per cluster center and examines pixels in dynamically computed window

__global__ void assignClustersV2_kernel(const float4* __restrict__ positions,
                                        float4* __restrict__ colors,
                                        const float4* __restrict__ normals,
                                        const float4* __restrict__ crd,
                                        cudaTextureObject_t tex_positions,
                                        cudaTextureObject_t tex_normals,
                                        cudaTextureObject_t tex_lab,
                                        cudaTextureObject_t tex_density,
                                        float4* __restrict__ acc_positions_sizes,
                                        float2* __restrict__ acc_centers,
                                        float4* __restrict__ acc_normals,
                                        float4* __restrict__ acc_colors_densities,
                                        Cov3* __restrict__ acc_shapes,
                                        int* __restrict__  im_labels,
                                        const float compactness,
                                        const float normal_weight,
                                        const float lambda,
                                        const float radius_meter,
                                        const int nb_clusters,
                                        const int step,
                                        const int rows,
                                        const int cols);

__global__ void assignClustersV3_kernel(const float4* __restrict__ positions,
                                        float4* __restrict__ colors,
                                        const float4* __restrict__ normals,
                                        const float4* __restrict__ crd,
                                        cudaTextureObject_t tex_positions,
                                        cudaTextureObject_t tex_normals,
                                        cudaTextureObject_t tex_lab,
                                        cudaTextureObject_t tex_density,
                                        int* __restrict__  im_labels,
                                        const float compactness,
                                        const float normal_weight,
                                        const float lambda,
                                        const float radius_meter,
                                        const int nb_clusters,
                                        const int step,
                                        const int rows,
                                        const int cols);

__global__ void updateClustersV1_kernel(float4* __restrict__ positions,
                                        float4* __restrict__ colors,
                                        float4* __restrict__ normals,
                                        float4* __restrict__ crd,
                                        Cov3* __restrict__ shapes,
                                        //int2* __restrict__ seeds,
                                        const float4* __restrict__ acc_positions_sizes,
                                        const float2* __restrict__ acc_centers,
                                        const float4* __restrict__ acc_normals,
                                        const float4* __restrict__ acc_colors_densities,
                                        const Cov3* __restrict__ acc_shapes,
                                        const int nb_clusters); // one thread per superpixel? Or one thread per pixel

__global__ void updateClustersV2_kernel(float4* __restrict__ positions,
                                        float4* __restrict__ colors,
                                        float4* __restrict__ normals,
                                        float4* __restrict__ crd,
                                        Cov3* __restrict__ shapes,
                                        //int2* __restrict__ seeds,
                                        float4* __restrict__ acc_positions_sizes,
                                        float2* __restrict__ acc_centers,
                                        float4* __restrict__ acc_normals,
                                        float4* __restrict__ acc_colors_densities,
                                        Cov3* __restrict__ acc_shapes,
                                        cudaTextureObject_t tex_positions,
                                        cudaTextureObject_t tex_normals,
                                        cudaTextureObject_t tex_lab,
                                        cudaTextureObject_t tex_density,
                                        cudaTextureObject_t tex_labels,
                                        const int nb_clusters,
                                        const int rows,
                                        const int cols);
    
__global__ void daspProcessKernel(float4* __restrict__ positions,
				  float4* __restrict__ colors,
				  float4* __restrict__ normals,
				  float4* __restrict__ crd,
				  cudaTextureObject_t tex_positions,
				  cudaTextureObject_t tex_normals,
				  cudaTextureObject_t tex_lab,
				  cudaTextureObject_t tex_density,
				  float4* __restrict__ acc_positions_sizes,
				  float2* __restrict__ acc_centers,
				  float4* __restrict__ acc_normals,
				  float4* __restrict__ acc_colors_densities,
				  Cov3* __restrict__ acc_shapes,
				  int* __restrict__  im_labels,
				  Queue<ushort2>* queue,
				  unsigned int *syncCounter,
				  const float compactness,
				  const float normal_weight,
				  const float lambda,
				  const float radius_meter,
				  const int nb_clusters,
				  const int step,
				  const int width,
				  const int height);
    
__global__ void renderBoundaryImage_kernel(uchar4* __restrict__ boundary_im,
                                           cudaTextureObject_t tex_labels,
                                           const int step,
                                           const int rows,
                                           const int cols);

__global__ void renderSuperpixelsColorImage_kernel(uchar4* __restrict__ color_im,
                                                   cudaTextureObject_t tex_labels,
                                                   const float4* __restrict__ superpixel_colors,
                                                   const int step,
                                                   const int rows,
                                                   const int cols);

} // cuda_dasp
