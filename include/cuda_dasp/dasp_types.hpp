#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda_dasp/matrix_types.h>

namespace cuda_dasp
{

struct Superpixels // use one big array of float instead? pos1_crd1_normals1_shape1_colors1_size1 pos2_crd2_normals2_shape2_colors2_size2 ...
{
    thrust::device_vector<float4> positions;
    thrust::device_vector<float4> crd; // pixel centers (x, y), radius, densities
    thrust::device_vector<float4> normals;
    //thrust::device_vector<float3> orientations;
    thrust::device_vector<Cov3> shapes;
    //thrust::devicd_vector<float2> dims;
    //thrust::device_vector<float> confidences;
    //thrust::device_vector<int2> stamps;
    thrust::device_vector<float4> colors;

    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<Cov3>::iterator,
                                               thrust::device_vector<float4>::iterator>> iterator;
    iterator begin()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions.begin(),
                                                            crd.begin(),
                                                            normals.begin(),
                                                            shapes.begin(),
                                                            colors.begin()));
    }

    iterator end()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions.end(),
                                                            crd.end(),
                                                            normals.end(),
                                                            shapes.end(),
                                                            colors.end()));
    }

    void clear()
    {
        positions.clear();
        crd.clear();
        normals.clear();
        shapes.clear();
        colors.clear();
    }

    void setZeros(size_t length)
    {
        cudaMemset(thrust::raw_pointer_cast(&positions[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&colors[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&shapes[0]), 0.0f, length*sizeof(Cov3));
        cudaMemset(thrust::raw_pointer_cast(&normals[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&crd[0]), 0.0f, length*sizeof(float4));
    }

}; // struct Superpixels

struct Accumulators
{
    thrust::device_vector<float4> positions_sizes;
    thrust::device_vector<float2> centers;
    thrust::device_vector<float4> normals;
    thrust::device_vector<float4> colors_densities;
    thrust::device_vector<Cov3> shapes;

    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<float2>::iterator,
                                               thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<float4>::iterator,
                                               thrust::device_vector<Cov3>::iterator>> iterator;
    iterator begin()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions_sizes.begin(),
                                                            centers.begin(),
                                                            normals.begin(),
                                                            colors_densities.begin(),
                                                            shapes.begin()));
    }

    iterator end()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions_sizes.end(),
                                                            centers.end(),
                                                            normals.end(),
                                                            colors_densities.end(),
                                                            shapes.begin()));
    }

    void clear()
    {
        positions_sizes.clear();
        centers.clear();
        normals.clear();
        colors_densities.clear();
        shapes.clear();
    }

    void setZeros(size_t length)
    {
        cudaMemset(thrust::raw_pointer_cast(&positions_sizes[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&colors_densities[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&normals[0]), 0.0f, length*sizeof(float4));
        cudaMemset(thrust::raw_pointer_cast(&centers[0]), 0.0f, length*sizeof(float2));
        cudaMemset(thrust::raw_pointer_cast(&shapes[0]), 0.0f, length*sizeof(Cov3));
    }

}; // struct Accumulators

template<typename T>
struct Queue;

} // cuda_dasp
