#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>


namespace cuda_dasp
{

template <typename T>
class Surface
{

private:
    cudaTextureObject_t surfaceObject;

public:
    Surface(const cv::cuda::GpuMat& img);

    ~Surface();

    inline cudaSurfaceObject_t& getSurfaceObject(){ return surfaceObject; }

}; // class Surface

template<typename T>
Surface<T>::Surface(const cv::cuda::GpuMat& img)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.data;
    resDesc.res.pitch2D.width = img.cols;
    resDesc.res.pitch2D.height = img.rows;
    resDesc.res.pitch2D.pitchInBytes = img.step;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();

    cudaCreateSurfaceObject(&surfaceObject, &resDesc);
}

template<typename T>
Surface<T>::~Surface()
{
    cudaDestroySurfaceObject(surfaceObject);
}

} // namespace cuda_dasp
