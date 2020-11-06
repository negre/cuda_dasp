#pragma once

#include <cuda_dasp/dasp_types.hpp>
#include <cuda_dasp/cam_param.hpp>
#include <cuda_dasp/texture.hpp>
//#include <cuda_dasp/surface.hpp>
#include <opencv2/core/cuda.hpp>


namespace cuda_dasp
{

class DASP
{

public:
    DASP(const CamParam& cam_param,
         float radius,
         float compactness_p,
         float normal_weight,
         int nb_iter,
         float lambda_p,
         int nb_superpixels = 0);
    void computeSuperpixels(const cv::Mat& rgb, const cv::Mat& depth);
    void seedsFloydSteinberg();
    void seedsRandom();

private:
    CamParam cam;
    float radiusMeter;
    cv::cuda::GpuMat rgbMat, depthMat;
    cv::cuda::GpuMat positionsMat, normalsMat, gradientMat, densityMat, labMat, labelsMat, weightsMat;
    cv::Ptr<Texture<float4>> positionsTex, normalsTex, labTex;
    cv::Ptr<Texture<uchar4>> rgbTex;
    cv::Ptr<Texture<float2>> gradientTex;
    cv::Ptr<Texture<float>> densityTex, weightsTex, depthTex;
    cv::Ptr<Texture<int>> labelsTex; // TODO use surface memory instead?
    //cv::Ptr<Surface<int>> labelsTex;
    thrust::device_vector<int2> seeds;
    Superpixels superpixels;
    Accumulators accumulators;
    float compactness, normalWeight;
    int nbIter;
    float lambda;
    int nbSuperpixels;
};

} // cuda_dasp
