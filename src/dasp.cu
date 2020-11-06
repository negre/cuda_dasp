#include <cuda_dasp/dasp.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_dasp/dasp_kernels.cuh>
#include <opencv2/highgui.hpp>
#include <cuda_dasp/cuda_error_check.h>
#include <random>
#include <opencv2/cudaarithm.hpp>
#include <chrono>


namespace cuda_dasp
{

DASP::DASP(const CamParam& cam_param,
           float radius,
           float compactness_p,
           float normal_weight,
           int nb_iter,
           float lambda_p,
           int nb_superpixels)
    : cam(cam_param),
      radiusMeter(radius),
      compactness(compactness_p),
      normalWeight(normal_weight),
      nbIter(nb_iter),
      lambda(lambda_p),
      nbSuperpixels(nb_superpixels)
{
    positionsMat.create(cam.height, cam.width, CV_32FC4);
    normalsMat.create(cam.height, cam.width, CV_32FC4);
    labMat.create(cam.height, cam.width, CV_32FC4);
    gradientMat.create(cam.height, cam.width, CV_32FC2);
    densityMat.create(cam.height, cam.width, CV_32FC1);
    weightsMat.create(cam.height, cam.width, CV_32FC1);
    labelsMat.create(cam.height, cam.width, CV_32SC1);
    rgbMat.create(cam.height, cam.width, CV_8UC4);
    depthMat.create(cam.height, cam.width, CV_32FC1);

    positionsTex = new Texture<float4>(positionsMat);
    normalsTex =  new Texture<float4>(normalsMat);
    labTex = new Texture<float4>(labMat);
    gradientTex = new Texture<float2>(gradientMat);
    densityTex = new Texture<float>(densityMat);
    weightsTex = new Texture<float>(weightsMat);
    labelsTex = new Texture<int>(labelsMat);
    //labelsTex = new Surface<int>(labelsMat);
    rgbTex = new Texture<uchar4>(rgbMat);
    depthTex = new Texture<float>(depthMat);
}

void DASP::seedsRandom()
{
    thrust::host_vector<int2> seeds_h;

    if(!seeds.empty())
        seeds.clear();

    cv::Mat density_mat_h;
    densityMat.download(density_mat_h);

    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // random sampling
    for(int c = 0; c < density_mat_h.cols; c++)
    {
        for(int r = 0; r < density_mat_h.rows; r++)
            if(distribution(generator) <= density_mat_h.at<float>(r, c))
            {
                int2 s;
                s.x = c;
                s.y = r;
                seeds_h.push_back(s);
            }
    }

    seeds = seeds_h;
}

void DASP::seedsFloydSteinberg()
{
    thrust::host_vector<int2> seeds_h;

    if(!seeds.empty())
        seeds.clear();

    cv::Mat dither;
    densityMat.download(dither);

    for(int c = 0; c < dither.cols - 1; c++)
    {
        dither.at<float>(1, c) += dither.at<float>(0, c);

        for(int r = 1; r < dither.rows - 1; r++)
        {
            float val = dither.at<float>(r, c);

            if(val >= 0.5f)
            {
                val -= 1.0f;
                int2 s;
                s.x = c;
                s.y = r;
                seeds_h.push_back(s);
            }

            dither.at<float>(r+1, c) += 7.0f / 16.0f * val;
            dither.at<float>(r-1, c+1) += 3.0f / 16.0f * val;
            dither.at<float>(r, c+1) += 5.0f / 16.0f * val;
            dither.at<float>(r+1, c+1) += 1.0f / 16.0f * val;
        }

        dither.at<float>(0, c+1) += dither.at<float>(dither.rows-1, c);
    }

    seeds = seeds_h;
}

void DASP::computeSuperpixels(const cv::Mat& rgb, const cv::Mat& depth)
{
    auto start = std::chrono::high_resolution_clock::now();

    cv::cuda::GpuMat rgb_d;
    rgb_d.upload(rgb);
    cv::cuda::cvtColor(rgb_d, rgbMat, cv::COLOR_RGB2RGBA);
    depthMat.upload(depth);
    cv::cuda::bilateralFilter(depthMat, depthMat, -1, 0.03/* depth_sigma*/, 4.5/* space_sigma*/);

    dim3 dim_block_im(16, 16);
    dim3 dim_grid_im((rgb.cols + dim_block_im.x - 1) / dim_block_im.x,
                     (rgb.rows + dim_block_im.y - 1) / dim_block_im.y);

    computeImages_kernel<<<dim_grid_im, dim_block_im>>>(rgbTex->getTextureObject(),
                                                        depthTex->getTextureObject(),
                                                        reinterpret_cast<float4*>(positionsMat.data),
                                                        reinterpret_cast<float4*>(normalsMat.data),
                                                        reinterpret_cast<float4*>(labMat.data),
                                                        reinterpret_cast<float2*>(gradientMat.data),
                                                        reinterpret_cast<float*>(densityMat.data),
                                                        radiusMeter,
                                                        cam.fx,
                                                        cam.fy,
                                                        cam.cx,
                                                        cam.cy,
                                                        cam.height,
                                                        cam.width,
                                                        positionsMat.step / sizeof(float4),
                                                        gradientMat.step / sizeof(float2),
                                                        densityMat.step / sizeof(float));
    cudaDeviceSynchronize();
    CudaCheckError();

    if(nbSuperpixels > 0)
    {
        float total_density = float(cv::cuda::sum(densityMat)[0]);
        float density_scale_factor = float(nbSuperpixels) / total_density;
        cv::cuda::multiply(densityMat, cv::Scalar(density_scale_factor), densityMat);
        //densityMat.convertTo(densityMat, CV_32FC1, density_scale_factor);
    }

    //seedsFloydSteinberg();
    seedsRandom();

    size_t nb_seeds = seeds.size();

    std::cout<<"nb seeds = "<<nb_seeds<<std::endl;

    if(!superpixels.positions.empty())
        superpixels.clear();

    superpixels.positions.resize(nb_seeds);
    superpixels.normals.resize(nb_seeds);
    superpixels.colors.resize(nb_seeds);
    superpixels.shapes.resize(nb_seeds);
    superpixels.crd.resize(nb_seeds);

    superpixels.setZeros(nb_seeds);

    dim3 dim_block_list(128);
    dim3 dim_grid_list((nb_seeds + dim_block_list.x - 1) / dim_block_list.x);

    initClusters_kernel<<<dim_grid_list, dim_block_list>>>(thrust::raw_pointer_cast(&superpixels.positions[0]),
                                                           thrust::raw_pointer_cast(&superpixels.colors[0]),
                                                           thrust::raw_pointer_cast(&superpixels.normals[0]),
                                                           thrust::raw_pointer_cast(&superpixels.crd[0]),
                                                           thrust::raw_pointer_cast(&seeds[0]),
                                                           positionsTex->getTextureObject(),
                                                           normalsTex->getTextureObject(),
                                                           labTex->getTextureObject(),
                                                           densityTex->getTextureObject(),
                                                           nb_seeds);
    cudaDeviceSynchronize();
    CudaCheckError();

    for(int i = 0; i < nbIter; i++)
    {
        if(!accumulators.positions_sizes.empty())
            accumulators.clear();

        accumulators.positions_sizes.resize(nb_seeds);
        accumulators.normals.resize(nb_seeds);
        accumulators.colors_densities.resize(nb_seeds);
        accumulators.shapes.resize(nb_seeds);
        accumulators.centers.resize(nb_seeds);

        accumulators.setZeros(nb_seeds);

        labelsMat.setTo(cv::Scalar(-1));

        assignClustersV1_kernel<<<dim_grid_im, dim_block_im>>>(thrust::raw_pointer_cast(&superpixels.positions[0]),
                                                               thrust::raw_pointer_cast(&superpixels.colors[0]),
                                                               thrust::raw_pointer_cast(&superpixels.normals[0]),
                                                               thrust::raw_pointer_cast(&superpixels.crd[0]),
                                                               positionsTex->getTextureObject(),
                                                               normalsTex->getTextureObject(),
                                                               labTex->getTextureObject(),
                                                               densityTex->getTextureObject(),
                                                               thrust::raw_pointer_cast(&accumulators.positions_sizes[0]),
                                                               thrust::raw_pointer_cast(&accumulators.centers[0]),
                                                               thrust::raw_pointer_cast(&accumulators.normals[0]),
                                                               thrust::raw_pointer_cast(&accumulators.colors_densities[0]),
                                                               thrust::raw_pointer_cast(&accumulators.shapes[0]),
                                                               reinterpret_cast<int*>(labelsMat.data),
                                                               compactness,
                                                               normalWeight,
                                                               lambda,
                                                               radiusMeter,
                                                               nb_seeds,
                                                               labelsMat.step / sizeof(int),
                                                               cam.height,
                                                               cam.width);
//        assignClustersV3_kernel<<<dim_grid_im, dim_block_im>>>(thrust::raw_pointer_cast(&superpixels.positions[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.colors[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.normals[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.crd[0]),
//                                                               positionsTex->getTextureObject(),
//                                                               normalsTex->getTextureObject(),
//                                                               labTex->getTextureObject(),
//                                                               densityTex->getTextureObject(),
//                                                               reinterpret_cast<int*>(labelsMat.data),
//                                                               compactness,
//                                                               normalWeight,
//                                                               lambda,
//                                                               radiusMeter,
//                                                               nb_seeds,
//                                                               labelsMat.step / sizeof(int),
//                                                               cam.height,
//                                                               cam.width);
        cudaDeviceSynchronize();
        CudaCheckError();

        updateClustersV1_kernel<<<dim_grid_list, dim_block_list>>>(thrust::raw_pointer_cast(&superpixels.positions[0]),
                                                                   thrust::raw_pointer_cast(&superpixels.colors[0]),
                                                                   thrust::raw_pointer_cast(&superpixels.normals[0]),
                                                                   thrust::raw_pointer_cast(&superpixels.crd[0]),
                                                                   thrust::raw_pointer_cast(&superpixels.shapes[0]),
                                                                   //thrust::raw_pointer_cast(&seeds[0]),
                                                                   thrust::raw_pointer_cast(&accumulators.positions_sizes[0]),
                                                                   thrust::raw_pointer_cast(&accumulators.centers[0]),
                                                                   thrust::raw_pointer_cast(&accumulators.normals[0]),
                                                                   thrust::raw_pointer_cast(&accumulators.colors_densities[0]),
                                                                   thrust::raw_pointer_cast(&accumulators.shapes[0]),
                                                                   nb_seeds);
//        updateClustersV2_kernel<<<dim_grid_im, dim_block_im>>>(thrust::raw_pointer_cast(&superpixels.positions[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.colors[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.normals[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.crd[0]),
//                                                               thrust::raw_pointer_cast(&superpixels.shapes[0]),
//                                                               thrust::raw_pointer_cast(&accumulators.positions_sizes[0]),
//                                                               thrust::raw_pointer_cast(&accumulators.centers[0]),
//                                                               thrust::raw_pointer_cast(&accumulators.normals[0]),
//                                                               thrust::raw_pointer_cast(&accumulators.colors_densities[0]),
//                                                               thrust::raw_pointer_cast(&accumulators.shapes[0]),
//                                                               positionsTex->getTextureObject(),
//                                                               normalsTex->getTextureObject(),
//                                                               labTex->getTextureObject(),
//                                                               densityTex->getTextureObject(),
//                                                               labelsTex->getTextureObject(),
//                                                               nb_seeds,
//                                                               cam.height,
//                                                               cam.width);
        cudaDeviceSynchronize();
        CudaCheckError();
    }

    //if(doEnforceConnectivity)
    //    enforceConnectivity();

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout<<"Segmentation time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()<<" ms"<<std::endl;

    cv::Mat vis_boundaries;
    cv::cuda::GpuMat vis_boundaries_d = rgbMat.clone();
    renderBoundaryImage_kernel<<<dim_grid_im, dim_block_im>>>(reinterpret_cast<uchar4*>(vis_boundaries_d.data),
                                                              labelsTex->getTextureObject(),
                                                              vis_boundaries_d.step / sizeof(uchar4),
                                                              cam.height,
                                                              cam.width);
    cudaDeviceSynchronize();
    CudaCheckError();

    cv::cuda::cvtColor(vis_boundaries_d, vis_boundaries_d, cv::COLOR_RGBA2BGR);
    vis_boundaries_d.download(vis_boundaries);

    cv::cuda::GpuMat vis_colors_d;
    vis_colors_d.create(cam.height, cam.width, CV_8UC4);
    renderSuperpixelsColorImage_kernel<<<dim_grid_im, dim_block_im>>>(reinterpret_cast<uchar4*>(vis_colors_d.data),
                                                                      labelsTex->getTextureObject(),
                                                                      thrust::raw_pointer_cast(&superpixels.colors[0]),
                                                                      vis_colors_d.step / sizeof(uchar4),
                                                                      cam.height,
                                                                      cam.width);
    cudaDeviceSynchronize();
    CudaCheckError();

    cv::Mat vis_colors;
    cv::cuda::cvtColor(vis_colors_d, vis_colors_d, cv::COLOR_RGBA2BGR);
    vis_colors_d.download(vis_colors);


    cv::Mat vis_density, vis_normals;
    densityMat.download(vis_density);
    normalsMat.download(vis_normals);
    cv::cvtColor(vis_normals, vis_normals, cv::COLOR_BGRA2BGR);
    cv::imshow("density", 50.0f*vis_density);
    cv::imshow("normals", vis_normals);
    cv::imshow("superpixels boundaries", vis_boundaries);
    cv::imshow("superpixels colors", vis_colors);

    cv::waitKey(1);
}

} // cuda_dasp
