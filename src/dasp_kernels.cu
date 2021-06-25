#include <cuda_dasp/dasp_kernels.cuh>
//#include <cuda_dasp/vector_math.cuh>
#include <cuda_dasp/matrix_math.cuh>
#include <cuda_dasp/cuda_utils_dev.cuh>
#include <stdio.h>
#include <curand_kernel.h>

#define PI 3.14159265

namespace cuda_dasp
{
__device__ void syncAllThreads(unsigned int* syncCounter)

{

	__syncthreads();

	if (threadIdx.x == 0) {

		atomicInc(syncCounter, gridDim.x-1);

		volatile unsigned int* counter = syncCounter;

		do

		{

		} while (*counter > 0);

	}

	__syncthreads();

}

__device__ void eigenDecomposition(const Cov3& A, Mat33& eigenVecs, float3& eigenVals, int n)
{
    Cov3 Ai = A / trace(A);
    Cov3 Bi = make_cov3(1.f - Ai.xx, -Ai.xy, -Ai.xz, 1.f - Ai.yy, -Ai.yz, 1.f - Ai.zz);

    for(int i = 0; i < n; ++i)
    {
        Ai = square(Ai);
        Ai /= trace(Ai);

        Bi = square(Bi);
        Bi /= trace(Bi);
    }

    float vmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(Ai.xx, Ai.xy), Ai.xz), Ai.yy), Ai.yz), Ai.zz);

    if(Ai.xx==vmax || Ai.xy==vmax || Ai.xz==vmax)
    {
        eigenVecs.rows[0]=normalize(make_float3(Ai.xx, Ai.xy, Ai.xz));
    }
    else if(Ai.yy==vmax || Ai.yz==vmax)
    {
        eigenVecs.rows[0]=normalize(make_float3(Ai.xy, Ai.yy, Ai.yz));
    }
    else
    {
        eigenVecs.rows[0]=normalize(make_float3(Ai.xz, Ai.yz, Ai.zz));
    }

    vmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(Bi.xx, Bi.xy), Bi.xz), Bi.yy), Bi.yz), Bi.zz);

    if(Bi.xx==vmax || Bi.xy==vmax || Bi.xz==vmax)
    {
        eigenVecs.rows[2]=normalize(make_float3(Bi.xx, Bi.xy, Bi.xz));
    }
    else if(Bi.yy==vmax || Bi.yz==vmax)
    {
        eigenVecs.rows[2]=normalize(make_float3(Bi.xy, Bi.yy, Bi.yz));
    }
    else
    {
        eigenVecs.rows[2]=normalize(make_float3(Bi.xz, Bi.yz, Bi.zz));
    }

    eigenVecs.rows[1] = cross(eigenVecs.rows[2], eigenVecs.rows[0]);

    float emax = fmaxf(fmaxf(eigenVecs.rows[0].x, eigenVecs.rows[0].y), eigenVecs.rows[0].z);

    if(eigenVecs.rows[0].x==emax)
        eigenVals.x = (A.xx*eigenVecs.rows[0].x+A.xy*eigenVecs.rows[0].y+A.xz*eigenVecs.rows[0].z) / eigenVecs.rows[0].x;
    else if(eigenVecs.rows[0].y==emax)
        eigenVals.x = (A.xy*eigenVecs.rows[0].x+A.yy*eigenVecs.rows[0].y+A.yz*eigenVecs.rows[0].z) / eigenVecs.rows[0].y;
    else
        eigenVals.x = (A.xz*eigenVecs.rows[0].x+A.yz*eigenVecs.rows[0].y+A.zz*eigenVecs.rows[0].z) / eigenVecs.rows[0].z;


    emax = fmaxf(fmaxf(eigenVecs.rows[1].x, eigenVecs.rows[1].y), eigenVecs.rows[1].z);

    if(eigenVecs.rows[1].x==emax)
        eigenVals.y = (A.xx*eigenVecs.rows[1].x+A.xy*eigenVecs.rows[1].y+A.xz*eigenVecs.rows[1].z) / eigenVecs.rows[1].x;
    else if(eigenVecs.rows[1].y==emax)
        eigenVals.y = (A.xy*eigenVecs.rows[1].x+A.yy*eigenVecs.rows[1].y+A.yz*eigenVecs.rows[1].z) / eigenVecs.rows[1].y;
    else
        eigenVals.y = (A.xz*eigenVecs.rows[1].x+A.yz*eigenVecs.rows[1].y+A.zz*eigenVecs.rows[1].z) / eigenVecs.rows[1].z;

    emax = fmaxf(fmaxf(eigenVecs.rows[2].x, eigenVecs.rows[2].y), eigenVecs.rows[2].z);

    if(eigenVecs.rows[2].x==emax)
        eigenVals.z = (A.xx*eigenVecs.rows[2].x+A.xy*eigenVecs.rows[2].y+A.xz*eigenVecs.rows[2].z) / eigenVecs.rows[2].x;
    else if(eigenVecs.rows[2].y==emax)
        eigenVals.z = (A.xy*eigenVecs.rows[2].x+A.yy*eigenVecs.rows[2].y+A.yz*eigenVecs.rows[2].z) / eigenVecs.rows[2].y;
    else
        eigenVals.z = (A.xz*eigenVecs.rows[2].x+A.yz*eigenVecs.rows[2].y+A.zz*eigenVecs.rows[2].z) / eigenVecs.rows[2].z;
}

__device__ __inline__ float computeDistance(const float3 pos1, const float3 lab1, const float3 norm1, const float2 cent1,
					    const float3 pos2, const float3 lab2, const float3 norm2, const float2 cent2,
					    const float radius,
					    const float compactness,
					    const float normal_weight,
					    const float win_size)
{
//     if(pos1.z==0.f || pos2.z==0.f)
//     {
// 	return (1.0f - compactness) * ((1.0f - normal_weight) * (0.008f * squared_length(lab1 - lab2)));
//     }else
//     {
        return compactness * (squared_length(pos1 - pos2) / (radius*radius))
	    + (1.0f - compactness) * ((1.0f - normal_weight) * (0.008f * squared_length(lab1 - lab2))
				      + normal_weight * (1.0f - dot(norm1, norm2)));
//     }
    
//    return compactness * (0.5f * squared_length(cent1 - cent2) + 0.5f * (squared_length(pos1 - pos2) / (radius*radius)))
//           + (1.0f - compactness) * ((1.0f - normal_weight) * (0.001f * squared_length(lab1 - lab2))
//           + normal_weight * (1.0f - dot(norm1, norm2)));
    //return length(lab1 - lab2) + 10.0f * length(cent1 - cent2) / win_size;
}

//__device__ float computeDistance(const float3 pos1, const float3 lab1, const float3 norm1,
//                                 const float3 pos2, const float3 lab2, const float3 norm2,
//                                 const float radius,
//                                 const float compactness,
//                                 const float normal_weight)
//{
//    return compactness * (squared_length(pos1 - pos2) / (radius*radius))
//           + (1.0f - compactness) * ((1.0f - normal_weight) * (0.001f * squared_length(lab1 - lab2))
//           + normal_weight * (1.0f - dot(norm1, norm2)));
//}

__device__ float localFiniteDifferences(float d0, float d1, float d2, float d3, float d4)
{
    if((d0 < 0.2f || d0 > 6.0f) && (d4 < 0.2f || d4 > 6.0f) && (d1 >= 0.2f && d1 <= 6.0f) && (d3 >= 0.2f && d3 <= 6.0f))
        return d3 - d1;

    bool left_invalid = (d0 < 0.2f || d0 > 6.0f || d1 < 0.2f || d1 > 6.0f);
    bool right_invalid = (d3 < 0.2f || d3 > 6.0f || d4 < 0.2f || d4 > 6.0f);

    if(left_invalid && right_invalid)
        return 0.0f;
    else if(left_invalid)
        return d4 - d2;
    else if(right_invalid)
        return d2 - d0;
    else
    {
        float a = fabsf(d2 + d0 - 2.0f * d1);
        float b = fabsf(d4 + d2 - 2.0f * d3);
        float p, q;

        if(a + b == 0.0f)
        {
            p = 0.5f;
            q = 0.5f;
        }
        else
        {
            p = a / (a + b);
            q = b / (a + b);
        }

        return q * (d2 - d0) + p * (d4 - d2);
    }
}

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
                                     const int step_ch1)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(c >= cols || r >= rows)
        return;

    int tid_ch4 = r * step_ch4 + c;
    int tid_ch2 = r * step_ch2 + c;
    int tid_ch1 = r * step_ch1 + c;

    uchar4 rgb = tex2D<uchar4>(tex_rgb, c, r);
    float4 lab = rgbToLab(make_float4(float(rgb.x), float(rgb.y), float(rgb.z), 0.0f));
    im_lab[tid_ch4] = lab;
    //im_lab[tid_ch4] = rgb;
    //printf("rgb: %f %f %f   lab: %f %f %f\n", rgb.x, rgb.y, rgb.z, lab.x, lab.y, lab.z);

    im_gradient[tid_ch2] = make_float2(0.0f, 0.0f);
    im_positions[tid_ch4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    im_normals[tid_ch4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    im_density[tid_ch1] = 0.0f;

    float z = tex2D<float>(tex_depth, c, r);

    if(z >= 0.2f && z <= 6.0f)
    {
        float radius_pixel = fx * radius_meter / z;

        //int dp = 2;
        //int dp = int(fmaxf(0.5f * radius_pixel + 0.5f, 4.0f));
        int dp = int(fmaxf(0.1f * radius_pixel + 0.5f, 4.0f));
        if(dp % 2 == 1)
            dp++;
        int dp2 = dp / 2;

        float3 position = make_float3(z * (float(c) - cx) / fx,
                                      z * (float(r) - cy) / fy,
                                      z);
        im_positions[tid_ch4] = make_float4(position);

        float z_rf = tex2D<float>(tex_depth, c, r + dp);
        float z_rb = tex2D<float>(tex_depth, c, r - dp);
        float z_cf = tex2D<float>(tex_depth, c + dp, r);
        float z_cb = tex2D<float>(tex_depth, c - dp, r);
        float z_rf2 = tex2D<float>(tex_depth, c, r + dp2);
        float z_rb2 = tex2D<float>(tex_depth, c, r - dp2);
        float z_cf2 = tex2D<float>(tex_depth, c + dp2, r);
        float z_cb2 = tex2D<float>(tex_depth, c - dp2, r);

        float dz_r = localFiniteDifferences(z_rb, z_rb2, z, z_rf2, z_rf);
        float dz_c = localFiniteDifferences(z_cb, z_cb2, z, z_cf2, z_cf);

        float2 grad = make_float2(dz_c, dz_r);
        grad *= fx / (float(dp) * z);
        im_gradient[tid_ch2] = grad;

        float3 normal = make_float3(grad.x, grad.y, -1.0f);
        normal /= length(normal);
        float q = dot(normal, -1.0f * position);
        if(q > 0.0f)
            normal *= -1.0f;

//        float3 normal = make_float3(0.0f, 0.0f, 0.0f);
//        if(z_rf >= 0.2f && z_rb >= 0.2f && z_cf >= 0.2f && z_cb >= 0.2f)
//        {
//            float3 p_cf = make_float3(z_cf * (float(c + dp) - cx) / fx,
//                                      z_cf * (float(r) - cy) / fy,
//                                      z_cf);
//            float3 p_cb = make_float3(z_cb * (float(c - dp) - cx) / fx,
//                                      z_cb * (float(r) - cy) / fy,
//                                      z_cb);
//            float3 p_rf = make_float3(z_rf * (float(c) - cx) / fx,
//                                      z_rf * (float(r + dp) - cy) / fy,
//                                      z_rf);
//            float3 p_rb = make_float3(z_rb * (float(c) - cx) / fx,
//                                      z_rb * (float(r - dp) - cy) / fy,
//                                      z_rb);
//            normal = normalize(cross((p_cf - p_cb), (p_rf - p_rb)));

//            //float2 grad = make_float2(z_cf - z_cb, z_rf - z_rb);
//            //grad *= fx / (float(dp) * z);
//            //im_gradient[tid_ch2] = grad;

//            im_normals[tid_ch4] = make_float4(normal);

//            //im_density[tid_ch1] = sqrtf(grad.x * grad.x + grad.y * grad.y + 1.0f) / (PI * radius_pixel * radius_pixel);
//        }

        im_normals[tid_ch4] = make_float4(normal);

        im_density[tid_ch1] = sqrtf(grad.x * grad.x + grad.y * grad.y + 1.0f) / (PI * radius_pixel * radius_pixel);
    }
}

__global__ void initClusters_kernel(float4* __restrict__ positions,
                                    float4* __restrict__ colors,
                                    float4* __restrict__ normals,
                                    float4* __restrict__ crd,
                                    const int2* __restrict__ seeds,
                                    cudaTextureObject_t tex_positions,
                                    cudaTextureObject_t tex_normals,
                                    cudaTextureObject_t tex_lab,
                                    cudaTextureObject_t tex_density,
                                    const int nb_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= nb_clusters)
        return;

    //printf("x %d  y %d", seeds[idx].x, seeds[idx].y);
    int2 p_seed = seeds[idx];

    positions[idx] = tex2D<float4>(tex_positions, p_seed.x, p_seed.y);
    normals[idx] = tex2D<float4>(tex_normals, p_seed.x, p_seed.y);
    colors[idx] = tex2D<float4>(tex_lab, p_seed.x, p_seed.y);

    float d = tex2D<float>(tex_density, p_seed.x, p_seed.y);
    float r = sqrtf(1.0f / (PI * d));
    crd[idx] = make_float4(float(p_seed.x), float(p_seed.y), r, d);
}

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
                                        const int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(c >= cols || r >= rows)
        return;

    float3 position = make_float3(tex2D<float4>(tex_positions,c, r));

    if(position.z == 0.0f)
        return;

    float3 color = make_float3(tex2D<float4>(tex_lab,c, r));
    float3 normal = make_float3(tex2D<float4>(tex_normals, c, r));
    float density = tex2D<float>(tex_density, c, r);

    int closest_cluster_id = -1;
    float min_dist = 1000000.0f;

    for(int i = 0; i < nb_clusters; i++)
    {
        float3 cluster_position = make_float3(positions[i]);
        float3 cluster_color = make_float3(colors[i]);
        //printf("%f %f %f", cluster_color.x, cluster_color.y, cluster_color.z);

        float3 cluster_normal = make_float3(normals[i]);
        float4 cluster_crd = crd[i];
        float2 cluster_center = make_float2(cluster_crd.x, cluster_crd.y);
        float cluster_radius = cluster_crd.z;
        //float cluster_density = crd.w;

        float win_size = lambda * cluster_radius;

        float2 p = make_float2(float(c), float(r));

        if(length(cluster_center - p) <= win_size)
        {
            float dist = computeDistance(cluster_position, cluster_color, cluster_normal, cluster_center,
                                         position, color, normal, p,
                                         radius_meter,
                                         compactness,
                                         normal_weight,
                                         win_size);
            if(dist < min_dist)
            {
                closest_cluster_id = i;
                min_dist = dist;
            }
        }
    }

    if(closest_cluster_id >= 0)
    {
        im_labels[r * step + c] = closest_cluster_id;

        atomicAdd(&acc_positions_sizes[closest_cluster_id].x, position.x);
        atomicAdd(&acc_positions_sizes[closest_cluster_id].y, position.y);
        atomicAdd(&acc_positions_sizes[closest_cluster_id].z, position.z);
        atomicAdd(&acc_positions_sizes[closest_cluster_id].w, 1.0f);
        atomicAdd(&acc_colors_densities[closest_cluster_id].x, color.x);
        atomicAdd(&acc_colors_densities[closest_cluster_id].y, color.y);
        atomicAdd(&acc_colors_densities[closest_cluster_id].z, color.z);
        atomicAdd(&acc_colors_densities[closest_cluster_id].w, density);
        atomicAdd(&acc_centers[closest_cluster_id].x, float(c));
        atomicAdd(&acc_centers[closest_cluster_id].y, float(r));
        atomicAdd(&acc_normals[closest_cluster_id].x, normal.x);
        atomicAdd(&acc_normals[closest_cluster_id].y, normal.y);
        atomicAdd(&acc_normals[closest_cluster_id].z, normal.z);

        Cov3 cov = outer_product(position);

        atomicAdd(&acc_shapes[closest_cluster_id].xx, cov.xx);
        atomicAdd(&acc_shapes[closest_cluster_id].xy, cov.xy);
        atomicAdd(&acc_shapes[closest_cluster_id].xz, cov.xz);
        atomicAdd(&acc_shapes[closest_cluster_id].yy, cov.yy);
        atomicAdd(&acc_shapes[closest_cluster_id].yz, cov.yz);
        atomicAdd(&acc_shapes[closest_cluster_id].zz, cov.zz);
    }
}

// TODO ensure no bank conflict
#define TPB 16
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
                                        const int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    //if(c >= cols || r >= rows)
    //    return;

    //float3 position = make_float3(tex2D<float4>(tex_positions,c, r));

    //if(position.z == 0.0f)
    //    return;

    //float3 color = make_float3(tex2D<float4>(tex_lab,c, r));
    //float3 normal = make_float3(tex2D<float4>(tex_normals, c, r));
    //float density = tex2D<float>(tex_density, c, r);

    float3 position, color, normal;
    float density;

    if(c < cols && r < rows)
    {
        position = make_float3(tex2D<float4>(tex_positions,c, r));
        color = make_float3(tex2D<float4>(tex_lab,c, r));
        normal = make_float3(tex2D<float4>(tex_normals, c, r));
        density = tex2D<float>(tex_density, c, r);
    }

    int closest_cluster_label = -1;
    float min_dist = 1000000.0f;

    int tid_c = threadIdx.x;
    int tid_r = threadIdx.y;

    __shared__ float4 sh_pos[TPB][TPB];
    __shared__ float4 sh_col[TPB][TPB];
    __shared__ float4 sh_norm[TPB][TPB];
    __shared__ float4 sh_crd[TPB][TPB];
    __shared__ int sh_id[TPB][TPB];

    for(int i = 0; i < (((nb_clusters - 1) / (TPB * TPB)) + 1); i++)
    {
        int cluster_id = tid_r * TPB + tid_c + i * TPB * TPB;

        // collaboratively loading seeds info into shared memory
        if(cluster_id < nb_clusters && c < cols && r < rows)
        {
            sh_pos[tid_r][tid_c] = positions[cluster_id];
            sh_col[tid_r][tid_c] = colors[cluster_id];
            sh_norm[tid_r][tid_c] = normals[cluster_id];
            sh_crd[tid_r][tid_c] = crd[cluster_id];
            sh_id[tid_r][tid_c] = cluster_id;
        }
        else
        {
            float4 zeros = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sh_pos[tid_r][tid_c] = zeros;
            sh_col[tid_r][tid_c] = zeros;
            sh_norm[tid_r][tid_c] = zeros;
            sh_crd[tid_r][tid_c] = zeros;
            sh_id[tid_r][tid_c] = -1;
        }

        __syncthreads();

        if(c < cols && r < rows && position.z > 0.0f)
        {
            for(int j = 0; j < TPB; j++)
            {
                for(int k = 0; k < TPB; k++)
                {
                    int label = sh_id[j][k];

                    if(label >= 0/* && c < cols && r < rows && position.z > 0.0f*/)
                    {
                        float3 cluster_position = make_float3(sh_pos[j][k]);
                        float3 cluster_color = make_float3(sh_col[j][k]);
                        float3 cluster_normal = make_float3(sh_norm[j][k]);
                        float4 cluster_crd = sh_crd[j][k];
                        float2 cluster_center = make_float2(cluster_crd.x, cluster_crd.y);
                        float cluster_radius = cluster_crd.z;

                        float win_size = lambda * cluster_radius;

                        float2 p = make_float2(float(c), float(r));

                        if(length(cluster_center - p) <= win_size)
                        {
                            float dist = computeDistance(cluster_position, cluster_color, cluster_normal, cluster_center,
                                                         position, color, normal, p,
                                                         radius_meter,
                                                         compactness,
                                                         normal_weight,
                                                         win_size);
                            if(dist < min_dist)
                            {
                                closest_cluster_label = label;
                                min_dist = dist;
                            }
                        }

                    }
                }
            }
        }

        // synchronize to make sure the preceding computation is done before loading new cluster infos
        //__syncthreads();
    }

    __syncthreads();

//    im_labels[r * step + c] = closest_cluster_label;

    if(closest_cluster_label >= 0)
    {
        im_labels[r * step + c] = closest_cluster_label;

        atomicAdd(&acc_positions_sizes[closest_cluster_label].x, position.x);
        atomicAdd(&acc_positions_sizes[closest_cluster_label].y, position.y);
        atomicAdd(&acc_positions_sizes[closest_cluster_label].z, position.z);
        atomicAdd(&acc_positions_sizes[closest_cluster_label].w, 1.0f);
        atomicAdd(&acc_colors_densities[closest_cluster_label].x, color.x);
        atomicAdd(&acc_colors_densities[closest_cluster_label].y, color.y);
        atomicAdd(&acc_colors_densities[closest_cluster_label].z, color.z);
        atomicAdd(&acc_colors_densities[closest_cluster_label].w, density);
        atomicAdd(&acc_centers[closest_cluster_label].x, float(c));
        atomicAdd(&acc_centers[closest_cluster_label].y, float(r));
        atomicAdd(&acc_normals[closest_cluster_label].x, normal.x);
        atomicAdd(&acc_normals[closest_cluster_label].y, normal.y);
        atomicAdd(&acc_normals[closest_cluster_label].z, normal.z);

        Cov3 cov = outer_product(position);

        atomicAdd(&acc_shapes[closest_cluster_label].xx, cov.xx);
        atomicAdd(&acc_shapes[closest_cluster_label].xy, cov.xy);
        atomicAdd(&acc_shapes[closest_cluster_label].xz, cov.xz);
        atomicAdd(&acc_shapes[closest_cluster_label].yy, cov.yy);
        atomicAdd(&acc_shapes[closest_cluster_label].yz, cov.yz);
        atomicAdd(&acc_shapes[closest_cluster_label].zz, cov.zz);
    }
}

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
                                        const int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(c >= cols || r >= rows)
        return;

    float3 position = make_float3(tex2D<float4>(tex_positions,c, r));

    if(position.z == 0.0f)
        return;

    float3 color = make_float3(tex2D<float4>(tex_lab,c, r));
    float3 normal = make_float3(tex2D<float4>(tex_normals, c, r));
    float density = tex2D<float>(tex_density, c, r);

    int closest_cluster_id = -1;
    float min_dist = 1000000.0f;

    for(int i = 0; i < nb_clusters; i++)
    {
        float3 cluster_position = make_float3(positions[i]);
        float3 cluster_color = make_float3(colors[i]);
        //printf("%f %f %f", cluster_color.x, cluster_color.y, cluster_color.z);

        float3 cluster_normal = make_float3(normals[i]);
        float4 cluster_crd = crd[i];
        float2 cluster_center = make_float2(cluster_crd.x, cluster_crd.y);
        float cluster_radius = cluster_crd.z;
        //float cluster_density = crd.w;

        float win_size = lambda * cluster_radius;

        float2 p = make_float2(float(c), float(r));

        if(length(cluster_center - p) <= win_size)
        {
            float dist = computeDistance(cluster_position, cluster_color, cluster_normal, cluster_center,
                                         position, color, normal, p,
                                         radius_meter,
                                         compactness,
                                         normal_weight,
                                         win_size);
            if(dist < min_dist)
            {
                closest_cluster_id = i;
                min_dist = dist;
            }
        }
    }

    im_labels[r * step + c] = closest_cluster_id; // TODO surface memory for im_labels?
}

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
                                        const int nb_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= nb_clusters)
        return;

    float4 acc_ps = acc_positions_sizes[idx];
    float size = acc_ps.w;
    float4 mean_cd = acc_colors_densities[idx] / size;

    float3 new_color = make_float3(mean_cd.x, mean_cd.y, mean_cd.z);
    //printf("%f %f %f", new_color.x, new_color.y, new_color.z);
    float new_density = mean_cd.w;
    float3 new_position = make_float3(acc_ps.x, acc_ps.y, acc_ps.z) / size;
    float2 new_center = acc_centers[idx] / size;

    Cov3 new_shape = acc_shapes[idx] / size - outer_product(new_position);

    float3 eigen_vals;
    Mat33 eigen_vecs;
    eigenDecomposition(new_shape, eigen_vecs, eigen_vals, 10);

    float3 new_normal;

    if(eigen_vals.x < 1e-6f || eigen_vals.y < 1e-6f || eigen_vals.x / eigen_vals.y > 100.f)
        new_normal = eigen_vecs.rows[2];
    else new_normal = make_float3(acc_normals[idx]) / size;

    float new_radius = sqrtf(1.0f / (PI * new_density));

    positions[idx] = make_float4(new_position);
    colors[idx] = make_float4(new_color);
    normals[idx] = make_float4(new_normal);
    crd[idx] = make_float4(new_center.x, new_center.y, new_radius, new_density);
    shapes[idx] = new_shape;

    //seeds[idx] = make_int2(int(new_center.x), int(new_center.y));

    // TODO: reset acc here
}

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
                                        const int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int tid_c = threadIdx.x;
    int tid_r = threadIdx.y;

    //if(c >= cols || r >= rows)
    //    return;

    __shared__ float4 sh_pos[TPB][TPB];
    __shared__ float4 sh_col[TPB][TPB];
    __shared__ float4 sh_norm[TPB][TPB];
    __shared__ float2 sh_cent[TPB][TPB];
    __shared__ float sh_dens[TPB][TPB];
    __shared__ int sh_label[TPB][TPB];

    int label = -1;

    if(c < cols && r < rows)
        label = tex2D<int>(tex_labels, c, r);

    if(label >= 0)
    {
        sh_label[tid_r][tid_c] = label;
        sh_dens[tid_r][tid_c] = tex2D<float>(tex_density, c, r);
        sh_cent[tid_r][tid_c] = make_float2(float(c), float(r));
        sh_col[tid_r][tid_c] = tex2D<float4>(tex_lab, c, r);
        sh_pos[tid_r][tid_c] = tex2D<float4>(tex_positions, c, r);
        sh_norm[tid_r][tid_c] = tex2D<float4>(tex_normals, c, r);
    }
    else
    {
        sh_label[tid_r][tid_c] = -1;
        sh_dens[tid_r][tid_c] = 0.0f;
        sh_cent[tid_r][tid_c] = make_float2(0.0f, 0.0f);
        sh_col[tid_r][tid_c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh_pos[tid_r][tid_c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh_norm[tid_r][tid_c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    if(tid_c == 0 && tid_r == 0)
    {
        // impossible car n_clusters connu seulement a l'execution
//        float4 pos_size_sum[nb_clusters];
//        float4 norm_sum[nb_clusters];
//        float2 cent_sum[nb_clusters];
//        float4 col_dens_sum[nb_clusters];
//        Cov3 cov_sum[nb_clusters];
        float4 pos_size_sum[1024]; // TODO dynamic shared memory instead
        float4 norm_sum[1024];
        float2 cent_sum[1024];
        float4 col_dens_sum[1024];
        Cov3 cov_sum[1024];

        for(int k = 0 ; k < nb_clusters; k++)
        {
            pos_size_sum[k] = make_float4(0.0f);
            norm_sum[k] = make_float4(0.0f);
            cent_sum[k] = make_float2(0.0f);
            col_dens_sum[k] = make_float4(0.0);
            cov_sum[k] = make_cov3(0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f,
                                   0.0f);
        }

        for(int i = 0; i < TPB; i++)
        {
            for(int j = 0; j < TPB; j++)
            {
                int l = sh_label[i][j];

                if(l >= 0)
                {
                    float3 pos = make_float3(sh_pos[i][j]);

                    pos_size_sum[l] += make_float4(pos.x, pos.y, pos.z, 1.0f);
                    norm_sum[l] += sh_norm[i][j];
                    cent_sum[l] += sh_cent[i][j];
                    col_dens_sum[l] += make_float4(sh_col[i][j].x, sh_col[i][j].y, sh_col[i][j].z, sh_dens[i][j]);
                    cov_sum[l] += outer_product(pos);
                }
            }
        }

        for(int k = 0 ; k < nb_clusters; k++)
        {
            // atomic operations here
            atomicAdd(&acc_positions_sizes[k].x, pos_size_sum[k].x);
            atomicAdd(&acc_positions_sizes[k].y, pos_size_sum[k].y);
            atomicAdd(&acc_positions_sizes[k].z, pos_size_sum[k].z);
            atomicAdd(&acc_positions_sizes[k].w, pos_size_sum[k].w);
            atomicAdd(&acc_colors_densities[k].x, col_dens_sum[k].x);
            atomicAdd(&acc_colors_densities[k].y, col_dens_sum[k].y);
            atomicAdd(&acc_colors_densities[k].z, col_dens_sum[k].z);
            atomicAdd(&acc_colors_densities[k].w, col_dens_sum[k].w);
            atomicAdd(&acc_centers[k].x, cent_sum[k].x);
            atomicAdd(&acc_centers[k].y, cent_sum[k].y);
            atomicAdd(&acc_normals[k].x, norm_sum[k].x);
            atomicAdd(&acc_normals[k].y, norm_sum[k].y);
            atomicAdd(&acc_normals[k].z, norm_sum[k].z);
            atomicAdd(&acc_shapes[k].xx, cov_sum[k].xx);
            atomicAdd(&acc_shapes[k].xy, cov_sum[k].xy);
            atomicAdd(&acc_shapes[k].xz, cov_sum[k].xz);
            atomicAdd(&acc_shapes[k].yy, cov_sum[k].yy);
            atomicAdd(&acc_shapes[k].yz, cov_sum[k].yz);
            atomicAdd(&acc_shapes[k].zz, cov_sum[k].zz);
        }
    }

    __syncthreads();

    //int idx = r * rows + c;
    int idx = c;

    // update here
    if(idx < nb_clusters)
    {
        float4 acc_ps = acc_positions_sizes[idx];
        float size = acc_ps.w;
        float4 mean_cd = acc_colors_densities[idx] / size;

        float3 new_color = make_float3(mean_cd.x, mean_cd.y, mean_cd.z);
        //printf("%f %f %f", new_color.x, new_color.y, new_color.z);
        float new_density = mean_cd.w;
        float3 new_position = make_float3(acc_ps.x, acc_ps.y, acc_ps.z) / size;
        float2 new_center = acc_centers[idx] / size;

        Cov3 new_shape = acc_shapes[idx] / size - outer_product(new_position);

        float3 eigen_vals;
        Mat33 eigen_vecs;
        eigenDecomposition(new_shape, eigen_vecs, eigen_vals, 10);

        float3 new_normal;

        if(eigen_vals.x < 1e-6f || eigen_vals.y < 1e-6f || eigen_vals.x / eigen_vals.y > 100.f)
            new_normal = eigen_vecs.rows[2];
        else new_normal = make_float3(acc_normals[idx]) / size;

        float new_radius = sqrtf(1.0f / (PI * new_density));

        positions[idx] = make_float4(new_position);
        colors[idx] = make_float4(new_color);
        normals[idx] = make_float4(new_normal);
        crd[idx] = make_float4(new_center.x, new_center.y, new_radius, new_density);
        shapes[idx] = new_shape;
    }
}


__global__ void initRandStates_kernel(curandState *state,
				      unsigned int width,
				      unsigned int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int id = y*width+x;
    
    if(x<width && y<height)
    {
	curand_init(1234, id, 0, &state[id]);
    }
}

__global__ void generateSeeds_kernel(ushort2* __restrict__ seeds_queue_buffer,
				     curandState *state,
				     int* nb_seeds,
				     cudaTextureObject_t tex_density,
				     const int width,
				     const int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x>=width | y>=height)
	return;
    
    int id = y*width+x;
    
    float density = tex2D<float>(tex_density, x, y);
    
    if(density>0.f)
    {
	curandState rng = state[id];
	
	if(curand_uniform(&rng) < density)
	{
	    unsigned int seedId = atomicAggInc(nb_seeds);
	    seeds_queue_buffer[seedId] = make_ushort2(x,y);
	}
	
	
	state[id] = rng;
    }
}


__global__ void initClusters2_kernel(float4* __restrict__ positions,
				     float4* __restrict__ colors,
				     float4* __restrict__ normals,
				     float4* __restrict__ crd,
				     const ushort2* __restrict__ seeds,
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
				     const int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx >= nb_clusters)
        return;
    
    //printf("x %d  y %d", seeds[idx].x, seeds[idx].y);
    ushort2 p_seed = seeds[idx];
    
    float4 pos = tex2D<float4>(tex_positions, p_seed.x, p_seed.y);
    float4 color = tex2D<float4>(tex_lab, p_seed.x, p_seed.y);
    float4 normal = tex2D<float4>(tex_normals, p_seed.x, p_seed.y); 
    float density = tex2D<float>(tex_density, p_seed.x, p_seed.y);
    positions[idx] = pos;
    normals[idx] = normal;
    colors[idx] = color;
    
    float r = sqrtf(1.0f / (PI * density));
    crd[idx] = make_float4(float(p_seed.x), float(p_seed.y), r, density);
    
    acc_positions_sizes[idx].x = pos.x;
    acc_positions_sizes[idx].y = pos.y;
    acc_positions_sizes[idx].z = pos.z;
    acc_positions_sizes[idx].w = 1.f;
    acc_centers[idx].x = p_seed.x;
    acc_centers[idx].y = p_seed.y;
    acc_colors_densities[idx].x = color.x;
    acc_colors_densities[idx].y = color.y;
    acc_colors_densities[idx].z = color.z;
    acc_colors_densities[idx].w = density;
    acc_normals[idx] = normal;
    
    Cov3 cov = outer_product(make_float3(pos));
    acc_shapes[idx] = cov;
        
    im_labels[p_seed.y * step + p_seed.x] = idx;
    
}

// #define SHM_QUEUE_SIZE 256

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
				  Queue<ushort2> * __restrict__ queue,
				  unsigned int * __restrict__ syncCounter,
				  int * __restrict__ endFlag,
				  const float compactness,
				  const float normal_weight,
				  const float lambda,
				  const float radius_meter,
				  const int max_iterations,
				  const int nb_clusters,
				  const int step,
				  const int width,
				  const int height)
{
//     __shared__ Queue<ushort2> shm_queue[2];
//     __shared__ ushort2 shm_buf[2*SHM_QUEUE_SIZE];
    
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    const int neighbor_pos[4][2] = {{-1,0},
				    {1,0},
				    {0,-1},
				    {0,1}};
    const int neighbor_id[4] = {-1,
				1,
				-step,
				step};
    
//     if(tid==0)
//     {
// 	shm_queue[0].buf = &shm_buf[0];
// 	shm_queue[1].buf = &shm_buf[SHM_QUEUE_SIZE];
// 	shm_queue[0].bufSize = shm_queue[1].bufSize = SHM_QUEUE_SIZE;
// 	shm_queue[0].head = shm_queue[1].head = 0;
// 	shm_queue[0].nbElt = shm_queue[1].nbElt = 0;
//     }
    
//     __syncthreads();
    
    int cur_queue = 0;
    bool finished = false;
    
    for(int k=0; k<max_iterations && !finished; k++)
    {
	for(;;)
	{
	    ushort2 elt;
// 	    bool has_elt = shm_queue[cur_queue].dequeue(&elt);
// 	    if(!has_elt)
// 		has_elt = queue[cur_queue].dequeue(&elt);
 	    bool has_elt = queue[cur_queue].dequeue(&elt);
	    
	    if(__syncthreads_and(!has_elt))
		break;
	    if(has_elt)
	    {
		int id = elt.y*step+elt.x;
		int label = im_labels[id];
				
		float3 cluster_position = make_float3(positions[label]);
		float3 cluster_color = make_float3(colors[label]);
		float3 cluster_normal = make_float3(normals[label]);
		float4 cluster_crd = crd[label];
		float2 cluster_center = make_float2(cluster_crd.x, cluster_crd.y);
		float cluster_radius = cluster_crd.z;
		float win_size = lambda * cluster_radius;
		
		for(int n=0; n<4; n++)
		{
		    int2 npos;
		    int n_id = id+neighbor_id[n];
		    npos.x = elt.x+neighbor_pos[n][0];
		    npos.y = elt.y+neighbor_pos[n][1];
		    
		    float2 pos = make_float2(npos.x, npos.y);
		    
		    if(npos.x>=0 && npos.x<width
		       && npos.y>=0 && npos.y<height)
		    {
			float3 position = make_float3(tex2D<float4>(tex_positions, npos.x, npos.y));
			
			if(position.z<=0.f)
			    continue;
			
			float3 color = make_float3(tex2D<float4>(tex_lab, npos.x, npos.y));
			float3 normal = make_float3(tex2D<float4>(tex_normals, npos.x, npos.y));
			float density = tex2D<float>(tex_density, npos.x, npos.y);
			
			float dist = computeDistance(cluster_position, cluster_color, cluster_normal, cluster_center,
						     position, color, normal, pos,
						     radius_meter,
						     compactness,
						     normal_weight,
						     win_size);
			
			
			for(;;)
			{
			    float curDist;
			    int cur_label = im_labels[n_id];
			    
			    if(cur_label<0){
				curDist = INFINITY;
			    }else if(cur_label==label){
				curDist = dist;
			    }else{
				float3 cur_cluster_position = make_float3(positions[cur_label]);
				float3 cur_cluster_color = make_float3(colors[cur_label]);
				float3 cur_cluster_normal = make_float3(normals[cur_label]);
				float4 cur_cluster_crd = crd[cur_label];
				float2 cur_cluster_center = make_float2(cur_cluster_crd.x, cur_cluster_crd.y);
				float cur_cluster_radius = cur_cluster_crd.z;
				float cur_win_size = lambda * cur_cluster_radius;
				
				curDist = computeDistance(cur_cluster_position, cur_cluster_color, cur_cluster_normal, cur_cluster_center,
							  position, color, normal, pos,
							  radius_meter,
							  compactness,
							  normal_weight,
							  cur_win_size);
				
			    }
			    if(dist<curDist)
			    {
				int old = atomicCAS(&im_labels[n_id], cur_label, label);
				if(old==cur_label)
				{
				    ushort2 n_elt = make_ushort2(npos.x, npos.y);
//  				    if(!shm_queue[1-cur_queue].enqueue(n_elt))
					queue[1-cur_queue].enqueue(n_elt);
				    
// 				    
//				    if(!queue[1-cur_queue].enqueue(n_elt))
// 				    {
// 					printf("queue full\n");
// 				    }
				    
				    // increment "label" accumulator
				    Cov3 cov = outer_product(position);
				    /*
				    atomicAdd(&acc_positions_sizes[label].x, 0.f);
				    atomicAdd(&acc_positions_sizes[label].y, 0.f);
				    atomicAdd(&acc_positions_sizes[label].z, 0.f);
				    atomicAdd(&acc_positions_sizes[label].w, 0.f);
				    atomicAdd(&acc_colors_densities[label].x, 0.f);
				    atomicAdd(&acc_colors_densities[label].y, 0.f);
				    atomicAdd(&acc_colors_densities[label].z, 0.f);
				    atomicAdd(&acc_colors_densities[label].w, 0.f);
				    atomicAdd(&acc_centers[label].x, 0.f);
				    atomicAdd(&acc_centers[label].y, 0.f);
				    atomicAdd(&acc_normals[label].x, 0.f);
				    atomicAdd(&acc_normals[label].y, 0.f);
				    atomicAdd(&acc_normals[label].z, 0.f);
				    
				    atomicAdd(&acc_shapes[label].xx, 0.f);
				    atomicAdd(&acc_shapes[label].xy, 0.f);
				    atomicAdd(&acc_shapes[label].xz, 0.f);
				    atomicAdd(&acc_shapes[label].yy, 0.f);
				    atomicAdd(&acc_shapes[label].yz, 0.f);
				    atomicAdd(&acc_shapes[label].zz, 0.f);
				    */
				    
// 				    if(position.z!=0)
// 				    {
					atomicAdd(&acc_positions_sizes[label].x, position.x);
					atomicAdd(&acc_positions_sizes[label].y, position.y);
					atomicAdd(&acc_positions_sizes[label].z, position.z);
					atomicAdd(&acc_positions_sizes[label].w, 1.0f);
					atomicAdd(&acc_colors_densities[label].x, color.x);
					atomicAdd(&acc_colors_densities[label].y, color.y);
					atomicAdd(&acc_colors_densities[label].z, color.z);
					atomicAdd(&acc_colors_densities[label].w, density);
					atomicAdd(&acc_centers[label].x, pos.x);
					atomicAdd(&acc_centers[label].y, pos.y);
					atomicAdd(&acc_normals[label].x, normal.x);
					atomicAdd(&acc_normals[label].y, normal.y);
					atomicAdd(&acc_normals[label].z, normal.z);
					
					atomicAdd(&acc_shapes[label].xx, cov.xx);
					atomicAdd(&acc_shapes[label].xy, cov.xy);
					atomicAdd(&acc_shapes[label].xz, cov.xz);
					atomicAdd(&acc_shapes[label].yy, cov.yy);
					atomicAdd(&acc_shapes[label].yz, cov.yz);
					atomicAdd(&acc_shapes[label].zz, cov.zz);
					
					
					// and decrement "cur_label" accumulator
					atomicAdd(&acc_positions_sizes[cur_label].x, -position.x);
					atomicAdd(&acc_positions_sizes[cur_label].y, -position.y);
					atomicAdd(&acc_positions_sizes[cur_label].z, -position.z);
					atomicAdd(&acc_positions_sizes[cur_label].w, -1.0f);
					atomicAdd(&acc_colors_densities[cur_label].x, -color.x);
					atomicAdd(&acc_colors_densities[cur_label].y, -color.y);
					atomicAdd(&acc_colors_densities[cur_label].z, -color.z);
					atomicAdd(&acc_colors_densities[cur_label].w, -density);
					atomicAdd(&acc_centers[cur_label].x, -pos.x);
					atomicAdd(&acc_centers[cur_label].y, -pos.y);
					atomicAdd(&acc_normals[cur_label].x, -normal.x);
					atomicAdd(&acc_normals[cur_label].y, -normal.y);
					atomicAdd(&acc_normals[cur_label].z, -normal.z);
					
					atomicAdd(&acc_shapes[cur_label].xx, -cov.xx);
					atomicAdd(&acc_shapes[cur_label].xy, -cov.xy);
					atomicAdd(&acc_shapes[cur_label].xz, -cov.xz);
					atomicAdd(&acc_shapes[cur_label].yy, -cov.yy);
					atomicAdd(&acc_shapes[cur_label].yz, -cov.yz);
					atomicAdd(&acc_shapes[cur_label].zz, -cov.zz);
// 				    }
				    
				    break;
				}
			    }else
			    {
				break;
			    }
			}
		    }
		}
	    }
	    
	    __syncthreads();
	}
	
//  	if(tid==0 && shm_queue[1-cur_queue].nbElt!=0)
//  	{
// 	    atomicAdd(endFlag, 1);
// // 	    printf("%d waiting other blocks\n", blockIdx.x);
//  	}
	
	syncAllThreads(&syncCounter[0]);
	
// 	if(*endFlag == 0)
// 	{
// 	    return;
// 	}

// 	if(tid==0 && shm_queue[1-cur_queue].nbElt!=0)
// 	{
// 	    atomicAdd(endFlag, -1);
// 	}
	
	if(queue[1-cur_queue].nbElt==0)
	{
	    finished = true;
// 	    if(tid==0 && blockIdx.x==0)
// 	    {
// 		printf("iterations : %d\n", k);
// 	    }
	    
// 	    printf("return\n");
// 	    return;
	}

	__syncthreads();
	
	cur_queue = 1-cur_queue;
	
	// update clusters
	for(int k = global_id; k<nb_clusters; k+=blockDim.x*gridDim.x)
	{
	    float4 acc_ps = acc_positions_sizes[k];
	    float size = acc_ps.w;
	    float4 mean_cd = acc_colors_densities[k] / size;
	    
	    float3 new_color = make_float3(mean_cd.x, mean_cd.y, mean_cd.z);
	    //printf("%f %f %f", new_color.x, new_color.y, new_color.z);
	    float new_density = mean_cd.w;
	    float3 new_position = make_float3(acc_ps.x, acc_ps.y, acc_ps.z) / size;
	    float2 new_center = acc_centers[k] / size;
	    
	    Cov3 new_shape = acc_shapes[k] / size - outer_product(new_position);
	    
	    float3 eigen_vals;
	    Mat33 eigen_vecs;
	    eigenDecomposition(new_shape, eigen_vecs, eigen_vals, 10);
	    
	    float3 new_normal;
	    
	    if(eigen_vals.x < 1e-6f || eigen_vals.y < 1e-6f || eigen_vals.x / eigen_vals.y > 100.f)
		new_normal = eigen_vecs.rows[2];
	    else new_normal = make_float3(acc_normals[k]) / size;
	    
	    float new_radius = sqrtf(1.0f / (PI * new_density));
	    
	    positions[k] = make_float4(new_position);
	    colors[k] = make_float4(new_color);
	    normals[k] = make_float4(new_normal);
	    crd[k] = make_float4(new_center.x, new_center.y, new_radius, new_density);
	    //shapes[k] = new_shape;
	}
	
	
	
// 	if(tid==0 && blockIdx.x==0)
// 	{
// 	    printf("swap buffer\n");
// 	}
	
	syncAllThreads(&syncCounter[1]);
	
    }

}



__global__ void renderBoundaryImage_kernel(uchar4* __restrict__ boundary_im,
                                           cudaTextureObject_t tex_labels,
                                           const int step,
                                           const int rows,
                                           const int cols)
{
    int c = blockIdx.x *blockDim.x + threadIdx.x;
    int r = blockIdx.y *blockDim.y + threadIdx.y;

    if(c == 0 || c >= cols - 1  || r == 0 || r >= rows - 1)
        return;

    int l1 = tex2D<int>(tex_labels, c, r);
    int l2 = tex2D<int>(tex_labels, c + 1, r);
    int l3 = tex2D<int>(tex_labels, c - 1, r);
    int l4 = tex2D<int>(tex_labels, c, r - 1);
//    int l5 = tex2D<int>(tex_labels, c, r + 1);
//    int l6 = tex2D<int>(tex_labels, c + 1, r + 1);
//    int l7 = tex2D<int>(tex_labels, c - 1, r - 1);
//    int l8 = tex2D<int>(tex_labels, c + 1, r - 1);
//    int l9 = tex2D<int>(tex_labels, c - 1, r + 1);

    if(l1 != l2 || l1 != l3 || l1 != l4/* || l1 != l5 || l1 != l6 || l1 != l7 || l1 != l8 || l1 != l9*/)
        boundary_im[r * step +  c] = make_uchar4(0, 0, 0, 1);
}

__global__ void renderSuperpixelsColorImage_kernel(uchar4* __restrict__ color_im,
                                                   cudaTextureObject_t tex_labels,
                                                   const float4* __restrict__ superpixel_colors,
                                                   const int step,
                                                   const int rows,
                                                   const int cols)
{
    int c = blockIdx.x *blockDim.x + threadIdx.x;
    int r = blockIdx.y *blockDim.y + threadIdx.y;

    if(c == 0 || c >= cols - 1  || r == 0 || r >= rows - 1)
        return;

    int id = tex2D<int>(tex_labels, c, r);

    if(id >= 0)
    {
        float3 rgb = labToRgb(make_float3(superpixel_colors[id]));
        //printf("%f %f %f \n", rgb.x, rgb.y, rgb.z);
        color_im[r * step +  c] = make_uchar4((unsigned char)rgb.x, (unsigned char)rgb.y, (unsigned char)rgb.z, 1);
    }
    else
        color_im[r * step +  c] = make_uchar4(0, 0, 0, 1);
}

} // cuda_dasp
