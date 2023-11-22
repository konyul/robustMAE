#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <thrust/device_vector.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

//__device__ __managed__ float *sparse_tensor_ptr[10];

//namespace {
//int const threadsPerBlock = sizeof(unsigned long long) * 8;
//}
//
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
//
//template <typename T_int>
//__global__ void coor_point_to_window_idx_kernel(const T_int* coor,
//                                         T_int* point_to_window_idx,
//                                         //T_int* point_to_pointidx,
//                                         const int num_points, const int NDim) {
//  CUDA_1D_KERNEL_LOOP(index, num_points) {
//    auto coor_offset = coor + index * NDim;
//    // skip invalid points
//    if ((index >= num_points) || (coor_offset[0] == -1)) return;
//
//    int num = 0;
//    int coor_x = coor_offset[0];
//    int coor_y = coor_offset[1];
//    int coor_z = coor_offset[2];
//    // only calculate the coors before this coor[index]
//    for (int i = 0; i < index; ++i) {
//      auto prev_coor = coor + i * NDim;
//      //if (prev_coor[0] == -1) continue;
//
//      // Find all previous points that have the same coors
//      // if find the same coor, record it
//      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
//          (prev_coor[2] == coor_z)) {
//        num++;
//      }
//    }
//    point_to_window_idx[index] = num;
//  }
//}

template <typename T_int>
__global__ void point_to_window_idx_kernel_(const T_int* coor,
                                         T_int* point_to_window_idx,
                                         //T_int* point_to_pointidx,
                                         const int num_points) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    int coor_index= coor[index];
    //if (index >= num_points) return;

    int num = 0;
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      int prev_coor = coor[i];
      //if (prev_coor == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if (prev_coor==coor_index)
        num++;
    }
    point_to_window_idx[index] = num;
  }
}

template <typename T, typename T_int, typename T_bool>
__global__ void assign_voxel_to_sparse_tensor_kernel_(const int nthreads, const T* points,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      T ** tensor_ptr,
                                      T_bool ** tensor_index_ptr,
                                      T_int ** points_index_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_features,
                                      const int num_points, const int NDim,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;
    int in_window_pos = point_to_window_idx[index];
    int window_index = window_indexes[index];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];
    //printf("%d %d %d %d %d\n",index,in_window_pos,window_index,tensor_index,tensor_length);
    //printf("%p %p %p\n",tensor_ptr,*(tensor_ptr+tensor_index),tensor_ptr[tensor_index]); //tensor_ptr+sizeof(float*)*tensor_index);
    //if(in_window_pos>-1){
    auto tensor_offset=
        tensor_ptr[tensor_index]+in_tensor_pos * tensor_length * num_features+in_window_pos*num_features;
    int k = thread_idx % num_features;
    tensor_offset[k] = points[thread_idx];

    auto tensor_index_offset=
        tensor_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    tensor_index_offset[in_window_pos]=true;
    auto points_index_offset=
        points_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    points_index_offset[in_window_pos]=index;
    //}
    //auto tensor_offset=tensor_ptr[tensor_index]+
  }
}



namespace sparse_index{
void sparse_index_gpu_(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  CHECK_INPUT(coors);
  at::cuda::CUDAGuard device_guard(coors.device());
  const int num_points = coors.size(0);
  const int num_features = voxel_features.size(1);
  /*
  auto point_to_window_idx = -at::ones(
    {
      num_points,
    },
    coors.options().dtype(at::kInt));
  */
  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);


  AT_DISPATCH_ALL_TYPES(
     coors.scalar_type(), "determin_point_to_window_idx_", ([&] {
        point_to_window_idx_kernel_<int><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            window_indexes.data_ptr<int>(), point_to_window_idx.data_ptr<int>(),
            num_points);
      }));

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  //std::vector<float*>sparse_tensor_ptr(10);
  //printf("size %d\n",sparse_tensor_list.size());

  thrust::device_vector<float*>sparse_tensor_ptr(10);
  thrust::device_vector<bool*>sparse_tensor_index_ptr(10);
  thrust::device_vector<int*>points_index_ptr(10);
//  float *sparse_tensor_ptr[9];
//  bool *sparse_tensor_index_ptr[9];
  for(int i=0;i<tensor_num;i++)
  {
    sparse_tensor_ptr[i]=sparse_tensor_list[i].contiguous().data_ptr<float>();
    sparse_tensor_index_ptr[i]=sparse_tensor_index_list[i].contiguous().data_ptr<bool>();
    points_index_ptr[i]=points_index_list[i].contiguous().data_ptr<int>();

  }
  // printf("test %p %p %p %p %p %p %p\n",sparse_tensor_ptr,*sparse_tensor_ptr,&sparse_tensor_ptr,&sparse_tensor_ptr[1],sparse_tensor_ptr[1],(sparse_tensor_ptr+1),*(sparse_tensor_ptr+1));

//  float** d_sparse_tensor_ptr;
//  bool **d_sparse_tensor_index_ptr;
//  cudaMalloc((void***)&d_sparse_tensor_ptr, sizeof(sparse_tensor_ptr));
//  cudaMalloc((void***)&d_sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr));
//  cudaMemcpy(d_sparse_tensor_ptr,sparse_tensor_ptr, sizeof(sparse_tensor_ptr), cudaMemcpyHostToDevice);
//  cudaMemcpy(d_sparse_tensor_index_ptr,sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr), cudaMemcpyHostToDevice);
  //return;
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      voxel_features.scalar_type(), "assign_voxel_to_tensor_", ([&] {
        assign_voxel_to_sparse_tensor_kernel_<float, int, bool>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                pts_output_size, voxel_features.contiguous().data_ptr<float>(),
                point_to_window_idx.contiguous().data_ptr<int>(),
                window_regions.contiguous().data_ptr<int>(),
                window_to_tensor_idx.contiguous().data_ptr<int>(),
                window_indexes.contiguous().data_ptr<int>(),
                thrust::raw_pointer_cast(&sparse_tensor_ptr[0]),
                thrust::raw_pointer_cast(&sparse_tensor_index_ptr[0]),
                thrust::raw_pointer_cast(&points_index_ptr[0]),
                tensor_lengthes.contiguous().data_ptr<int>(),
                num_features,num_points, NDim,tensor_num);
      }));
//  cudaFree(d_sparse_tensor_ptr);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return;
  }
}