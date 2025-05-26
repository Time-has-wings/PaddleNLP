# # import numpy as np
# # from scipy.optimize import curve_fit

# # # 定义线性函数模型
# # def linear_func(x, k, b):
# #     return k * x + b  # 注意：自变量x必须在第一个参数

# # # 生成模拟数据（示例）
# # x_data = np.array([0, 1, 2, 3, 4])
# # y_data = np.array([1.1, 2.9, 4.8, 6.7, 8.5])  # 真实关系：y = 2x + 1（含噪声）

# # # 执行拟合
# # params, covariance = curve_fit(linear_func, x_data, y_data)
# # print('params:', params)  # 输出拟合参数
# # print(type(params))  # 输出参数类型
# # k_fit, b_fit = params  # 拟合参数
# # print(f"拟合斜率 k = {k_fit:.2f}, 截距 b = {b_fit:.2f}")

# # python -m paddle.distributed.launch --gpus 4,5,6,7 --log_dir ./output/shard_learning learning.py 
# import paddle
# import paddle.distributed as dist
# from paddle.distributed.auto_parallel.api import *

# if __name__ == '__main__':
#     normal_tensor = paddle.to_tensor([  # (batch_size, seq_len) (4, 8)
#                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#                                         [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
#                                         [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
#                                         [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
#                                     ])
#     mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['tp', 'dp'])
#     dtensor = dist.shard_tensor(normal_tensor, mesh, [dist.Replicate(), dist.Shard(0)])
#     print(f'dtensor is {dtensor}')
#     local_tensor = dtensor._local_value()
#     print(f'local_value is {local_tensor}')
    
#     mesh2 = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['tp', 'dp'])
#     rank = dist.get_rank()
#     if rank == 1 or rank == 3: 
#         # shape = local_tensor.shape
#         # local_tensor = paddle.empty([2, 8])
#         pass
#     dtensor = dtensor_from_local(local_tensor, mesh2, [dist.Replicate(), dist.Shard(0)])
#     print(f'dtensor is {dtensor}')
#     print(f'dtensor._local_value() is {dtensor._local_value()}')
    
#     # dtensor2normaltensor = dist.unshard_dtensor(dtensor)
#     # print(f'dtensor2normaltensor is {dtensor2normaltensor}')

#     # rubbsih = dtensor_to_local(dtensor, mesh, [dist.Shard(0), dist.Replicate()]) # 这个的效果和dtensor._local_value()是一样的
#     # print(f'rubbsih is {rubbsih}')
    
#     # mesh2 = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['tp', 'dp'])
#     # rubbsih2 = dtensor_from_local(rubbsih, mesh2, [dist.Replicate(), dist.Shard(0)])
#     # print(f'rubbsih2 is {rubbsih2}')
    
#     # rank = dist.get_rank()
    
#     # if rank == 0:
#     #     local_tensor = paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
#     # rank = dist.get_rank()
#     # local_list = [rank for _ in range(12)]
#     # local_tensor = paddle.to_tensor(local_list)
#     # mesh = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['tp', 'dp'])
#     # dtensor = dtensor_from_local(local_tensor, mesh, [dist.Replicate(), dist.Shard(0)])
#     # print(f'dtensor is {dtensor}')
#     # dtensor_local_value = dtensor._local_value()
#     # print(f'dtensor_local_value is {dtensor_local_value}')
    
#     # mesh2 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['tp', 'dp'])
#     # dtensor2 = dtensor_to_local(dtensor, mesh2, [dist.Shard(0), dist.Replicate()])
#     # print(f'dtensor2 is {dtensor2}')

import numpy as np
def chunk_like_torch(size, chunks):
    """Implement torch.arange(size).chunk(chunks) behavior using numpy"""
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    
    # Calculate chunk size like PyTorch does
    chunk_size = (size + chunks - 1) // chunks  # ceiling division
    
    # Create splits
    splits = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= size:
            break
        end = min(start + chunk_size, size)
        splits.append(np.arange(start, end))
    
    return splits

if __name__ == '__main__':
    size = 10
    chunks = 3
    result = chunk_like_torch(size, chunks)
    print(result) 
    
    size = 10
    chunks = 4
    result = chunk_like_torch(size, chunks)
    print(result)  
    
    size = 32
    chunks = 4
    result = chunk_like_torch(size, chunks)
    print(result)
    for i in range(len(result)):
        print(result[i].shape)