# 该操作本质上其实有dist.send和dist.recv的操作
# 目前使用分布式张量的reshard方式实现
# 再使用一下dist.send操作试下
# 使用dist.send较为合理
import numpy as np
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset
import paddle
import paddle.profiler
from tqdm import tqdm
import os
import argparse
import json

def printf(string, prefix="[ProfileHardware] "):
    text_color = "\033[32m"
    reset_color = "\033[0m"
    print(f"{text_color}{prefix}{string}{reset_color}")

class RandomDataset(Dataset):
    def __init__(self, local_batch_size):
        self.dataset_size = local_batch_size * 11  # [note] 设置为11
        self.input = np.random.rand(*(self.dataset_size, 512, 1024))
        self.label = np.random.rand(*(self.dataset_size, 512, 1024))

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")
        input = paddle.to_tensor(self.input[idx], dtype='float32')
        label = paddle.to_tensor(self.label[idx], dtype='float32')
        return input, label

class LinearModel(nn.Layer):
    def __init__(self, pp_rank, pp_world_size, mesh):
        super().__init__()
        
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size
        self.mesh = mesh
        
    def forward(self, x):
        x.stop_gradient = False
        for i in range(self.pp_world_size):
            if i != self.pp_world_size - 1:
                x = dist.reshard(x, self.mesh[i + 1], [dist.Replicate()])
        return x

class P2PCommModel(nn.Layer):
    def __init__(self, rank, send_rank, recv_rank, mesh):
        super().__init__()
        self.rank = rank
        self.send_rank = send_rank
        self.recv_rank = recv_rank
        self.mesh = mesh
        
    def forward(self, x):
        x.stop_gradient = False
        for i in range(2):
            if i == 0:
                x = dist.reshard(x, self.mesh[i + 1], [dist.Replicate()])
        return x
    
def test(args):
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = dist.ParallelEnv().local_rank
    group = dist.new_group(ranks=[i for i in range(world_size)])
    mesh = dist.ProcessMesh([i for i in range(world_size)], dim_names=['pp'])
    model = LinearModel(rank, world_size, mesh)
    
    local_batch_size = args.local_batch_size
    printf(f"local_batch_size: {local_batch_size}")
    dataset = RandomDataset(local_batch_size)
    sampler = BatchSampler(dataset, batch_size=local_batch_size)
    trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
    dist_dataloader = dist.shard_dataloader(trainloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])
    
    # print some info
    p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
    printf(f"p2p_message_size: {p2p_message_size} MB")
    
    with paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.GPU],
        scheduler=(1, 12),
        ) as p:
        for i, input in enumerate(tqdm(dist_dataloader)):
            data = input[0]
            out = model(data)
            p.step() 
            
    # save profiler result
    save_file_path = f'./profile_logs/profile_p2p/rank{rank}.json'
    if os.path.exists(os.path.dirname(save_file_path)) == False:
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)    
    p.export(save_file_path)  
    
    # analyse profiler result
    def timestr2timenum(timestr):
        string = timestr.split(" ")
        num = float(string[0])
        unit = string[1]
        if unit == "ms":
            return num
        elif unit == "us":
            return num / 1000
        else:
            assert False, f"unit {unit} not supported"
    
    send_recv_time_list = []
    with open(save_file_path, "r") as f:
        data = json.load(f)
        traceEvents = data["traceEvents"]
        for event in traceEvents:
            if "name" in event:
                event_name = event['name']
                if "SendRecv" in event_name:
                    start_time = timestr2timenum(event["args"]["start_time"])
                    end_time = timestr2timenum(event["args"]["end_time"])
                    send_recv_time_list.append(end_time - start_time)
    
    send_recv_time_sum = sum(send_recv_time_list)
    send_recv_time_sum /= 10  # 10次profiler迭代
    printf(f'send_recv_time_sum: {send_recv_time_sum} ms')
    
# def train(args):
#     dist.init_parallel_env()
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     # TODO 构建PP通信组
#     send_rank = world_size // 2 - 1
#     recv_rank = world_size // 2
#     mesh = dist.ProcessMesh([send_rank, recv_rank], dim_names=['pp'])
#     printf(f'mesh is {mesh}')
    
#     # set model
#     if rank == send_rank or rank == recv_rank:
#         model = P2PCommModel(rank, send_rank, recv_rank, mesh)
#     else:
#         model = None
    
#     # set dataset
#     local_batch_size = args.local_batch_size
#     printf(f"local_batch_size: {local_batch_size}")
#     dataset = RandomDataset(local_batch_size)
#     sampler = BatchSampler(dataset, batch_size=local_batch_size)
#     trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
#     dist_dataloader = dist.shard_dataloader(trainloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])
    
#     # print some info
#     p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
#     printf(f"p2p_message_size: {p2p_message_size} MB")
    
#     # profiler
#     with paddle.profiler.Profiler(
#         targets=[paddle.profiler.ProfilerTarget.GPU],
#         scheduler=(1, 12),
#         ) as p:
#         # for input in 
#         for i, input in enumerate(tqdm(dist_dataloader)):
#             data = input[0]
#             if rank == send_rank or rank == recv_rank:
#                 printf(f'step:{i}')
#                 out = model(data)
#             p.step()   
    
#     # save profiler result
#     save_file_path = f'./profile_logs/profile_p2p/rank{rank}.json'
#     if os.path.exists(os.path.dirname(save_file_path)) == False:
#         os.makedirs(os.path.dirname(save_file_path), exist_ok=True)    
#     p.export(save_file_path)
    
#     # TODO analyze profiler result
    

def use_dist_send_recv(args):
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    local_batch_size = args.local_batch_size
    printf(f"local_batch_size: {local_batch_size}")
    dataset = RandomDataset(local_batch_size)
    sampler = BatchSampler(dataset, batch_size=local_batch_size)
    trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
    
    # print some info
    p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
    printf(f"p2p_message_size: {p2p_message_size} MB")
    
    with paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.GPU],
        scheduler=(1, 12),
        ) as p:
        for i, input in enumerate(tqdm(trainloader)):
            data = input[0]
            data = data.to(paddle.get_device())
            if rank == 0:
                dist.send(data, dst=1)
            elif rank == 1:
                dist.recv(data, src=0)
            p.step() 
            
    # save profiler result
    save_file_path = f'./profile_logs/profile_p2p/rank{rank}_dist.json'
    if os.path.exists(os.path.dirname(save_file_path)) == False:
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)    
    p.export(save_file_path)  
    
    # analyse profiler result
    def timestr2timenum(timestr):
        string = timestr.split(" ")
        num = float(string[0])
        unit = string[1]
        if unit == "ms":
            return num
        elif unit == "us":
            return num / 1000
        else:
            assert False, f"unit {unit} not supported"
    
    send_recv_time_list = []
    with open(save_file_path, "r") as f:
        data = json.load(f)
        traceEvents = data["traceEvents"]
        for event in traceEvents:
            if "name" in event:
                event_name = event['name']
                if "SendRecv" in event_name:
                    start_time = timestr2timenum(event["args"]["start_time"])
                    end_time = timestr2timenum(event["args"]["end_time"])
                    send_recv_time_list.append(end_time - start_time)
    
    send_recv_time_sum = sum(send_recv_time_list)
    send_recv_time_sum /= 10  # 10次profiler迭代
    comm_coe = p2p_message_size / send_recv_time_sum
    
    printf(f'send_recv_time_sum: {send_recv_time_sum} ms')
    printf(f'comm_coe: {comm_coe} MB/ms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddlePaddle P2P Communication Profiler")
    parser.add_argument("--local_batch_size", type=int, default=32, help="local batch size for each rank" )
    args = parser.parse_args()
    # train(args)
    # test(args)
    use_dist_send_recv(args)
