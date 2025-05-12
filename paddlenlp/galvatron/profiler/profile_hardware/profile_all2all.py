import paddle
import numpy as np
import random
import json
import argparse

def printf(msg, prefix="[ProfileHardware] "):
    text_color = "\033[32m"
    reset_color = "\033[0m"
    print(f"{text_color}{prefix}{msg}{reset_color}")

def single_all_to_all(input, group):
    world_size = paddle.distributed.get_world_size(group)
    input_t = input.reshape((world_size, -1))
    output = paddle.empty_like(input_t)
    paddle.distributed.alltoall_single(output, input_t, group=group)
    return output

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)
    
def train(args):
    # Initialize the distributed environment
    paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    local_rank = paddle.distributed.ParallelEnv().local_rank
    set_seed(rank)
    world_size = paddle.distributed.get_world_size()

    # set group # TODO 根据情况设置并行组
    mp_group = paddle.distributed.new_group(ranks=[i for i in range(world_size)])

    # profile
    start_event = paddle.device.Event(enable_timing=True)
    end_event = paddle.device.Event(enable_timing=True)
    time_list = []

    printf("warm up...")
    for _ in range(5):
        input = np.random.rand(*(args.local_batch_size, 512, 1024))
        input = paddle.to_tensor(input, dtype='bfloat16', place=f'gpu:{local_rank}')
        output = single_all_to_all(input, group=mp_group)
        
    printf("start profiling...")
    for _ in range(20):
        input = np.random.rand(*(args.local_batch_size, 512, 1024))
        input = paddle.to_tensor(input, dtype='bfloat16', place=f'gpu:{local_rank}')
        
        paddle.device.synchronize()
        paddle.distributed.barrier(group=mp_group)
        start_event.record()
        output = single_all_to_all(input, group=mp_group)
        end_event.record()
        paddle.device.synchronize()
        duration = start_event.elapsed_time(end_event)
        print(f"device: {local_rank}, time: {duration}")
        time_list.append(duration)
    
    printf(f"origin time list: {time_list}")
    mean = np.mean(time_list)
    std = np.std(time_list)
    threshold = 3 * std
    time_list = [x for x in time_list if abs(x - mean) <= threshold]
    printf(f"filtered time list: {time_list}")
    
    per_comm_time = sum(time_list) / len(time_list)
    per_comm_time = paddle.to_tensor([per_comm_time], dtype='float32', place=f'gpu:{local_rank}')
    paddle.distributed.all_reduce(per_comm_time, op=paddle.distributed.ReduceOp.SUM, group=mp_group)
    per_comm_time = per_comm_time.numpy()[0] / mp_group.nranks
    
    printf(f'sum(time_list): {sum(time_list)}, len(time_list): {len(time_list)}')
    print(f'comm_time_{args.local_batch_size}MB_ppsize{args.pp_degree}_tp{args.mp_degree}: {per_comm_time}ms')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Profile all2all communication")
    parser.add_argument('--local_batch_size', type=int, default=1, help='Local batch size')
    parser.add_argument('--mp_degree', type=int, default=1, help='Pipeline parallel degree')
    parser.add_argument('--pp_degree', type=int, default=1, help='Tensor parallel degree')
    args = parser.parse_args()
    train(args)