from .base_profiler import BaseProfiler, ProfileArguments
import paddle
from ..utils import read_json_config, write_json_config, num2str
import numpy as np
from paddle import core, framework

class RuntimeProfiler(BaseProfiler):
    def __init__(self, args:ProfileArguments):
        super().__init__(args)
        self.profile_time_flag = args.profile_time_flag
        self.profile_memory_flag = args.profile_memory_flag
        self.profile_forward_only = args.profile_forward_only
        
    # ================Time Profiling================
    def set_time_profiler(self, start_iter=10, end_iter=20): # [start_iter, end_iter)
        assert end_iter > start_iter, "End iteration must be greater than start iteration"
        
        self.start_iter = start_iter
        self.end_iter = end_iter
        
        self.start_event = paddle.device.Event(enable_timing=True)
        self.end_event = paddle.device.Event(enable_timing=True)
        self.time_list = []
    
    def profile_time_start(self, iter):
        if self.profile_time_flag == False:
            return
        
        if self.start_iter <= iter and iter < self.end_iter:
            paddle.device.synchronize()
            self.start_event.record()
        elif iter == self.end_iter:
            self._save_time_results()
        
    def profile_time_end(self, iter):
        if self.profile_time_flag == False:
            return
        
        if self.start_iter <= iter and iter < self.end_iter:
            self.end_event.record()
            paddle.device.synchronize()
            time_cost = self.start_event.elapsed_time(self.end_event)
            self.time_list.append(time_cost)
    
    def _save_time_results(self):
        if self.profile_time_flag == False:
            return
        
        ave_time = sum(self.time_list) / len(self.time_list)
        print(f"Average time cost: {ave_time:.4f} ms")
        print(f'original time cost: {self.time_list}')
        
        mean, std = np.mean(self.time_list), np.std(self.time_list)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        print(f'Time mean: {mean:.4f} ms, std: {std:.4f} ms')
        print(f"Time cost range: [{lower_bound:.4f}, {upper_bound:.4f}]")
        self.time_list = [time for time in self.time_list if time >= lower_bound and time <= upper_bound]
        print(f"After removing outliers, time cost: {self.time_list}")
        
        time_path = self.get_time_profiling_path()
        config = read_json_config(time_path) 
        
        layernum_info = num2str(self.args.profile_layer_num, "layernum")
        seq_info = num2str(self.args.profile_seq_len, "seq")
        
        key = f'{layernum_info}_bsz{self.args.profile_global_batch_size}_{seq_info}'
        config[key] = ave_time
        
        write_json_config(time_path, config)
        print(f"Already written profiled time into config file {time_path}!\n")
            
    # ================Memory Profiling================
    def set_memory_profiler(self, max_profile_memory_iter=5):
        self.max_profile_memory_iter = max_profile_memory_iter
        self.mem_dict = {}
        self.current_device = framework._current_expected_place_()

    def profile_memory(self, iter, stage:str=""):
        if self.args.profile_memory_flag == False:
            return
        
        if stage == "Before Forward":
            core.device_memory_stat_reset_peak_value("Allocated", self.current_device.get_device_id())
        
        mem_dict = self.mem_dict
        max_memory_allocated = core.device_memory_stat_peak_value("Allocated", self.current_device.get_device_id()) / 2**20
        current_memory_allocated = core.device_memory_stat_current_value("Allocated", self.current_device.get_device_id()) / 2**20
        print(f'stage: {stage}, iter: {iter}, current_memory_allocated: {current_memory_allocated} MB, max_memory_allocated: {max_memory_allocated} MB')
        if stage == "Before Forward":
            mem_dict[f'iter_{iter}_before_forward'] = current_memory_allocated
        elif stage == "After Forward":
            mem_dict[f'iter_{iter}_after_forward'] = current_memory_allocated
        elif stage == "After Backward":
            mem_dict[f'iter_{iter}_after_backward'] = current_memory_allocated
            mem_dict[f'iter_{iter}_after_backward_max'] = max_memory_allocated
        elif stage == "After Optimizer":
            pass

    def post_profile_memory(self, iter):
        if self.args.profile_memory_flag == False:
            return
        
        if iter == self.max_profile_memory_iter:
            mem_dict = self.mem_dict
            mem_dict["model_states"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_backward']
            mem_dict["model_states_and_peak_activation"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_backward_max']
            mem_dict["peak_activation"] = mem_dict["model_states_and_peak_activation"] - mem_dict["model_states"]
            

            # mem_dict["model_states_and_activation"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_forward']
            # mem_dict["activation"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_forward'] - mem_dict[f"iter_{self.max_profile_memory_iter - 1}_before_forward"]
            
            print("Memory profiling results:")
            for key, val in mem_dict.items():
                print(f'\t{key}: {val:.2f} MB')
            # mem_dict = self.mem_dict
            # for key in mem_dict.keys():
            #     mem_dict[key] = mem_dict[key] / 1024 / 1024
            
            # print(f"Memory dict: {mem_dict}")
            # write_json_config(self.get_memory_profiling_path(), mem_dict)
            # print(f"Already written profiled memory into config file {self.get_memory_profiling_path()}!\n")