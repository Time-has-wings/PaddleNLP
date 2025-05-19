import os
import paddle.distributed as dist
from dataclasses import dataclass, field

@dataclass 
class ProfileArguments:
    profile_time_flag: int = field(default=0, metadata={"help": "Whether to enable time profiling."})
    profile_forward_only: int = field(default=0, metadata={"help": "Whether to enable forward profiling."})
    profile_memory_flag: int = field(default=0, metadata={"help": "Whether to enable memory profiling."})
    
    profile_seq_len: int = field(default=1024, metadata={"help": "The sequence length for profiling."})
    profile_mixed_precision: str = field(default='bf16', metadata={"help": "Whether to enable mixed precision profiling."})
    profile_global_batch_size: int = field(default=8, metadata={"help": "The global batch size for profiling."})
    profile_layer_num: int = field(default=16, metadata={"help": "The number of layers for profiling."})
    profile_model_name: str = field(default='llama', metadata={"help": "The model name for profiling."})
    
    profile_type: str = field(default='memory', metadata={"help": "The type of profiling, either 'memory' or 'computation'."})
    profile_mode: str = field(default='static', metadata={"help": "The mode of profiling, either 'static', 'batch' or 'sequence'."})

    profile_min_batch_size: int = field(default=1, metadata={"help": "The minimum batch size for profiling."})
    profile_max_batch_size: int = field(default=12, metadata={"help": "The maximum batch size for profiling."})    
    profile_batch_size_step: int = field(default=1, metadata={"help": "The step size for batch size profiling."})
    
    layernum_min: int = field(default=2, metadata={"help": "The minimum number of layers for profiling."})
    layernum_max: int = field(default=4, metadata={"help": "The maximum number of layers for profiling."})
    
    profile_seq_length_list: str = field(default='1024,2048', metadata={"help": "The sequence length list for profiling."})
    
    num_layertype: int = field(default=1, metadata={"help": "1:decoder-only and encoder-only, 2:encoder-decoder"})    

    max_tp_deg: int = field(default=1, metadata={"help": "The maximum tensor parallel degree."})


class BaseProfiler():
    def __init__(self, args:ProfileArguments):
        self.args = args
        self.time_path = None
        self.mem_path = None
        self.path = os.getcwd()
        
    def get_memory_profiling_path(self):
        if self.mem_path is not None:
            return self.mem_path
        args = self.args
        memory_file_name = f'configs/memory_profiling_{args.profile_mixed_precision}_{args.profile_model_name}_rank[{dist.get_rank()}].json'
        self.mem_path = os.path.join(self.path, memory_file_name)
        return self.mem_path
    
    def get_time_profiling_path(self):
        if self.time_path is not None:
            return self.time_path
        args = self.args
        time_file_name = f'configs/computation_profiling_{args.profile_mixed_precision}_{args.profile_model_name}_rank[{dist.get_rank()}].json'
        self.time_path = os.path.join(self.path, time_file_name)
        return self.time_path