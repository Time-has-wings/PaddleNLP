from itertools import product
import sys
import copy
import os
from ..utils import read_json_config, write_json_config
from dataclasses import dataclass, field

@dataclass
class ModelProfilerArguments:
    profile_type: str = field(default='memory', metadata={"help": "The type of profiling, either 'memory' or 'computation'."})
    profile_mode: str = field(default='static', metadata={"help": "The mode of profiling, either 'static', 'batch' or 'sequence'."})
    
    profile_fixed_batch_size: int = field(default=8, metadata={"help": "The fixed batch size for profiling."})
    profile_min_batch_size: int = field(default=1, metadata={"help": "The minimum batch size for profiling."})
    profile_max_batch_size: int = field(default=12, metadata={"help": "The maximum batch size for profiling."})
    profile_batch_size_step: int = field(default=1, metadata={"help": "The step size for batch size profiling."})
    
    layernum_min: int = field(default=2, metadata={"help": "The minimum number of layers for profiling."})
    layernum_max: int = field(default=4, metadata={"help": "The maximum number of layers for profiling."})
    
    profile_fixed_seq_length_list: str = field(default='1024,2048', metadata={"help": "The sequence length list for profiling."})
    profile_min_seq_length: int = field(default=1024, metadata={"help": "The minimum sequence length for profiling."})
    profile_max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length for profiling."})
    profile_seq_length_step: int = field(default=1, metadata={"help": "The step size for sequence length profiling."})
    
    num_layertype: int = field(default=1, metadata={"help": "1:decoder-only and encoder-only, 2:encoder-decoder"})
    
    max_tp_deg: int = field(default=1, metadata={"help": "The maximum tensor parallel degree."})

    def initialize(self, args_dict:dict):
        self.profile_type = args_dict.pop('--profile_type', 'memory')
        self.profile_mode = args_dict.pop('--profile_mode', 'static')
        self.profile_fixed_batch_size = int(args_dict.pop('--profile_fixed_batch_size', 1))
        self.profile_min_batch_size = int(args_dict.pop('--profile_min_batch_size', 1))
        self.profile_max_batch_size = int(args_dict.pop('--profile_max_batch_size', 12))
        self.profile_batch_size_step = int(args_dict.pop('--profile_batch_size_step', 1))
        self.layernum_min = int(args_dict.pop('--layernum_min', 2))
        self.layernum_max = int(args_dict.pop('--layernum_max', 4))
        self.profile_fixed_seq_length_list = args_dict.pop('--profile_fixed_seq_length_list', '1024,2048')
        self.num_layertype = int(args_dict.pop('--num_layertype', 1))
        self.max_tp_deg = int(args_dict.pop('--max_tp_deg', 1))

def get_current_all_args():
    args_dict = {}
    i = 0
    argv = sys.argv
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('-'):
            if i + 1 < len(argv) and not argv[i + 1].startswith('-'):
                args_dict[arg] = argv[i + 1]
                i += 2  
            else:
                args_dict[arg] = True
                i += 1
        else:
            i += 1
    return args_dict

class ModelProfiler:
    def __init__(self, args:ModelProfilerArguments, args_dict:dict):
        self.args = args
        self.args_dict = args_dict
        self.set_bsz_list()
        self.set_layernum_lists()
        self.set_seqlen_list()

    def launch_profiling(self):
        args = self.args
        if args.profile_type == 'memory':
            self.launch_memory_profiling_scripts()
        elif args.profile_type == 'computation':
            self.launch_time_profiling_scripts()
        else:
            raise ValueError(f"Unsupported profile type: {args.profile_type}. Supported types are 'memory' and 'computation'.")
    
    def process_data(self):
        args = self.args
        if args.profile_type == 'memory':
            self._process_memory_data()
        elif args.profile_type == 'computation':
            self._process_computation_data()
        else:
            raise ValueError(f"Unsupported profile type: {args.profile_type}. Supported types are 'memory' and 'computation'.")
    
    # =================Time Profiling================
    def launch_time_profiling_scripts(self):  # 先只支持decoder-only
        CMD_LIST = []
        ARGS = copy.deepcopy(self.args_dict)
        for layernum_list in self.layernum_lists:  # self.layernum_lists = [[2, 2], [4, 2], [2, 4]] or [[2], [4]]
            for bsz in self.batch_size_list: # self.batch_size_list = [1, 2, 3, 4] or [4]
                for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]                    
                    ARGS['--profile_time_flag'] = 1
                    ARGS['--profile_forward_only'] = 1
                    
                    ARGS['--to_static'] = 0  # use dynamic graph
                    ARGS['--sharding_parallel_degree'] = 1
                    ARGS['--sharding'] = "stage2"  # when sharding_parallel_degree == 1, sharding set any value is ok
                    ARGS['--tensor_parallel_degree'] = 1
                    ARGS['--pipeline_parallel_degree'] = 1
                    
                    ARGS['--num_hidden_layers'] = layernum_list[0]  # 只支持decoder-only
                    ARGS['--seq_length'] = seq_tuple[0]
                    
                    ARGS['--per_device_train_batch_size'] = bsz // ARGS['--sharding_parallel_degree']
                    ARGS['--gradient_accumulation_steps'] = 1
                    
                    LAUNCHER = os.getenv('LAUNCHER')
                    
                    CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                    CMD_LIST.append(CMD)

        print(f'[auto-parallel] All commands have been generated')
        for CMD in CMD_LIST:
            print(CMD)
                        
        print(f'[auto-parallel] Please run the following commands to get the time profiling data:')
        for CMD in CMD_LIST:
            print("[auto-parallel] run command: ", CMD)
            os.system(CMD)               
                    
    def _process_computation_data(self) -> None:
        time_config_path = self.get_time_profiling_path()
        config = read_json_config(time_config_path)
        
        for bsz in self.batch_size_list: # self.batch_size_list = [1, 2, 3, 4] or [4]
            for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]
                key_base = f'layernum[{self.args.layernum_min}]_bsz{bsz}_seq{seq_tuple[0]}'
                val_base = config[key_base]
                
                key = f'layernum[{self.args.layernum_max}]_bsz{bsz}_seq{seq_tuple[0]}'
                val = config[key]
                
                avg_time = (val - val_base) / bsz / (self.args.layernum_max - self.args.layernum_min)
                write_key = f"layertype_{0}_bsz{bsz}_seq{seq_tuple[0]}"
                config[write_key] = avg_time
                
                other_time = val_base
                other_time -= avg_time * bsz * self.args.layernum_min
                other_time /= bsz
                write_key = f"layertype_other_bsz{bsz}_seq{seq_tuple[0]}"
                config[write_key] = max(other_time, 0)
                
        write_json_config(time_config_path, config)
        print(f"Already written processed computation time into env config file {time_config_path}!\n")
   
    # =================Memory Profiling================
    def launch_memory_profiling_scripts(self):
        args = self.args
        assert args.profile_mode == 'static' or args.profile_mode == "sequence", 'memory profile support static and sequence mode'
        
        world_size = int(os.getenv('WORLD_SIZE'))
        max_tp_deg = min(world_size, args.max_tp_deg)
        if args.profile_mode != 'static':
            max_tp_deg = 1
        
        CMD_LIST = []
        ARGS = copy.deepcopy(self.args_dict)
        for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]
            pp_deg = 1
            for checkpoint in [0, 1]:
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for layernum_list in self.layernum_lists:
                            ARGS['--profile_memory_flag'] = 1
                            
                            ARGS['--to_static'] = 0  # use dynamic graph
                            ARGS['--pipeline_parallel_degree'] = pp_deg
                            ARGS['--tensor_parallel_degree'] = tp_deg
                            ARGS['--sharding_parallel_degree'] = world_size // pp_deg // tp_deg
                            ARGS['--sharding'] = "stage3"
                            
                            ARGS['--recompute'] = checkpoint
                            
                            ARGS['--num_hidden_layers'] = layernum_list[0]  # 只支持decoder-only
                            ARGS['--seq_length'] = seq_tuple[0]
                            
                            ARGS['--per_device_train_batch_size'] = args.profile_fixed_batch_size // ARGS['--sharding_parallel_degree']
                            ARGS['--gradient_accumulation_steps'] = 1
                            
                            LAUNCHER = os.getenv('LAUNCHER')
                    
                            CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                            CMD_LIST.append(CMD)
                            
                    if checkpoint:
                        break
                    tp_deg *= 2
            
            for pp_deg in [2, 4]:
                layer_num = pp_deg
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        ARGS['--profile_memory_flag'] = 1

                        ARGS['--to_static'] = 0  # use dynamic graph
                        ARGS['--pipeline_parallel_degree'] = pp_deg
                        ARGS['--tensor_parallel_degree'] = tp_deg
                        ARGS['--sharding_parallel_degree'] = world_size // pp_deg // tp_deg
                        ARGS['--sharding'] = "stage3"
                        
                        ARGS['--recompute'] = 0
                        
                        ARGS['--num_hidden_layers'] = layer_num  # 只支持decoder-only
                        ARGS['--seq_length'] = seq_tuple[0]
                        
                        ARGS['--per_device_train_batch_size'] = args.profile_fixed_batch_size // ARGS['--sharding_parallel_degree']
                        ARGS['--gradient_accumulation_steps'] = 1
                        
                        LAUNCHER = os.getenv('LAUNCHER')
                    
                        CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                        CMD_LIST.append(CMD)
                    tp_deg *= 2
                    
        print(f'[auto-parallel] All commands have been generated')
        for CMD in CMD_LIST:
            print(CMD)
        
        print(f'[auto-parallel] Please run the following commands to get the memory profiling data:')
        for CMD in CMD_LIST:
            print("[auto-parallel] run command: ", CMD)
            os.system(CMD)
    
    def _process_memory_data(self):
        pass
    
    # =================Utils Functions================
    def set_bsz_list(self):
        args = self.args
        if args.profile_mode == 'static':
            assert args.profile_fixed_batch_size is not None
            self.batch_size_list = [args.profile_fixed_batch_size]
        elif args.profile_mode == 'batch':
            assert args.profile_min_batch_size is not None and args.profile_max_batch_size is not None and args.profile_batch_size_step is not None, 'please set the min batch size, max batch size and batch size step'
            self.batch_size_list = list(range(args.profile_min_batch_size, args.profile_max_batch_size + 1, args.profile_batch_size_step))
        elif args.profile_mode == 'sequence':
            self.batch_size_list = [args.profile_fixed_batch_size]  # [NOTE] 此处查看一下到底是什么？

        print(f'[auto-parallel] batch size list: {self.batch_size_list}')
    
    def set_layernum_lists(self):
        args = self.args
        self.layernum_lists = []
        
        base_list = [args.layernum_min] * args.num_layertype  # 注意需要设置这个值
        self.layernum_lists.append(base_list)
        
        for idx in range(args.num_layertype):
            lst = base_list.copy()
            lst[idx] = args.layernum_max
            self.layernum_lists.append(lst)
        
        print(f'[auto-parallel] layernum_lists: {self.layernum_lists}')        
    
    def set_seqlen_list(self): # [NOTE] 暂时未适配swin模型
        self.sequence_length_list = []
        
        args = self.args
        if args.profile_mode == 'static' or args.profile_mode == 'batch':
            assert args.profile_fixed_seq_length_list is not None, 'please set the seq length list'
            profile_seq_length_list = list(map(int, args.profile_fixed_seq_length_list.split(',')))
            for i in range(args.num_layertype):
                self.sequence_length_list.append([profile_seq_length_list[i]])
        elif args.profile_mode == 'sequence':
            assert args.num_layertype == 1, 'sequence length profiling only support one layer type'
            assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None and args.profile_seq_length_step is not None, 'please set the min seq length, max seq length and seq length step'
            
            if args.profile_type == 'memory':
                pass # TODO
            elif args.profile_type == 'computation':
                for i in range(args.num_layertype):
                    self.sequence_length_list.append(list(range(args.profile_min_seq_length, args.profile_max_seq_length + 1, args.profile_seq_length_step)))

        self.product_sequence_length_list = list(product(*self.sequence_length_list))
        
        print(f'[auto-parallel] sequence length list: {self.sequence_length_list}')            
        print(f'[auto-parallel] product sequence length list: {self.product_sequence_length_list}')

    def get_memory_profiling_path(self):
        path = os.getcwd()
        model_name = self.args_dict['--model_name_or_path']
        mixed_precision = 'bf16' if self.args_dict.get('--bf16') else 'fp16' if self.args_dict.get('--fp16') else 'fp32'
        first_rank_file_name = f'configs/memory_profiling_{mixed_precision}_{model_name}_first.json'
        last_rank_file_name = f'configs/memory_profiling_{mixed_precision}_{model_name}_last.json'
        first_rank_path = os.path.join(path, first_rank_file_name)
        last_rank_path = os.path.join(path, last_rank_file_name)
        return first_rank_path, last_rank_path
    
    def get_time_profiling_path(self):
        path = os.getcwd()
        model_name = self.args_dict['--model_name_or_path']
        mixed_precision = 'bf16' if self.args_dict.get('--bf16') else 'fp16' if self.args_dict.get('--fp16') else 'fp32'
        time_file_name = f'configs/computation_profiling_{mixed_precision}_{model_name}_rank[0].json'  # when time profiling, only one gpu is used
        time_path = os.path.join(path, time_file_name)
        return time_path
        
        