from itertools import product
from .base_profiler import BaseProfiler, ProfileArguments
import sys
import copy
import os
from ..utils import read_json_config, write_json_config

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

class ModelProfiler(BaseProfiler):
    def __init__(self, args:ProfileArguments, args_dict:dict):
        super().__init__(args)
        self.args_dict = args_dict
        self.set_bsz_list()
        self.set_layernum_lists()
        self.set_seqlen_list()
        
    # =================Time Profiling================
    def launch_time_profiling_scripts(self):  # 先只支持decoder-only
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
                    print(f'[auto-parallel] CMD: {CMD}')  
                    os.system(CMD)                  
                    
    def _process_computation_data(self) -> None:
        time_config_path = self.get_time_profiling_path()
        config = read_json_config(time_config_path)
        
        for bsz in self.batch_size_list: # self.batch_size_list = [1, 2, 3, 4] or [4]
            for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]
                key_base = f'layernum{self.layernum_lists[0][0]}_bsz{bsz}_seq{seq_tuple[0]}'
                key_base = f'layernum{self.args.layernum_min}_bsz{bsz}_seq{seq_tuple[0]}'
                val_base = config[key_base]
                
                key = f'layernum{self.args.layernum_max}_bsz{bsz}_seq{seq_tuple[0]}'
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
                            
                            ARGS['--per_device_train_batch_size'] = args.profile_global_batch_size // ARGS['--sharding_parallel_degree']
                            ARGS['--gradient_accumulation_steps'] = 1
                            
                            LAUNCHER = os.getenv('LAUNCHER')
                    
                            CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                            print(f'[auto-parallel] CMD: {CMD}')  
                            # os.system(CMD)   
                            
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
                        
                        ARGS['--per_device_train_batch_size'] = args.profile_global_batch_size // ARGS['--sharding_parallel_degree']
                        ARGS['--gradient_accumulation_steps'] = 1
                        
                        LAUNCHER = os.getenv('LAUNCHER')
                    
                        CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                        print(f'[auto-parallel] CMD: {CMD}') 
                        # os.system(CMD)  
                    tp_deg *= 2
    
    # =================Utils Functions================
    def set_bsz_list(self):
        args = self.args
        if args.profile_mode == 'static':
            assert args.profile_global_batch_size is not None
            self.batch_size_list = [args.profile_global_batch_size]
        elif args.profile_mode == 'batch':
            assert args.profile_min_batch_size is not None and args.profile_max_batch_size is not None and args.profile_batch_size_step is not None, 'please set the min batch size, max batch size and batch size step'
            self.batch_size_list = list(range(args.profile_min_batch_size, args.profile_max_batch_size + 1, args.profile_batch_size_step))
        elif args.profile_mode == 'sequence':
            self.batch_size_list = [args.profile_global_batch_size]  # [NOTE] 此处查看一下到底是什么？

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
            assert args.profile_seq_length_list is not None, 'please set the seq length list'
            profile_seq_length_list = list(map(int, args.profile_seq_length_list.split(',')))
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
        
        