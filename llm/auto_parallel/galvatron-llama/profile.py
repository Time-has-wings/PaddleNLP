from paddlenlp.galvatron.profiler.model_profiler import ModelProfiler, get_current_all_args
from paddlenlp.galvatron.profiler.base_profiler import ProfileArguments

if __name__ == '__main__':
    args_dict = get_current_all_args()
    profile_args = ProfileArguments()
    profile_args.profile_time_flag = int(args_dict.get('--profile_time_flag', 0))
    profile_args.profile_memory_flag = int(args_dict.get('--profile_memory_flag', 0))
    profile_args.profile_forward_only = int(args_dict.get('--profile_forward_only', 0))
    # profile_args.profile_seq_len = int(args_dict.get('--profile_seq_len', 1024))  # 这5个值都是在runtime的内部被定义的，所以此处不需要设置，并且runtime的脚本也没有设置这些值
    # profile_args.profile_mixed_precision = args_dict.get('--profile_mixed_precision', 'bf16')
    profile_args.profile_global_batch_size = int(args_dict.pop('--profile_global_batch_size', 1))
    # profile_args.profile_layer_num = int(args_dict.get('--profile_layer_num', 16))
    # profile_args.profile_model_name = args_dict.get('--profile_model_name', 'llama')
    profile_args.profile_type = args_dict.pop('--profile_type', 'memory')
    profile_args.profile_mode = args_dict.pop('--profile_mode', 'static')
    profile_args.profile_min_batch_size = int(args_dict.pop('--profile_min_batch_size', 1))
    profile_args.profile_max_batch_size = int(args_dict.pop('--profile_max_batch_size', 12))
    profile_args.profile_batch_size_step = int(args_dict.pop('--profile_batch_size_step', 1))
    profile_args.layernum_min = int(args_dict.pop('--layernum_min', 2))
    profile_args.layernum_max = int(args_dict.pop('--layernum_max', 4))
    profile_args.profile_seq_length_list = args_dict.pop('--profile_seq_length_list', '1024,2048')
    profile_args.num_layertype = int(args_dict.pop('--num_layertype', 1))
    profile_args.max_tp_deg = int(args_dict.pop('--max_tp_deg', 1))
        
    model_profiler = ModelProfiler(profile_args, args_dict)
    
    if profile_args.profile_type == 'memory':
        model_profiler.launch_memory_profiling_scripts()
    elif profile_args.profile_type == 'time':
        model_profiler.launch_time_profiling_scripts()
        model_profiler._process_computation_data()
    
    