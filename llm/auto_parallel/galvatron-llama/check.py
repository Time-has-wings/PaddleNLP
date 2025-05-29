from paddlenlp.galvatron.cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
from paddlenlp.galvatron.utils import get_current_all_args, Strategy

tasks_list = [
    "A100_2_2_2_zero2_FALSE_128_16_8_4_O1_TRUE_16_1024",
    "A100_1_1_8_zero2_FALSE_256_16_16_2_O1_TRUE_8_1024",
    "A100_1_8_1_ddp_FALSE_64_16_4_4_O1_TRUE_16_1024",
    "A100_1_8_1_ddp_FALSE_128_16_8_8_O1_TRUE_16_1024",
    "A100_8_1_1_ddp_FALSE_64_64_1_1_O1_TRUE_16_1024",
    "A100_8_1_1_ddp_FALSE_16_16_1_1_O1_TRUE_16_1024",
    "A100_2_4_1_ddp_FALSE_64_16_4_4_O1_TRUE_16_1024",
    "A100_2_4_1_ddp_FALSE_128_16_8_8_O1_TRUE_16_1024",
    "A100_4_2_1_ddp_FALSE_64_16_4_4_O1_TRUE_16_1024",
    "A100_4_2_1_ddp_FALSE_128_32_4_4_O1_TRUE_16_1024",
    "A100_2_1_4_zero2_FALSE_128_16_8_2_O1_TRUE_16_1024",
    "A100_4_1_2_zero2_FALSE_128_32_4_2_O1_TRUE_16_1024",
    "A100_1_4_2_zero2_FALSE_128_32_4_2_O1_TRUE_16_1024",
    "A100_1_4_2_zero2_FALSE_128_16_8_4_O1_TRUE_16_1024",
    "A100_1_2_4_zero2_FALSE_128_16_8_2_O1_TRUE_16_1024",
    "A100_1_2_4_zero2_FALSE_128_8_16_4_O1_TRUE_16_1024"
]

total_task_predict_time_result = []
total_task_predict_memory_result = []

def parse_task_name(task_name):
    parts = task_name.split('_')
    pp = parts[1]
    tp = parts[2]
    dp = parts[3]
    stage = 2 if parts[4] == 'zero2' else (3 if parts[4] == 'zero3' else (1 if parts[4] == 'zero1' else 0 ))
    recompute = 0 if parts[5] == 'FALSE' else 1
    gbsz = int(parts[6])
    accumulate_steps = int(parts[7])
    
    strategy_str = f'pp{pp}_tp{tp}_dp{dp}_stage{stage}_recompute{recompute}'
    
    task = {
        'strategy_str': strategy_str,
        'gbsz': gbsz,
        'accumulate_steps': accumulate_steps
    }
    return task 

def do_predict():
    for idx, task in enumerate(tasks_list):
        print("=" * 250)
        parse_result = parse_task_name(task)
        strategy = Strategy()
        strategy.deserialize(parse_result['strategy_str'])
        print(f'task {idx}: {tasks_list[idx]}')
        print(f'current strategy: {strategy}', end=" ")
        print(f'global batch size: {parse_result["gbsz"]}', end=" ")
        print(f'accumulation steps: {parse_result["accumulate_steps"]}')
        
        print(f'\nmemory cost calculating...')
        memory_cost = profile_data_parser.get_memory_cost_for_specific_strategy(strategy, parse_result['gbsz'], 'bf16', parse_result['accumulate_steps'])
        print('======== memory cost result ========')
        for stage_idx in range(strategy.pp_size):
            print(f'stage {stage_idx}: {memory_cost[stage_idx]}')
        total_task_predict_memory_result.append(memory_cost)
            
        print(f'\ntime cost calculating...')
        time_cost = profile_data_parser.get_time_cost_for_specific_strategy(strategy, parse_result['gbsz'], 'bf16', parse_result['accumulate_steps'])
        print('======== time cost result ========')
        print(f'time cost: {time_cost}')
        total_task_predict_time_result.append(time_cost)

if __name__ == "__main__":
    args_dict = get_current_all_args()
    
    profile_data_parser_args = ProfileDataParserArguments()
    profile_data_parser_args.initialize(args_dict=args_dict)
    profile_data_parser = ProfileDataParser(profile_data_parser_args)
    print('profile_data_parser constructed.')
    
    do_predict()
    
    print('======== all tasks memory cost result ========')
    for idx, task in enumerate(tasks_list):
        print(f'task {idx}: {task}, memory cost: {total_task_predict_memory_result[idx]}')
    print('======== all tasks time cost result ========')
    for idx, task in enumerate(tasks_list):
        print(f'{total_task_predict_time_result[idx]:.4f}')
        # print(f'task {idx}: {task}, time cost: {total_task_predict_time_result[idx]}')
    print('======== all tasks stage0 memory cost result ========')
    for idx, task in enumerate(tasks_list):
        result = total_task_predict_memory_result[idx][0] / 1024
        print(f'{result:.4f}')
    
    