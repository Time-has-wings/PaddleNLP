from dataclasses import dataclass, field
from ..utils import Strategy

@dataclass
class MemoryCostModelArguments:
    strategy: Strategy = field(default=None, metadata={"help": "The strategy of the model."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size of the model."})
    mixed_precision_type: str = field(default='fp16', metadata={"help": "The mixed precision type of the model."})
    stage_idx: int = field(default=0, metadata={"help": "The stage index of the model."})
    accumulation_steps: int = field(default=-1, metadata={"help": "The number of accumulation steps."})

    parameter_memory: float = field(default=0.0, metadata={"help": "The parameter memory of the model."})
    tp_activation_per_bsz_dict:dict = field(default_factory=lambda: {1:85, 2:47, 4:28, 8:18.5})
 
class MemoryCostModel:
    def __init__(self, args:MemoryCostModelArguments):
        self.args = args
        self.validate()
        self.initialize()
        self.estimate_parameter_size()
        
    def validate(self):
        args = self.args
        assert args.accumulation_steps > 0, f'Accumulation steps should be greater than 0, but got {args.accumulation_steps}'
    
    def initialize(self):
        args = self.args
        
        strategy = args.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.dp_size = strategy.dp_size
        self.sharding_stage = strategy.sharding_stage
        self.recompute = strategy.recompute
        
        self.local_batch_size = args.global_batch_size // self.dp_size
        # TODO 调整1f1b比例
        
        # TODO 确认一下paddle是不是异步梯度累积
        if args.accumulation_steps == 1:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003) * 5/4
        
    def estimate_parameter_size(self):
        args = self.args
        self.parameter_size = args.parameter_memory / self.tp_size
        
    def estimate_model_states_size(self):
        self.model_states_size = 4 * self.parameter_size
        if self.sharding_stage == 3:
            self.model_states_size *= self.zero3_ratio(self.dp_size)
        elif self.sharding_stage == 2:
            self.model_states_size *= self.zero2_ratio(self.dp_size)
    
    def estimate_activation_size(self):
        args = self.args
        if self.recompute:
            self.activation_size = args.tp_activation_per_bsz_dict['checkpoint'] * self.local_batch_size # TODO 修改为累积bsz
        else:
            self.activation_size = args.tp_activation_per_bsz_dict[self.tp_size] * self.local_batch_size # TODO 修改为累积bsz
        
    def get_memory_cost(self):
        result = {}
        result['parameter_size'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        return result
    
@dataclass
class OtherMemoryCostModelArguments:
    min_tp_size: int = field(default=1, metadata={"help": "The minimum tp size of the model."})
    max_tp_size: int = field(default=8, metadata={"help": "The maximum tp size of the model."})
    world_size: int = field(default=8, metadata={"help": "The world size of the model."})
    pp_size: int = field(default=1, metadata={"help": "The pp size of the model."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size of the model."})
    accumulation_steps: int = field(default=1, metadata={"help": "The number of accumulation steps."})
    paddle_context_memory: float = field(default=0.0, metadata={"help": "The paddle context memory of the model."})
    other_memory_pp_off:dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on:dict = field(default_factory=lambda: {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}})

class OtherMemoryCostModel:
    def __init__(self, args:OtherMemoryCostModelArguments):
        self.args = args
        
    def initialize(self):
        args = self.args
        # TODO 修改一下混合精度等等的逻辑
        if args.accumulation_steps == 1:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003) * 5/4
        self.zero_ratio = self.zero2_ratio if args.sharding_stage == 2 else self.zero3_ratio
    
    def estimate_memory_cost(self):
        args = self.args
        
        tp_size_list, tp_size = [], args.min_tp_size
        while tp_size <= args.max_tp_size and tp_size * args.pp_size <= args.world_size:
            tp_size_list.append(tp_size)
            tp_size *= 2
        
        self.other_memory_cost = {}
        for tp_size in tp_size_list:
            dp_size = args.world_size // args.pp_size // tp_size
            tp_other_memory_cost = [0 for _ in range(args.pp_size)]
            other_layers_bsz = args.global_batch_size // dp_size // args.accumulation_steps

            if self.pp_size == 1: # no pp -> only one stage
                tp_other_memory_cost[0] = args.other_memory_pp_off['model_states'][tp_size] * self.zero_ratio(dp_size) + args.other_memory_pp_off['activation'][tp_size] * other_layers_bsz
            else: # pp -> 0:first stage, -1:last stage
                other_layers_bsz_first = other_layers_bsz * args.pp_size
                other_layers_bsz_last = other_layers_bsz * 1
                tp_other_memory_cost[0] = args.other_memory_pp_on['first_stage']['model_states'][tp_size] * self.zero_ratio(dp_size) + args.other_memory_pp_on['first_stage']['activation'][tp_size] * other_layers_bsz_first
                tp_other_memory_cost[-1] = args.other_memory_pp_on['last_stage']['model_stage'][tp_size] * self.zero_ratio(dp_size) + args.other_memory_pp_on['last_stage']['activation'][tp_size] * other_layers_bsz_last
            
            for i in range(len(tp_other_memory_cost)):
                tp_other_memory_cost[i] += args.paddle_context_memory
            
            self.other_memory_cost[tp_size] = tp_other_memory_cost
            
    def get_other_memory_cost(self):
        return self.other_memory_cost