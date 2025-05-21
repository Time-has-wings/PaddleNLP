from ..utils import Strategy
from dataclasses import dataclass, field

@dataclass
class SearchEngineArguments:
    search_granularity: str = field(default="coarse-grained", metadata={"help": "The granularity of the search space."})
    world_size: int = field(default=8, metadata={"help": "The number of processes to use for distributed training."})

class SearchEngine:
    def __init__(self, args:SearchEngineArguments):
        self.args = args
        self.search_granularity = "coarse-grained" # fine-grained
        self.generate_strategies()

    def generate_strategies(self):
        args = self.args
        
        self.strategt_set = []
        
        i, degree_set = 1, []
        while i <= args.world_size:
            degree_set.append(i)
            i *= 2
        
        for pp_size in degree_set:
            for tp_size in degree_set:
                if pp_size * tp_size > args.world_size:
                    continue
                for dp_size in degree_set:
                    if pp_size * tp_size * dp_size > args.world_size:
                        continue
                    for recompute in [0, 1]:
                        sharding_stage_set = [0, 2, 3] if dp_size > 1 else [0]
                        for sharding_stage in sharding_stage_set:
                            strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=recompute)
                            self.strategt_set.append(strategy)