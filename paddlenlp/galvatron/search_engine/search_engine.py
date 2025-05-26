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
                dp_size = args.world_size // (pp_size * tp_size)
                sharding_stage_set = [0, 2, 3] if dp_size > 1 else [0]
                for recompute in [0, 1]:
                    for sharding_stage in sharding_stage_set:
                        strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=recompute)
                        self.strategt_set.append(strategy)
                            
        print(f'SearchEngine strategt_set: {self.strategt_set}')
        
    def parallelism_optimization(self):
        results = dict()

        for bsz in results:
            for accumulation_steps in results[bsz]:
                # 在粗粒度的时候 这个部分应该是可以重新写的
                for min_tp in results[bsz][accumulation_steps]:
                    for max_tp in results[bsz][accumulation_steps][min_tp]:
                        for min_pp in results[bsz][accumulation_steps][min_tp][max_tp]:
                            for max_pp in results[bsz][accumulation_steps][min_tp][max_tp][min_pp]:
                                for recompute in results[bsz][accumulation_steps][min_tp][max_tp][min_pp][max_pp]:
                                    for sharding_stage in results[bsz][accumulation_steps][min_tp][max_tp][min_pp][max_pp][recompute]:
                                        pass
        