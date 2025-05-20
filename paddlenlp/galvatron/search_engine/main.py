from ..utils import Strategy

if __name__ == '__main__':
    strategy = Strategy()
    strategy.pp_size = 2
    strategy.tp_size = 4
    strategy.dp_size = 8
    strategy.sharding_stage = 1
    strategy.recompute = 0

    serialized_text = strategy.serialize()
    print(f"Serialized text: {serialized_text}")
    print(strategy)

    new_strategy = Strategy()
    new_strategy.deserialize(serialized_text)
    print(f"Deserialized strategy: pp_size={new_strategy.pp_size}, tp_size={new_strategy.tp_size}, dp_size={new_strategy.dp_size}, sharding_stage={new_strategy.sharding_stage}, recompute={new_strategy.recompute}")