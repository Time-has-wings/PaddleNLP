from paddlenlp.galvatron.utils import get_current_all_args
from paddlenlp.galvatron.search_engine.search_engine import SearchEngine

if __name__ == "__main__":
    args_dict = get_current_all_args()
    search_engine = SearchEngine(args_dict)
    results, optimal_solution = search_engine.parallelism_optimization()