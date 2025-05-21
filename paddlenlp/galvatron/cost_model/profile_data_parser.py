from dataclasses import dataclass, field
from paddlenlp.galvatron.utils import read_json_config
from scipy.optimize import curve_fit

@dataclass
class ProfileDataParserArguments:
    time_profile_mode: str = field(default='static', metadata={"help": "The mode of time profiling."})
    time_profile_data_path: str = field(default=None, metadata={"help": "The path of time profiling data."})
    
    memory_profile_mode: str = field(default='static', metadata={"help": "The mode of memory profiling."})
    memory_profile_data_path: str = field(default=None, metadata={"help": "The path of memory profiling data."})

    num_layertype: int = field(default=1, metadata={"help": "1: decoder-only or encoder-only, 2: encoder-decoder"})
    hidden_size_list: str = field(default="4096", metadata={"help": "The hidden size of the model."})
    layernum_list: str = field(default="12", metadata={"help": "The number of layers of the model."})
    seqlen_list: str = field(default="1024", metadata={"help": "The sequence length of the model."})
    
class ProfileDataParser:
    """
        A class to parse the profile data from json file.
    """
    def __init__(self, args:ProfileDataParserArguments):
        self.args = args
        self.validate_args()
        
    def validate_args(self):
        args = self.args
        assert args.time_profile_mode in ['static', 'batch', 'sequence'], f'Unsupported time profile mode: {args.time_profile_mode}'
        assert args.memory_profile_mode in ['static', 'sequence'], f'Unsupported memory profile mode: {args.memory_profile_mode}'
        assert args.time_profile_data_path is not None, f'Time profile data path is None'
        assert args.memory_profile_data_path is not None, f'Memory profile data path is None'
        
        self.hidden_size_list = [int(x) for x in args.hidden_size_list.split(',')]
        self.layernum_list = [int(x) for x in args.layernum_list.split(',')]
        self.seqlen_list = [int(x) for x in args.seqlen_list.split(',')]
        assert len(self.hidden_size_list) == len(self.layernum_list) == len(self.seqlen_list) == args.num_layertype, f'The length of hidden_size_list, layernum_list and seqlen_list should be equal to num_layertype: {args.num_layertype}, but got {len(self.hidden_size_list)}, {len(self.layernum_list)}, {len(self.seqlen_list)}'
    
    def parse_profile_computation_configs(self):
        args = self.args
        self.time_profiled_list = []  # transformer layers
        self.other_time_profiled_list = []  # embedding layer, classifier layer, etc. Actuall, other_time is independent of num_layertype, but we still use a List to store it.
        self.time_config = read_json_config(args.time_profile_data_path)
        
        if args.time_profile_mode == 'static':
            for i in range(args.num_layertype):
                for key, value in self.time_config.items(): # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_{i}_'):
                        self.time_profiled_list.append(value)
                    elif key.startswith(f'layertype_other_'):
                        self.other_time_profiled_list.append(value)
        elif args.time_profile_mode == 'batch':
            # process transformer layers
            for i in range(args.num_layertype): 
                x_data, y_data = [], []  
                for key, value in self.time_config.items():  # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_{i}_') and f'_seq{self.seqlen_list[i]}' in key:
                        bsz = int(key.split('_')[-2][3:])  
                        x_data.append(bsz)
                        y_data.append(value * bsz)
                assert len(x_data) >= 8, f'Different batch size data is less than 8, please check the time profile data'
                # fit using a linear function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                self.time_profiled_list.append(popt)
                print("Fitted popt for transformer layers:", popt)
            # process other layers, like embedding layer, classifier layer, etc.
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items(): # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_other_') and f'_seq{self.seqlen_list[i]}' in key:
                        bsz = int(key.split('_')[-2][3:])  
                        x_data.append(bsz)
                        y_data.append(value * bsz)
                assert len(x_data) >= 8, f'Different batch size data is less than 8, please check the time profile data'
                # fit using a linear function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                self.other_time_profiled_list.append(popt)
                print("Fitted popt for other layers:", popt)
        elif args.time_profile_mode == 'sequence':
            # process transformer layers
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items():
                    if key.startswith(f'layertype_{i}_') and f'_bsz1_' in key:
                        x_data.append(int(key.split('seq')[-1]))
                        y_data.append(value)
                # fit using a quadratic function
                def quadratic_func(x, a, b, c):
                    return a * x**2 + b * x + c
                popt, _ = curve_fit(quadratic_func, x_data, y_data)
                self.time_profiled_list.append(popt)
                print("Fitted popt for transformer layers:", popt)
            # process other layers, like embedding layer, classifier layer, etc.
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items():
                    if key.startswith(f'layertype_other_') and f'_bsz1_' in key:
                        x_data.append(int(key.split('seq')[-1]))
                        y_data.append(value)
                # fit using a quadratic function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                self.other_time_profiled_list.append(popt)
                print("Fitted popt for other layers:", popt)
        else:
            raise ValueError(f"Unsupported time profile mode: {args.time_profile_mode}")
    
    def parse_profile_memory_configs(self):
        pass
    
    # =================Utils Functions=================
    def get_memory_profiling_path(self):
        pass
    
    def get_time_profiling_path(self):
        pass