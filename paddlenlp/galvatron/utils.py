import json
import os
from typing import List

def read_json_config(path):
    if os.path.exists(path) == False:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            fp.write("{}")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(path, config):
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

# 这个注意一下格式啊
def num2str(num, type:str):
    if type == 'seq':
        if isinstance(num, List):
            info = f'seq[{",".join(map(str, num))}]'
        else:
            info = f'seq{num}'
    elif type == 'layernum':
        info = 'layernum' + str(num)
    return info
    