import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

def main(opt):
    # Load model
    model = torch.load(opt['model.model_path'], map_location=torch.device('cuda' if torch.cuda.is_available() and opt['data.cuda'] else 'cpu'))
    model.eval()

    # Load options
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # Construct data options
    data_opt = { 'data.' + k: v for k, v in filter_opt(model_opt, 'data').items() }

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    # Seed settings
    torch.manual_seed(1234)
    data_opt['data.cuda'] = data_opt.get('data.cuda', False) and torch.cuda.is_available()  # Ensure proper CUDA handling

    if data_opt['data.cuda']:
        print("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(1234)
        model = model.cuda()
    else:
        print("CUDA not available. Using CPU.")

    # Load data
    data = data_utils.load(model_opt, ['test'])

    # Initialize meters
    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    # Evaluate the model
    model_utils.evaluate(model, data['test'], meters, desc="test")

    # Save results
    output = {"test": {}}
    for field, meter in meters.items():
        mean, std = meter.value()
        output["test"][field] = {
            "mean": mean,
            "confidence": 1.96 * std / math.sqrt(data_opt['data.test_episodes'])
        }

    output_file = os.path.join(os.path.dirname(opt['model.model_path']), 'eval.txt')
    with open(output_file, 'w') as fp:
        json.dump(output, fp)
