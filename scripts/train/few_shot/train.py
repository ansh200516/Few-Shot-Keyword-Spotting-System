import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils
from datetime import datetime, timedelta
import time


def main(opt):
    # Add a timestamp for unique experiment directories
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Updated format for safe filenames
    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'], timestamp)
    os.makedirs(opt['log.exp_dir'], exist_ok=True)

    # Save options for reproducibility
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)

    # Ensure CUDA availability and proper setup
    opt['data.cuda'] = opt.get('data.cuda', False) and torch.cuda.is_available()
    if opt['data.cuda']:
        print("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(1234)
    else:
        print("CUDA not available or disabled. Using CPU.")

    # Load data
    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    # Load model
    model = model_utils.load(opt)

    # Ensure model uses the appropriate device
    if opt['data.cuda']:
        model.cuda()
    else:
        print("CUDA not available, using CPU.")

    engine = Engine()

    # Initialize performance meters
    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }
    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }

    # Hooks
    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        # Evaluate on validation set if available
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

            model_utils.evaluate(state['model'], val_loader, meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        # Log results
        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        # Save model if it's the best so far
        model_path = os.path.join(opt['log.exp_dir'], 'best_model.pt')
        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> Best model found (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))
                state['model'].cpu()
                torch.save(state['model'], model_path)
                if opt['data.cuda']:
                    state['model'].cuda()
                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1
                if hook_state['wait'] > opt['train.patience']:
                    print("==> Patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            print("Saving model (no validation)...")
            state['model'].cpu()
            torch.save(state['model'], model_path)
            if opt['data.cuda']:
                state['model'].cuda()

        print(f"Model saved to {model_path}")

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    # Start training
    start_time = time.time()
    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opt['train.optim_method']),
        optim_config={
            'lr': opt['train.learning_rate'],
            'weight_decay': opt['train.weight_decay']
        },
        max_epoch=opt['train.epochs']
    )
    elapsed_time = str(timedelta(seconds=(time.time() - start_time)))
    print(f"Total Training Time: {elapsed_time}")
