import os
import sys
import glob
import math
import pickle
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import gpt
import torch

# Precompute parameter counts for GPT configurations to be instant
_param_cache = {}
def get_gpt_params(n_embd, n_layer=12, block_size=100):
    key = (n_embd, n_layer, block_size)
    if key in _param_cache:
        return _param_cache[key]
    config = gpt.GPTConfig(
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=1,
        bias=True,
        is_initialize_attention_weights_to_zero=False
    )
    m = gpt.GPT(config, torch.nn.NLLLoss(reduction='none'))
    count = sum(p.numel() for p in m.parameters())
    _param_cache[key] = count
    return count


class FastModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: None
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            def dummy_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
                class DummyTensor:
                    def numel(self):
                        return math.prod(size)
                return DummyTensor()
            return dummy_tensor
        return super().find_class(module, name)


def process_12layer_folder(folder):
    pkls = [p for p in glob.glob(f'{folder}/*.pkl') if not p.endswith('_all_sequences.pkl')]
    if not pkls:
        return None
    fname = pkls[0]
    try:
        with open(fname, 'rb') as f:
            data = FastModelUnpickler(f).load()
    except Exception:
        return None

    if 'logs' not in data or len(data['logs']) < 60:
        return None

    args = data.get('args', {})
    num_hidden = args.get('num_hidden_features', 1)
    num_heads = args.get('num_heads', 1)
    D = num_hidden * num_heads
    n_layer = args.get('num_layers', 12)
    block_size = args.get('len_context', 100)

    try:
        num_parameters = sum(p.numel() for k, p in data['model'].items())
    except Exception:
        num_parameters = 0

    if num_parameters == 0:
        num_parameters = get_gpt_params(D, n_layer=n_layer, block_size=block_size)

    train_losses = [l['train_loss'] for l in data['logs']]
    final_loss = train_losses[-1]

    # Exact filter specified by user
    if num_parameters > 3e6 and final_loss > 0.21:
        failed = True
    elif num_parameters > 1e6 and final_loss > 0.4:
        failed = True
    else:
        failed = False

    # Capacity data at epochs 60, 120, 190 (60k, 120k, 190k iters)
    cap_records = []
    switch_records = []
    
    for l in data['logs']:
        ep = l.get('epoch')
        if ep in [60, 120, 190]:
            tm = l.get('test_metrics', {})
            if 'accuracy' in tm and 'sequence_rank' in tm:
                acc = np.asarray(tm['accuracy'])
                ranks = np.asarray(tm['sequence_rank']) * 10
                met = ranks[acc > 0.90]
                max_rank = int(met.max()) if len(met) > 0 else 0
                if max_rank > 0:
                    cap_records.append({
                        'num_parameters': num_parameters,
                        'epoch': ep,
                        'iters': ep * 1000,
                        'max_rank': max_rank,
                        'failed': failed,
                    })

        # Switch accuracy over first 3000 ranks for broken axis:
        if (ep is not None) and ((0 <= ep <= 10) or (40 <= ep <= 50)):
            tm = l.get('test_metrics', {})
            if 'accuracy' in tm and 'sequence_rank' in tm:
                acc = np.asarray(tm['accuracy'])
                ranks = np.asarray(tm['sequence_rank']) * 10
                mask_3k = ranks <= 3000
                if np.any(mask_3k):
                    avg_acc_3k = float(acc[mask_3k].mean())
                    switch_records.append({
                        'num_parameters': num_parameters,
                        'epoch': ep,
                        'iteration': ep * 1000,
                        'accuracy': avg_acc_3k,
                        'failed': failed,
                    })

    return {
        'loss_record': {
            'folder': os.path.basename(folder),
            'num_parameters': num_parameters,
            'num_heads': num_heads,
            'd_raw': num_hidden,
            'num_hidden_features': D,
            'train_loss': final_loss,
            'epoch': len(train_losses) - 1,
            'failed': failed,
        },
        'cap_records': cap_records,
        'switch_records': switch_records,
    }

def main():
    os.makedirs('./cache', exist_ok=True)
    gpt2_cache = '/scratch/qanguyen/gautam/cache/memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample/*'
    folders = glob.glob(gpt2_cache)
    print(f'Starting extraction of {len(folders)} 12-layer folders using 20 parallel workers...')
    t0 = time.time()

    with Pool(20) as pool:
        results = pool.map(process_12layer_folder, folders)

    results = [r for r in results if r is not None]
    print(f'Processed in {time.time() - t0:.2f}s! Valid runs: {len(results)}')

    # 1. Train loss dataset
    loss_list = [r['loss_record'] for r in results]
    df_loss = pd.DataFrame(loss_list)
    loss_csv = './cache/df_train_loss_12layer.csv'
    df_loss.to_csv(loss_csv, index=False)
    print(f'Saved train loss: {loss_csv} (Total: {len(df_loss)}, Failed: {df_loss["failed"].sum()}, OK: {(~df_loss["failed"]).sum()})')

    # 2. Capacity dataset
    cap_list = []
    for r in results:
        cap_list.extend(r['cap_records'])
    df_cap = pd.DataFrame(cap_list)
    cap_csv = './cache/df_capacity_12layer.csv'
    df_cap.to_csv(cap_csv, index=False)
    print(f'Saved capacity: {cap_csv} (Total rows: {len(df_cap)})')

    # 3. Data switch dataset
    sw_list = []
    for r in results:
        sw_list.extend(r['switch_records'])
    df_sw = pd.DataFrame(sw_list)
    sw_csv = './cache/df_dataswitch_12layer.csv'
    df_sw.to_csv(sw_csv, index=False)
    print(f'Saved data switch: {sw_csv} (Total rows: {len(df_sw)})')

if __name__ == '__main__':
    main()
