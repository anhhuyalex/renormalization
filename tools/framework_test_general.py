import argparse
import os
import pdb
import importlib
import setuptools

from openselfsup.framework.epoch_based_runner import EpochBasedRunner
from openselfsup.framework.dist_utils import init_dist
from openselfsup.framework.hooks.validate_hook import ValidateHook


def get_parser():
    parser = argparse.ArgumentParser(
            description='Pytorch training framework for general dist training')
    parser.add_argument(
            '--setting', 
            default=None, type=str, 
            action='store', required=True)
    parser.add_argument(
            '--local_rank', type=int, default=0,
            help='Used during distributed training')
    return parser


def get_setting_func(setting):
    assert len(setting.split(':')) == 2, \
            'Setting should be "script_path:func_name"'
    script_path, func_name = setting.split(':')
    assert script_path.endswith('.py'), \
            'Script should end with ".py"'
    module_name = script_path[:-3].replace('/', '.')
    while module_name.startswith('.'):
        module_name = module_name[1:]
    load_setting_module = importlib.import_module(module_name)
    setting_func = getattr(load_setting_module, func_name)
    return setting_func


def main():
    parser = get_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    init_dist('pytorch')

    param_func = get_setting_func(args.setting)
    params = param_func(args)
    runner = EpochBasedRunner(**params)
    for _hook in runner._hooks:
        if isinstance(_hook, ValidateHook):
            _hook._run_validate(runner)


if __name__ == '__main__':
    main()
