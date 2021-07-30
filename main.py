import argparse
import logging
import os
import importlib
from pathlib import Path
import run


def get_parser(**parser_kwargs):
    def mode(v):
        if v != 'train' and v != 'sample':
            raise argparse.ArgumentTypeError('train or sample expected')
        else:
            return v

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        'workdir',
        help='Work folder',
    )
    parser.add_argument(
        'config',
        help='The config file'
    )
    parser.add_argument(
        'mode',
        type=mode,
        help='train or sample'
    )
    parser.add_argument(
        '--sample_mode',
        default='',
        help='cond or uncond'
    )

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    #Create workdir
    Path(os.path.join('output', args.workdir)).mkdir(parents=True, exist_ok=True)

    #Initialize logging
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    logger = logging.getLogger()

    gfile_stream = open(os.path.join(os.path.join('output', args.workdir, 'stdout.txt')), 'w')
    file_handler = logging.StreamHandler(gfile_stream)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel('INFO')

    #Load config
    config = importlib.import_module('configs.ve.' + args.config).get_config()

    if args.mode == 'train':
        run.train(config, os.path.join('output', args.workdir))
    if args.mode == 'sample':
        run.image_synthesis(config, os.path.join('output', args.workdir), args.sample_mode)
