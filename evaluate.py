import torch
import os
from vreason_bench import VReasonBench
from datetime import datetime

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='VReasonBench')
    parser.add_argument(
        "--task",
        nargs='+',
        required=True,
        help="list of evaluation tasks, usage: --task <task_1> <task_2>",
    )
    parser.add_argument(
        "--generated_videos",
        type=str,
        required=True,
        help="path to directory containing generated videos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f'args: {args}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = VReasonBench(device)

    print('start evaluation')
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    results = bench.evaluate(
        name=f'results_{args.task}',
        task_list=args.task,
        video_dir=args.generated_videos,
    )

    print('Done!')


if __name__ == "__main__":
    main()