import os
import argparse
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model", required=True)
parser.add_argument("--num_gpus", type=int, help="number of GPUs used", default=4)
parser.add_argument("--log_path", type=str, help="path to log file", required=True)
parser.add_argument("--results_path", type=str, help="path to results file", default="DGX_H100_Power.csv")
parser.add_argument("--group_size", type=int, help="group size", default=128)
parser.add_argument("--date", type=str, help="date", required=True)
parser.add_argument("--start_time", type=str, help="start time", required=True)
parser.add_argument("--end_time", type=str, help="end time", required=True)
args = parser.parse_args()

start_time = datetime.strptime(args.date + " " + args.start_time, "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime(args.date + " " + args.end_time, "%Y-%m-%d %H:%M:%S")

gpu_power = {}
for i in range(args.num_gpus):
    gpu_power[i] = []

with open(args.log_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        time_str = lst[0] + " " + lst[1]
        time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        if time < start_time or time > end_time or "[W]" in line:
            continue
        else:
            gpu_id = int(lst[2][-1])
            gpu_power[gpu_id].append(float(lst[3]))

grouped_averages = {}
for i in range(args.num_gpus):
    power_data = gpu_power[i]
    group_size = len(power_data) // args.group_size
    grouped_averages[i] = [
        sum(power_data[j:j + group_size]) / group_size
        for j in range(0, len(power_data), group_size)
    ]

overall_average = []
for i in range(args.group_size):
    overall_average.append(sum([grouped_averages[j][i] for j in range(args.num_gpus)]) / args.num_gpus)

print(sum(overall_average)/len(overall_average))

power_path = args.results_path

columns = ["model", "num_gpu"] + [f"{i*1024}" for i in range(args.group_size)]
if os.path.exists(power_path):
    df = pd.read_csv(power_path)
else:
    df = pd.DataFrame(columns=columns)

new_row = [args.model, args.num_gpus] + overall_average
new_df = pd.DataFrame([new_row], columns=columns)
df = pd.concat([df, new_df], ignore_index=True)

df.to_csv(power_path, index=False)