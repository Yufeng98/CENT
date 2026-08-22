#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details) # Set up batch job settings
#SBATCH --job-name=cuda_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24
#SBATCH --time=8:05:00
#SBATCH --account=kdur
#SBATCH --partition=project_l
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1
#SBATCH --output=/home/yufenggu/LLM-Orchestra/gpu/slurm_logs/%x-%j.log
# Run your program
# (">" redirects the print output of your program,
# in this case to "output.txt")
export TOKENIZERS_PARALLELISM=False
np=1
# model=Llama-3.1-8B




# for model in DeepSeek-V2-Lite
# do
# for prefill in 512
# do
# for decoding in 3584
# do
# for batch in 512
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done


for model in Llama-3.1-8B #Llama-2-7B
do
for prefill in 512
do
for decoding in 3584
do
for batch in 256
do
mkdir -p outputs/${model}-power
mkdir -p errors/${model}-power
attention=paged_attention

# Measure power for all 8 GPUs and log with date and time
rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
for i in 0; do
    nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
done

VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# Kill all background processes (power measurement)
pkill -P $$

done
done
done
done

# for model in Llama-3.1-8B
# do
# for prefill in 128
# do
# for decoding in 512
# do
# for batch in 512 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done






# for model in Llama-3.1-8B
# do
# for prefill in 128
# do
# for decoding in 1024
# do
# for batch in 256 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done








# for model in Llama-3.1-8B
# do
# for prefill in 2048
# do
# for decoding in 512
# do
# for batch in 128 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done









# for model in Llama-3.1-8B
# do
# for prefill in 4096
# do
# for decoding in 512
# do
# for batch in 64 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done









# for model in Llama-3.1-8B
# do
# for prefill in 8192
# do
# for decoding in 512
# do
# for batch in 32 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done






# for model in Llama-3.1-8B
# do
# for prefill in 16384
# do
# for decoding in 1024
# do
# for batch in 16 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done





# for model in Llama-3.1-8B
# do
# for prefill in 32768
# do
# for decoding in 512
# do
# for batch in 8 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done









# for model in Llama-3.1-8B
# do
# for prefill in 65536
# do
# for decoding in 512
# do
# for batch in 4 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done







# for model in Llama-3.1-8B
# do
# for prefill in 131072
# do
# for decoding in 1024
# do
# for batch in 2 #8
# do
# mkdir -p outputs/${model}-power
# mkdir -p errors/${model}-power
# attention=paged_attention

# # Measure power for all 8 GPUs and log with date and time
# rm power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# for i in 0; do
#     nvidia-smi -i $i --query-gpu=power.draw,clocks.current.sm --format=csv --loop-ms=500 | while IFS= read -r line; do echo "$(date +"%Y-%m-%d %H:%M:%S") GPU$i $line"; done | tee -a power_logs/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt &
# done

# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-power/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt

# # Kill all background processes (power measurement)
# pkill -P $$

# done
# done
# done
# done












