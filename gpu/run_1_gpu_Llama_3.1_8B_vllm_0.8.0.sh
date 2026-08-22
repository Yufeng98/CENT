#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details) # Set up batch job settings
#SBATCH --job-name=cuda_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24g
#SBATCH --time=2:05:00
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
# model=Llama-3.2-1B
# model=Llama-3.1-8B
# model=Llama-3.1-8B
# for model in Llama-3.2-1B
# do
# for prefill in 16
# do
# for decoding in 8
# do
# for batch in 4
# do
# attention=paged_attention
# mkdir -p outputs/${model}-vllm-0.8.0
# # export LD_LIBRARY_PATH=/home/yufenggu/data/nccl/build/lib:$LD_LIBRARY_PATH
# # export NCCL_ROOT=/home/yufenggu/data/nccl/build
# VLLM_SKIP_P2P_CHECK=1 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# for model in Llama-3.1-8B
# do
# for prefill in 512
# do
# for decoding in 3584
# do
# for batch in 256 512 #128
# do
# attention=paged_attention
# mkdir -p outputs/${model}-vllm-0.8.0
# # export LD_LIBRARY_PATH=/home/yufenggu/data/nccl/build/lib:$LD_LIBRARY_PATH
# # export NCCL_ROOT=/home/yufenggu/data/nccl/build
# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done

# for model in Llama-3.1-8B
# do
# for prefill in 1
# do
# for decoding in 1024
# do
# for batch in 1024 2048 4096
# do
# mkdir -p outputs/${model}-vllm-0.8.0
# attention=paged_attention
# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
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
# for batch in 1 2
# do
# mkdir -p outputs/${model}-vllm-0.8.0
# mkdir -p errors/${model}-vllm-0.8.0
# attention=paged_attention
# VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


for model in Llama-3.1-8B
do
for prefill in 65536
do
for decoding in 1024
do
for batch in 1 2 4
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 32768
do
for decoding in 1024
do
for batch in 1 2 4 8
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 16384
do
for decoding in 1024
do
for batch in 1 2 4 8 16
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 8192
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 4096
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32 64
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 2048
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32 64 128
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 1024
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32 64 128 256
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 512
do
for decoding in 512
do
for batch in 1 2 4 8 16 32 64 128 256 512
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in 256
do
for decoding in 256
do
for batch in 1 2 4 8 16 32 64 128 256 512 1024
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done


for model in Llama-3.1-8B
do
for prefill in 128
do
for decoding in 128
do
for batch in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done












for model in Llama-3.1-8B
do
for prefill in 130046
do
for decoding in 1024
do
for batch in 1 2
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done


for model in Llama-3.1-8B
do
for prefill in $((65536-1024))
do
for decoding in 1024
do
for batch in 1 2 4
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in $((32768-1024))
do
for decoding in 1024
do
for batch in 1 2 4 8
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in $((16384-1024))
do
for decoding in 1024
do
for batch in 1 2 4 8 16
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in $((8192-1024))
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done

for model in Llama-3.1-8B
do
for prefill in $((4096-1024))
do
for decoding in 1024
do
for batch in 1 2 4 8 16 32 64
do
mkdir -p outputs/${model}-vllm-0.8.0
mkdir -p errors/${model}-vllm-0.8.0
attention=paged_attention
VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-vllm-0.8.0/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done







