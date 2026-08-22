#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details) # Set up batch job settings
#SBATCH --job-name=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=lh1800
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=22g
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

# rm ncu_profile.ncu-rep; ncu -o ncu_profile --page=details --target-processes all --metrics sm__inst_executed,sm__inst_executed_pipe_alu,sm__inst_executed_pipe_fma,sm__inst_executed_pipe_fp16,sm__inst_executed_pipe_fp64,sm__inst_executed_pipe_ipa,sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_dmma,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_hmma_type_hfma2,sm__inst_executed_pipe_tensor_op_imma,sm__inst_executed_pipe_tex,sm__inst_executed_pipe_uniform,sm__inst_executed_pipe_xu,sm__inst_issued --replay-mode app-range torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir /dev/shm/7b_1/ --tokenizer_path tokenizer.model --max_seq_len 4096 --max_gen_len 16 --max_batch_size 4 2>&1 | tee results/7b/batch_4/prof_2.txt


# for model in Llama-3.1-8B
# do
# for prefill in 16
# do
# for decoding in 8
# do
# for batch in 1
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# rm -rf ncu_profile-1gpu.ncu-rep; VLLM_SKIP_P2P_CHECK=1 NCCL_NVLS_ENABLE=0 ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np > outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 128
# do
# for decoding in 32
# do
# for batch in 2048
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 256
# do
# for decoding in 32
# do
# for batch in 1024
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 512
# do
# for decoding in 32
# do
# for batch in 512
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done

# for model in Llama-3.1-8B
# do
# for prefill in 1024
# do
# for decoding in 32
# do
# for batch in 256
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 2048
# do
# for decoding in 32
# do
# for batch in 128
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 4096
# do
# for decoding in 32
# do
# for batch in 64
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 8192
# do
# for decoding in 32
# do
# for batch in 32
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 16384
# do
# for decoding in 32
# do
# for batch in 16
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# for model in Llama-3.1-8B
# do
# for prefill in 32768
# do
# for decoding in 32
# do
# for batch in 8
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# for model in Llama-3.1-8B
# do
# for prefill in 65536
# do
# for decoding in 32
# do
# for batch in 4
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 131000
# do
# for decoding in 32
# do
# for batch in 2
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes all \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done




# mkdir -p profile/Llama-3.1-8B-1gpu-all-processes
# mv ncu_profile-1gpu-batch-* profile/Llama-3.1-8B-1gpu-all-processes/





# for model in Llama-3.1-8B
# do
# for prefill in 128
# do
# for decoding in 32
# do
# for batch in 2048
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 256
# do
# for decoding in 32
# do
# for batch in 1024
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 512
# do
# for decoding in 32
# do
# for batch in 512
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 1024
# do
# for decoding in 32
# do
# for batch in 256
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 2048
# do
# for decoding in 32
# do
# for batch in 128
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 4096
# do
# for decoding in 32
# do
# for batch in 64
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 8192
# do
# for decoding in 32
# do
# for batch in 32
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 16384
# do
# for decoding in 32
# do
# for batch in 16
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# for model in Llama-3.1-8B
# do
# for prefill in 32768
# do
# for decoding in 32
# do
# for batch in 8
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# for model in Llama-3.1-8B
# do
# for prefill in 65536
# do
# for decoding in 32
# do
# for batch in 4
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done


# for model in Llama-3.1-8B
# do
# for prefill in 131000
# do
# for decoding in 32
# do
# for batch in 2
# do
# attention=paged_attention
# mkdir -p outputs/${model}-profile
# mkdir -p errors/${model}-profile
# mkdir -p profile/${model}
# export VLLM_SKIP_P2P_CHECK=1
# export NCCL_NVLS_ENABLE=0
# METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
# rm ncu_profile-1gpu.ncu-rep
# ncu -o ncu_profile-1gpu.ncu-rep \
#     --metrics $METRICS \
#     --target-processes application-only \
#     --replay-mode app-range \
#     --csv \
#     python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

# ncu --page raw \
#     --import ncu_profile-1gpu.ncu-rep \
#     --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# # ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
# done
# done
# done
# done



# mkdir -p profile/Llama-3.1-8B-1gpu-application-only-dram
# mv ncu_profile-1gpu-batch-* profile/Llama-3.1-8B-1gpu-application-only-dram




for model in Llama-3.1-8B
do
for prefill in 512
do
for decoding in 3584
do
for batch in 512
do
attention=paged_attention
mkdir -p outputs/${model}-profile
mkdir -p errors/${model}-profile
mkdir -p profile/${model}
export VLLM_SKIP_P2P_CHECK=1
export NCCL_NVLS_ENABLE=0
METRICS=dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
rm ncu_profile-1gpu.ncu-rep
ncu -o ncu_profile-1gpu.ncu-rep \
    --metrics $METRICS \
    --target-processes application-only \
    --replay-mode app-range \
    --csv \
    python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np

ncu --page raw \
    --import ncu_profile-1gpu.ncu-rep \
    --csv &> ncu_profile-1gpu-batch-${batch}-prefill-${prefill}-decoding-${decoding}.csv
# ncu -o ncu_profile-1gpu --page=details --target-processes application-only --metrics dram__bytes,dram__bytes_read,dram__bytes_write,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --log-file outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.log --replay-mode app-range python run.py --model_dir /home/yufenggu/data/$model --attention_type $attention --prompt_token $prefill --generate_token $decoding --batch_size $batch --np $np #> outputs/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt 2> errors/${model}-profile/${model}_${attention}_${np}_gpu_${batch}_batch_${prefill}_prefill_${decoding}_decoding.txt
done
done
done
done


mkdir -p profile/Llama-3.1-8B-1gpu-application-only-dram
mv ncu_profile-1gpu-batch-* profile/Llama-3.1-8B-1gpu-application-only-dram


# mkdir -p profile/DeepSeek-V2-Lite-1gpu-application-only-dram
# mv ncu_profile-1gpu-batch-* profile/DeepSeek-V2-Lite-1gpu-application-only-dram
