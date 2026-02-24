# Pipeline Parallel
threads=$1
seqlen_gap=$2

python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-13B --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Model Parallel
python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-13B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-70B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Long Context
python3 run_sim.py --model Llama2-70B --simulation_result_path simulation_results_long_context.csv --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --seqlen 2304 6400 14592 30976
python3 run_sim.py --model Llama2-70B --model_parallel --inter-device-attention --simulation_result_path simulation_results_long_context.csv --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --seqlen 2304 
