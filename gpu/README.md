## Installation

```bash
# vllm-0.6.6.post1 does not support MLA in DeepSeek-V2, using MHA instead
conda create -n vllm-0.6.6.post1 python==3.12
conda activate vllm-0.6.6.post1
pip install -r requirements.txt
pip install vllm==0.6.6.post1
cp envs/vllm/lib/python3.12/site-packages/vllm/engine/llm_engine.py ../../miniforge3/envs/vllm-0.6.6.post1/lib/python3.12/site-packages/vllm/engine/

# vllm-0.7.1 supports MLA in DeepSeek-V2, but not well implemented
# vllm-0.7.2 optmizes DeepSeek-V2 but still far behind the old MHA implementation
# Install transformer manually before transformer 4.49 released, to run Qwen2.5
conda create -n vllm-0.7.2 python==3.12
conda activate vllm-0.7.2
pip install -r requirements.txt
pip install vllm==0.7.2
pip install git+https://github.com/huggingface/transformers
cp envs/vllm/lib/python3.12/site-packages/vllm/engine/llm_engine_0.7.2.py ../../miniforge3/envs/vllm-0.7.1/lib/python3.12/site-packages/vllm/engine/

# vllm-0.8.0 supports MLA in DeepSeek-V2 well
conda create -n vllm-0.8.0 python==3.12
conda activate vllm-0.8.0
pip install -r requirements.txt
pip install vllm==0.8.0
pip install git+https://github.com/huggingface/transformers
cp envs/vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py ../../miniforge3/envs/vllm-0.8.0/lib/python3.12/site-packages/vllm/v1/engine/
# DeepSeek-V2: Newer transformers require rope_scalings to be floats (thus the “must be a float” messages). Edit your local model’s config.json to add decimals.

# use this for GPU utilization measurement for now
conda create -n vllm-new python==3.12
conda activate vllm-new
pip install -r requirements.txt
pip install vllm==0.8.0
pip install git+https://github.com/vllm-project/vllm
cp envs/vllm-profile/lib/python3.12/site-packages/vllm/worker/model_runner.py ../../miniforge3/envs/vllm-0.6.6.post1/lib/python3.12/site-packages/vllm/worker/
cp envs/vllm-profile/lib/python3.12/site-packages/vllm/v1/engine/core_ncu.py ../../miniforge3/envs/vllm-0.8.0/lib/python3.12/site-packages/vllm/v1/engine/

# This environment print time for each token, using for power monitoring
conda create -n vllm-profile python==3.12
conda activate vllm-profile
pip install -r requirements.txt
pip install vllm==0.6.6.post1
cp envs/vllm-profile/lib/python3.12/site-packages/vllm/engine/llm_engine.py ../../miniforge3/envs/vllm-profile/lib/python3.12/site-packages/vllm/engine/
```


Set start time and end time to collect power for each GPU and store in DGX_H100_power.csv.

```bash
python parse_power_logs.py --model Llama-3.1-8B --num_gpu 4 --date 2025-02-02 --start_time 15:26:02 --end_time 15:51:36 --log_path power_logs/Llama-3.1-8B_paged_attention_4_gpu_8_batch_1_prefill_131072_decoding.txt
```
