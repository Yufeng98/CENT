import torch
from inspect import signature

from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torch.profiler import profile, ProfilerActivity
import argparse
import os
import random
import uuid

from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.engine.arg_utils import EngineArgs

# # Only require in vllm 0.10.0
# from vllm import _custom_ops as vllm_custom_ops
# from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size_triton

def _fallback_moe_sum(input_tensor, output_tensor):
    if input_tensor.shape == output_tensor.shape:
        output_tensor.copy_(input_tensor)
        return output_tensor
    if input_tensor.dim() == output_tensor.dim() + 1:
        torch.sum(input_tensor, dim=1, out=output_tensor)
        return output_tensor
    batch = output_tensor.shape[0]
    hidden = output_tensor.shape[-1]
    reshaped = input_tensor.contiguous().view(batch, -1, hidden)
    reduced = reshaped.sum(dim=1)
    output_tensor.copy_(reduced.view_as(output_tensor))
    return output_tensor


def _fallback_topk_softmax(topk_weights, topk_ids, token_expert_indices,
                           gating_output):
    scores = torch.softmax(gating_output.float(), dim=-1)
    selected_weights, selected_ids = torch.topk(scores,
                                                k=topk_weights.shape[-1],
                                                dim=-1,
                                                sorted=False)
    topk_weights.copy_(selected_weights.to(topk_weights.dtype))
    topk_ids.copy_(selected_ids.to(topk_ids.dtype))
    token_expert_indices.copy_(
        torch.arange(topk_weights.shape[-1],
                     device=token_expert_indices.device,
                     dtype=token_expert_indices.dtype).expand_as(
                         token_expert_indices))
    return topk_weights, topk_ids


def _ensure_moe_op(name, fallback_fn):
    try:
        moe_namespace = getattr(torch.ops, '_moe_C')
    except AttributeError:
        moe_namespace = None
    has_custom_op = bool(moe_namespace and hasattr(moe_namespace, name))
    if has_custom_op:
        return

    print(f"Warning: _moe_C.{name} missing; installing fallback.")
    setattr(vllm_custom_ops, name, fallback_fn)


def _maybe_stop_cuda_profiler():
    if os.environ.get("CUDA_PROFILER_CONTROL") != "1":
        return
    if not torch.cuda.is_available():
        print("Warning: CUDA profiler control requested, but CUDA is unavailable; skipping cudaProfilerStop().")
        return
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except RuntimeError as exc:
        print(f"Warning: cudaProfilerStop() failed: {exc}")

# # Only require in vllm 0.10.0
# _ensure_moe_op("moe_align_block_size", lambda topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad: moe_align_block_size_triton(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad))
# _ensure_moe_op("moe_sum", _fallback_moe_sum)
# _ensure_moe_op("topk_softmax", _fallback_topk_softmax)


def main():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355' 
    os.environ['VLLM_SKIP_P2P_CHECK'] = '1' 
    os.environ['NCCL_NVLS_ENABLE'] = '0' 

    PROFILE_MEM_HISTORY = 0
    PRINT_MODEL_PARAM_BUFFER_SIZE = 0
    PYTORCH_PROFILE_ACTIVITY = 0
    PRINT_RESULTS = 0
    PRINT_MEM_USAGE = 1

    model_directory = "Llama-3.1-8B"
    attention_types = ["eager", "flash_attention_2", "sdpa", "paged_attention"]

    parser = argparse.ArgumentParser(description="LLM inference")
    parser.add_argument("--model_dir", help="LLM model directory", default=model_directory)
    parser.add_argument("--attention_type", help="LLM model directory", default=attention_types[0], choices=attention_types)
    parser.add_argument("--prompt_token", help="Number of prompt tokens", required=True)
    parser.add_argument("--generate_token", help="Number of generated tokens", required=True)
    parser.add_argument("--batch_size", help="Number of batches", required=True)
    parser.add_argument("--np", help="Number of gpus", default=1)
    parser.add_argument("--num_frames", type=int, help="Number of frames in video", default=100)
    parser.add_argument("--video", help="Use video input", action="store_true")
    parser.add_argument("--enable_ep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expert parallelism for MoE models. Defaults to on for DeepSeek models.")
    parser.add_argument("--enable_eplb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expert parallel load balancing (requires expert parallelism).")
    parser.add_argument("--expert_placement_strategy",
        choices=["linear", "round_robin"],
        default=None,
        help="Placement strategy when expert parallelism is enabled.")
    parser.add_argument("--eplb_window_size",
        type=int,
        default=None,
        help="EPLB window size override.")
    parser.add_argument("--eplb_step_interval",
        type=int,
        default=None,
        help="EPLB step interval override.")
    parser.add_argument("--eplb_num_redundant_experts",
        type=int,
        default=None,
        help="Number of redundant experts when EPLB is active.")
    parser.add_argument("--eplb_log_balancedness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log EPLB balancedness metrics each step.")
    args = parser.parse_args()
    _maybe_stop_cuda_profiler()

    model_directory = args.model_dir
    attention_type = args.attention_type
    prompt_token_count = int(args.prompt_token)
    output_token_count = int(args.generate_token)
    batch_size = int(args.batch_size)
    requested_tp = int(args.np)
    np = requested_tp
    engine_arg_params = set(signature(EngineArgs.__init__).parameters.keys())
    supports_enable_ep = "enable_expert_parallel" in engine_arg_params
    supports_expert_placement = "expert_placement_strategy" in engine_arg_params
    supports_enable_eplb = "enable_eplb" in engine_arg_params
    supports_eplb_config = "eplb_config" in engine_arg_params

    # is_deepseek_model = "deepseek" in model_directory.lower()
    is_deepseek_model = False

    enable_ep = args.enable_ep
    enable_eplb = args.enable_eplb

    if not supports_enable_ep:
        if enable_ep is not None:
            print("Warning: installed vLLM does not support expert parallelism; ignoring --enable_ep/--no-enable_ep.")
        enable_ep = False

    if not supports_enable_eplb:
        if enable_eplb is not None:
            print("Warning: installed vLLM does not support expert parallel load balancing; ignoring --enable_eplb/--no-enable_eplb.")
        enable_eplb = False

    if enable_ep is None:
        enable_ep = supports_enable_ep and is_deepseek_model
    if enable_eplb is None:
        enable_eplb = supports_enable_eplb and enable_ep and is_deepseek_model
    if enable_eplb and not enable_ep and supports_enable_ep:
        enable_ep = True

    eplb_forced_off = False
    if enable_eplb:
        tp_size = np
        dp_size = 1
        if tp_size <= 1 and dp_size <= 1:
            print(f"Warning: EPLB requires tensor or data parallel > 1; disabling for current run (np={np}).")
            enable_eplb = False
            eplb_forced_off = True

    requested_eplb_config = {}
    if args.eplb_window_size is not None:
        requested_eplb_config["window_size"] = args.eplb_window_size
    if args.eplb_step_interval is not None:
        requested_eplb_config["step_interval"] = args.eplb_step_interval
    if args.eplb_num_redundant_experts is not None:
        requested_eplb_config["num_redundant_experts"] = args.eplb_num_redundant_experts
    if args.eplb_log_balancedness is not None:
        requested_eplb_config["log_balancedness"] = args.eplb_log_balancedness

    if enable_eplb and supports_eplb_config:
        eplb_config = requested_eplb_config
    else:
        if requested_eplb_config:
            if not supports_eplb_config:
                print("Warning: installed vLLM does not support EPLB configuration overrides; ignoring provided arguments.")
            else:
                if eplb_forced_off:
                    print("Warning: EPLB disabled because tensor or data parallel size is 1; ignoring EPLB configuration overrides.")
                else:
                    print("Warning: EPLB disabled; ignoring EPLB configuration overrides.")
        eplb_config = {}

    expert_placement = None
    if args.expert_placement_strategy:
        if supports_expert_placement:
            expert_placement = args.expert_placement_strategy
        else:
            print("Warning: installed vLLM does not support expert_placement_strategy; ignoring argument.")

    ep_kwargs = {}
    if enable_ep:
        ep_kwargs["enable_expert_parallel"] = True
        if expert_placement:
            ep_kwargs["expert_placement_strategy"] = expert_placement
    if enable_eplb:
        ep_kwargs["enable_eplb"] = True
        if eplb_config:
            ep_kwargs["eplb_config"] = eplb_config

    llm_kwargs = dict(ep_kwargs)

    if enable_ep:
        mode_msg = "Expert parallelism enabled"
        if enable_eplb:
            mode_msg += " with load balancing"
        print(mode_msg)

    print(f"Running {attention_type} for {batch_size} batches of {prompt_token_count} input tokens and {output_token_count} output tokens.")

    output_token_count += 1
    if PROFILE_MEM_HISTORY:
        print("PROFILE_MEM_HISTORY")
        torch.cuda.memory._record_memory_history()
    model = None
    if attention_type == "paged_attention":
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_token_count, ignore_eos=True)
        if "Qwen2-VL" in model_directory or "Qwen2.5-VL" in model_directory:
            print("flag1")
            model = LLM(model=model_directory, tensor_parallel_size=np, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, limit_mm_per_prompt={"image": 1}, **llm_kwargs)
        elif "DeepSeek-VL2" in model_directory:
            print("flag2")
            model = LLM(model=model_directory, tensor_parallel_size=np, enable_chunked_prefill=False, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]}, **llm_kwargs)
        elif "DeepSeek" in model_directory or "MiniCPM" in model_directory or "Llama-4" in model_directory or "Llama4" in model_directory:
            print("flag3")
            # model = LLM(model=model_directory, tensor_parallel_size=np, enable_chunked_prefill=False, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, max_seq_len_to_capture=prompt_token_count+output_token_count+1, trust_remote_code=True)
            model = LLM(model=model_directory, tensor_parallel_size=np, enable_chunked_prefill=False, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, trust_remote_code=True, **llm_kwargs)
        else:
            print("flag4")
            # model = LLM(model=model_directory, tensor_parallel_size=np, enable_chunked_prefill=False, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, max_seq_len_to_capture=prompt_token_count+output_token_count+1)
            model = LLM(model=model_directory, tensor_parallel_size=np, enable_chunked_prefill=False, max_num_seqs=batch_size, max_model_len=prompt_token_count+output_token_count+1, **llm_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_directory, device_map="auto", torch_dtype=torch.float16, attn_implementation=attention_type)

    model_size = 0
    if PRINT_MEM_USAGE:
        for i in range(torch.cuda.device_count()):
            model_size += torch.cuda.max_memory_allocated(device=torch.cuda.device(i))

    if PRINT_MODEL_PARAM_BUFFER_SIZE:
        print("PRINT_MODEL_PARAM_BUFFER_SIZE")
        param_size = 0
        for name, param in model.named_buffers():
            param_size += param.nelement() * param.element_size()
            print(f"param({name}): shape({param.size()}) nelements({param.nelement()}) element_size({param.element_size()}) size({param.nelement() * param.element_size()})")
        buffer_size = 0
        for name, buffer in model.named_buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            print(f"buffer({name}): shape({buffer.size()}) nelements({buffer.nelement()}) element_size({buffer.element_size()}) size({buffer.nelement() * buffer.element_size()})")
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

    tokenizer = AutoTokenizer.from_pretrained(model_directory, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    sentence = "The accident occurred during a test of the steam turbines ability to power the emergency feedwater pumps in the event of a simultaneous loss of external power and coolant pipe rupture Following an accidental drop in reactor power to near zero the operators restarted the reactor in preparation for the turbine test with a prohibited control rod configuration Upon successful completion of the test the reactor was then shut down for maintenance Due to a variety of factors this action resulted in a power surge at the base of the reactor which brought about the rupture of reactor components and the loss of coolant This process led to steam explosions and a meltdown which destroyed the containment building This was followed by a reactor core fire which lasted until the fourth of May nineteen eighty six during which airborne radioactive contaminants were spread throughout the USSR and Europe In response to the initial accident a ten kilometer or sixty four miles radius exclusion zone was created thirty six hours after the accident from which approximately forty nine thousand people were evacuated primarily from Pripyat The exclusion zone was later increased to a radius of thirty kilometers nineteen miles from which an additional nearly sixty eight thousand people were evacuated Following the reactor explosion which killed two engineers and severely burned two more an emergency operation to put out the fires and stabilize the surviving reactor began during which two hundred thirty seven workers were hospitalized of which a hundred thirty four exhibited symptoms of acute radiation"
    question = "What is the content of each image? "

    # 7952x5304 16254 tokens in Qwen2-VL-7B
    # 284×189=53676=53960 tokens w/ patch size 28, compression
    image_lion = "002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"  
    # 3722x2353 11202 tokens in Qwen2-VL-7B
    # 132×84=11088 tokens w/ patch size 28
    image_duck = "2015_Kaczka_krzyżowka_w_wodzie_(samiec).jpg"  
    # 2048x1365 3607 tokens in Qwen2-VL-7B
    # 73×48=3504 tokens w/ patch size 28
    image_demo = "demo.jpeg"
    # https://www.pexels.com/photo/purple-foot-bridge-220769/
    # 1280x960 1594 tokens in Qwen2-VL-7B
    # 45×34=1530 tokens w/ patch size 28
    image_1280x960 = "1280x960.jpg"    
    # 640x480 421 tokens in Qwen2-VL-7B
    # 22×17=374 tokens w/ patch size 28
    image_640x480 = "640x480.jpg"  

    # https://huggingface.co/datasets/raushan-testing-hf/videos-test/blob/main/sample_demo_1.mp4
    # 640x360_25fps video
    # 1 frames 329 tokens, 22×12=264 w/ patch size 28
    # 100 frames (4s) 14980 tokens, compression
    # 400 frames (16s) 59830 tokens, compression
    # 1000 frames (40s) 149530 tokens, compression
    video_demo = "sample_demo_1.mp4"
    # 3840x2160_30fps video
    # 1 frames 329 tokens, 137x77=10549 w/ patch size 28, compression
    # 100 frames 14980 tokens, compression
    video_UHD = "11536947-uhd_3840_2160_30fps.mp4"


    image_token_length = {
        image_lion: 16254,
        image_duck: 11202,
        image_demo: 3607
    }

    def tokenize_prompt(batch_size, prompt_token_count, sentence):
        
        while len(sentence.split(" ")) < prompt_token_count:
            sentence += " " + sentence
        # print (len(sentence.split(" ")))

        tokens = tokenizer.encode(sentence, add_special_tokens=False)[:prompt_token_count]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)

        prompts = []
        for _ in range(batch_size):
            prompts.append(prompt)

        return prompts

    # def tokenize_prompt_shuffle(batch_size, prompt_token_count, sentence):
    #     tokens = tokenizer.encode(sentence, add_special_tokens=False)
    #     if not tokens:
    #         raise ValueError("Base sentence produced no tokens; cannot create prompts.")

    #     target_length = prompt_token_count + max(1, min(batch_size, 1024))
    #     while len(tokens) < target_length:
    #         tokens = tokens + tokens

    #     max_start = len(tokens) - prompt_token_count
    #     if max_start > 0 and batch_size <= max_start + 1:
    #         start_offsets = random.sample(range(max_start + 1), batch_size)
    #     else:
    #         start_offsets = [random.randint(0, max_start) if max_start > 0 else 0 for _ in range(batch_size)]

    #     prompts = []
    #     for offset in start_offsets:
    #         prompt_tokens = tokens[offset:offset + prompt_token_count]
    #         prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    #         prompts.append(prompt)

    #     return prompts

    def Qwen_prompt(modality, question):
        if modality == "image":
            placeholder = "<|image_pad|>"
        elif modality == "video":
            placeholder = "<|video_pad|>"
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")
        return prompt

    def MiniCPM_prompt(modality, question):
        if modality == "image":
            placeholder = "(<image>./</image>)"
        elif modality == "video":
            placeholder = "(<video>./</video>)"
        messages = [{
            'role': 'user',
            'content': f'{placeholder[modality]}\n{question}'
        }]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def image_prompt(image, question):
        image = Image.open(image)
        if "MiniCPM" in model_directory:
            prompt_pad = MiniCPM_prompt("image", question)
        elif "Qwen2-VL" in model_directory or "Qwen2.5-VL" in model_directory:
            prompt_pad = Qwen_prompt("image", question)
        elif "DeepSeek-VL2" in model_directory:
            prompt_pad = f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"

        prompt = {
            "prompt": prompt_pad,
            "multi_modal_data": {"image": image}
        }
        return prompt

    def video_prompt(video, question):
        if "MiniCPM" in model_directory:
            prompt_pad = MiniCPM_prompt("video", question)
        elif "Qwen2-VL" in model_directory or "Qwen2.5-VL" in model_directory:
            prompt_pad = Qwen_prompt("video", question)
        prompt = {
            "prompt": prompt_pad,
            "multi_modal_data": {"video": video}
        }
        return prompt

    def multi_modal_prompts(batch_size, prompt_token_count, question, image_demo, video_demo):
        if args.video:
            video = VideoAsset(name="sample_demo_1.mp4", num_frames=args.num_frames).np_ndarrays
            prompt = video_prompt(video, question)
        else:
            if prompt_token_count <= 1024:
                prompt = image_prompt(image_640x480, question)
            elif prompt_token_count <= 2048:
                prompt = image_prompt(image_1280x960, question)
            elif prompt_token_count <= 4096:
                prompt = image_prompt(image_demo, question)
            elif prompt_token_count <= 8192:
                question += " " + question * 500
                prompt = image_prompt(image_demo, question)
            elif prompt_token_count <= 16384:
                prompt = image_prompt(image_lion, question)
            elif prompt_token_count <= 32768:
                question += " " + question * 2000
                prompt = image_prompt(image_lion, question)
            elif prompt_token_count <= 65536:
                question += " " + question * 6100
                prompt = image_prompt(image_lion, question)
            elif prompt_token_count <= 131072:
                question += " " + question * 14300
                prompt = image_prompt(image_lion, question)
            else:
                raise ValueError("Prompt token count too high")
        prompts = []
        for _ in range(batch_size):
            prompts.append(prompt)
        return prompts
        

    if "VL" in model_directory or "MiniCPM" in model_directory:
        multi_modal_inputs = multi_modal_prompts(batch_size, prompt_token_count, question, image_demo, video_UHD)
    else:
        model_inputs = tokenize_prompt(batch_size, prompt_token_count, sentence)

    generated_ids = None

    def model_generate():
        global generated_ids
        if attention_type == "paged_attention":
            if "VL" in model_directory:
                generated_ids = model.generate(multi_modal_inputs, sampling_params=sampling_params)
            else:
                # generated_ids = model.generate(prompt_token_ids=model_inputs['input_ids'].tolist(), sampling_params=sampling_params)
                generated_ids = model.generate(model_inputs, sampling_params=sampling_params)
        else:
            generated_ids = model.generate(**model_inputs, max_new_tokens=output_token_count)

    if PYTORCH_PROFILE_ACTIVITY:
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, use_cuda=True, profile_memory=True) as prof:
            model_generate()
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10000))
    else:
        model_generate()

    if PRINT_MEM_USAGE:
        print("PRINT_MEM_USAGE")
        total_size = 0
        print(torch.cuda.device_count(), "GPU")
        for i in range(torch.cuda.device_count()):
            total_size += torch.cuda.max_memory_allocated(device=torch.cuda.device(i))
        total_batch_size = total_size - model_size
        average_batch_size = total_batch_size / batch_size
        print("Total: {:.2f} GB \b\tModel: {:.2f} GB \b\tBatch: {:.2f} GB (avg {:.2f} GB / batch)".format((total_size / 1024 ** 3), (model_size / 1024 ** 3), (total_batch_size / 1024 ** 3), (average_batch_size / 1024 ** 3)))

    if PROFILE_MEM_HISTORY:
        print("PROFILE_MEM_HISTORY")
        torch.cuda.memory._dump_snapshot("snapshot_{}_pt{}_dt{}_b{}.pickle".format(attention_type,prompt_token_count,output_token_count,batch_size))

    if PRINT_RESULTS:
        if attention_type == "paged_attention":
            for output in generated_ids:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        else:
            print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

if __name__ == "__main__":
    main()
