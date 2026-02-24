import math
import torch
import torch.nn.functional as F
from aim_sim import PIM
from utils import compare, apply_rotary_emb, repeat_kv, RMSNorm

debug = True

class TransformerBlock(PIM):
    """
    Llama TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(args)
        self.pim_compute = args.pim_compute
        if args.op_trace:
            self.trace_prepare = True
            self.trace_norm = True
            self.trace_fc_kqvo = True
            self.trace_attention = True
            self.trace_softmax = True
            self.trace_fc_kqvo = True
            self.trace_fc_ffn = True
            self.trace_activation = True
        else:
            self.trace_prepare = args.trace_prepare
            self.trace_norm = args.trace_norm
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_attention = args.trace_attention
            self.trace_softmax = args.trace_softmax
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_fc_ffn = args.trace_fc_ffn
            self.trace_activation = args.trace_activation
        self.model = args.model
        self.seqlen = args.seqlen
        self.vocab_size = 32000
        self.FC_devices = args.FC_devices
        self.embedding = args.embedding
        self.only_FC = args.only_FC
        self.only_trace = args.only_trace
        self.model_parallel = args.model_parallel
        self.pipeline_parallel = args.pipeline_parallel
        if args.channels_per_block:
            self.channels_per_block = args.channels_per_block
        else:
            self.channels_per_block = args.num_channels
        self.GEMV_order = args.GEMV
        self.reuse_size = args.reuse_size
        if "TP_param" in dic_model.keys():
            self.TP_param = dic_model["TP_param"].item()
        else:
            self.TP_param = 1
        self.dim = dic_model["dim"].item()
        self.n_heads = dic_model["n_heads"].item()
        self.head_dim = self.dim // self.n_heads // self.TP_param
        self.max_seq_len = args.max_seq_len
        self.GQA = False
        self.inter_device_attention = args.inter_device_attention
        self.n_repeat = 1
        if "n_kv_heads" in dic_model.keys():
            self.GQA = True
            self.n_kv_heads = dic_model["n_kv_heads"].item()
            self.n_repeat = self.n_heads // self.n_kv_heads
        else:
            self.n_kv_heads = self.n_heads
        self.x = dic_model["x"].float()
        self.SANorm = dic_model["SANorm"].float()
        self.FFNNorm = dic_model["FFNNorm"].float()
        if "freqs_cis" in dic_model.keys():
            self.freqs_cis = dic_model["freqs_cis"]
        self.start_pos = dic_model["start_pos"]
        self.sa = dic_model["sa"].float()
        self.h = dic_model["h"].float()
        self.out = dic_model["out"].float()
        self.wq = dic_model["wq"].float()
        self.wk = dic_model["wk"].float()
        self.wv = dic_model["wv"].float()
        self.xq = dic_model["xq"].float()
        self.xk = dic_model["xk"].float()
        self.xv = dic_model["xv"].float()
        self.cache_k = dic_model["cache_k"].float()
        self.cache_v = dic_model["cache_v"].float()
        self.scores = dic_model["scores"].float()
        self.output = dic_model["output"].float()
        self.wo = dic_model["wo"].float()
        self.w1 = dic_model["w1"].float()
        self.w2 = dic_model["w2"].float()
        if "w3" in dic_model.keys():
            self.w3 = dic_model["w3"].float()
        self.ffn = dic_model["ffn"].float()
        self.mode = {"vector":0, "weights":1, "cache_k":2, "cache_v":3, "score":4}
        self.total_banks = self.channels_per_block * self.num_banks
        if self.model_parallel:
            self.FC_total_banks = self.total_banks * self.FC_devices
            self.intra_device_attention = False if self.inter_device_attention else True
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1
            if banks_per_head < self.num_banks:
                self.intra_device_attention = True
        else:
            self.FC_total_banks = self.total_banks
            self.intra_device_attention = True

    def bank_index(self, index):
        # look for the bank to store a head
        dimm_index = index // (self.num_banks * self.num_channels)
        channel_index = (index - dimm_index * self.num_banks * self.num_channels) // self.num_banks
        bank_index = index % self.num_banks
        return dimm_index, channel_index, bank_index

    def store_to_DRAM_multi_channel(self, data, row_index, mode, op_trace):
        if mode == self.mode["cache_k"]:
            # Store each seq in a bank
            seqlen = data.shape[0]
            shape = data.shape
            for seq in range(seqlen):
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                data_seq = data[seq].reshape(-1)
                rows = self.head_dim * self.n_kv_heads // self.DRAM_column
                for row in range(rows):
                    data_row = data_seq[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, data_row, op_trace)  
                              
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = data.shape[-1]
                shape = data.shape
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iterations = self.head_dim // self.num_banks
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                dim = dim_iter * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
            else:
                seqlen = data.shape[-1]
                shape = data.shape
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                print("banks_per_head: ", banks_per_head)
                print("channels_per_head: ", channels_per_head)
                print("devices_per_head: ", devices_per_head)
                print("dim_iterations: ", dim_iterations)
                print("rows_per_seq_iteration: ", rows_per_seq_iteration)
                print("seq_iterations: ", seq_iterations)
                print("rows_per_seq: ", rows_per_seq)
                print("channels_per_row_offset: ", channels_per_row_offset)
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
        else:
            bank_dim = (data.shape[0] - 1) // self.total_banks + 1
            utilized_banks = (data.shape[0] - 1) // bank_dim + 1
            # print(data.shape, bank_dim, utilized_banks)
            if mode == self.mode["weights"]:
                if self.model_parallel:
                    bank_dim = (data.shape[0] - 1) // self.FC_total_banks + 1
                    utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    vector_length = data.shape[1]
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    if i < utilized_banks - 1:
                        num_vectors = bank_dim
                        shape = data[i * num_vectors : (i+1) * num_vectors].shape
                    else:
                        num_vectors = data.shape[0] - bank_dim * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        data_vector = data[i * bank_dim + vector]
                        for row in range(rows_per_vector):
                            # print(i * bank_dim + vector, dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row)
                            if row == rows_per_vector - 1:
                                data_tmp = data_vector[row * self.DRAM_column:]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, data_tmp, op_trace)
                            else:
                                data_tmp = data_vector[row * self.DRAM_column:(row + 1) * self.DRAM_column]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, data_tmp, op_trace)
                # print(shape)
            elif mode == self.mode["vector"]:
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim : (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim :]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    data_bank = data[i].reshape(-1)
                    shape = data_bank.shape
                    data_size = data_bank.shape
                    rows = (data_size[0] - 1) // self.DRAM_column + 1
                    for row in range(rows-1):
                        data_tmp = data_bank[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, data_tmp, op_trace)
                    data_tmp = data_bank[(rows-1) * self.DRAM_column:]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows - 1, 0, data_tmp.shape[0], data_tmp, op_trace)
            elif "vector_bank_group" in mode:
                # Gather the values in 4 banks in a bank group to 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 4) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1

                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     bank_group_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+bank_group_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "vector_neighbor_bank" in mode:
                # Gather the values in 2 neighboring banks in 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 2) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                # print(data.shape, bank_dim, utilized_banks)
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                # e.g. Llama2-7B model has 4096 dim, with 10 channels, there are total 160 banks, bank_dim = 4096 // 80 + 1 = 52, bursts_per_bank = 52 // 16 + 1 = 4
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "scores_bank_group" in mode:
                seqlen = data.shape[-1]
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                shape = data.shape
                bank_group_index = int(mode[-1])
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length : k * self.DRAM_column + (burst + 1) * self.burst_length]
                                    else:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length :]
                                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     shape = data.shape
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     data_bank = data[head][k * self.DRAM_column :]
                #                 else:
                #                     data_bank = data[head][k * self.DRAM_column : (k + 1) * self.DRAM_column]
                #                 self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, data_bank.shape[0], data_bank, op_trace)
        return shape
    
    def load_from_DRAM_multi_channel(self, shape, row_index, mode, offset, op_trace):
        result = []
        if mode == self.mode["cache_k"]:
            seqlen = offset
            for seq in range(seqlen):
                seqs = []
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                rows = self.head_dim * self.n_kv_heads // self.DRAM_column
                for row in range(rows):
                    seqs.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, op_trace))
                result.append(torch.cat(seqs).reshape(-1))
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = offset
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    dim_iterations = self.head_dim // self.num_banks
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        result_head = []
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                result_dim = []
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, op_trace))
                                result_head.append(torch.cat(result_dim))
                        result.append(torch.cat(result_head))
            else:
                seqlen = offset
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                result_heads = [[[] for dim in range(self.head_dim)] for head in range(self.n_kv_heads)]
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, op_trace))
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, op_trace))
                for head in range(self.n_kv_heads):
                    result_head = []
                    for dim in range(self.head_dim):
                        result_head.append(torch.cat(result_heads[head][dim]))
                    result.append(torch.cat(result_head))
        else:
            # print(shape, offset)
            if mode == self.mode["weights"]:
                vector_length = shape[1]
                rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                utilized_banks = (shape[0] - 1) // offset + 1  # shape = [4096, 11008]
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    vectors = []
                    # for vector in range(self.head_dim):
                    if i < utilized_banks - 1:
                        num_vectors = offset
                    else:
                        num_vectors = shape[0] - offset * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        rows = []
                        for row in range(rows_per_vector):
                            if row == rows_per_vector - 1:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, op_trace))
                            else:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, op_trace))
                        vectors.append(torch.cat(rows))
                    result.append(torch.cat(vectors))
                # print(torch.cat(result).reshape(shape).shape)
            elif mode == self.mode["vector"]:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows = []
                    rows_used = (offset - 1) // self.DRAM_column + 1
                    for row in range(rows_used-1):
                        rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, op_trace))
                    rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_used - 1, 0, offset - (rows_used - 1) * self.DRAM_column, op_trace))
                    result.append(torch.cat(rows).reshape(-1))
            elif "vector_bank_group" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                # print(utilized_banks)
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "vector_neighbor_bank" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "scores_bank_group" in mode:
                seqlen = offset
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                bank_group_index = int(mode[-1])
                score_heads = [[] for _ in range(self.n_heads)]
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        num_cols = self.burst_length
                                    else:
                                        num_cols = bank_dim - burst * self.burst_length
                                    score_heads[head].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, num_cols, op_trace))
                for head in range(self.n_heads):
                    score = torch.cat(score_heads[head])
                    result.append(score)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             scores = []
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, offset - k * self.DRAM_column, op_trace))
                #                 else:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, self.DRAM_column, op_trace))
                #             result.append(torch.cat(scores))
        return torch.cat(result).reshape(shape)

    def broadcast_store_query(self, channels_required, dest_row_index, data, op_trace):
        # match GQA score with key cache
        # xq[0][128] xq[8][128]  ... xq[56][128]
        # xq[1][128] xq[9][128]  ... xq[57][128]
        #                        ...
        # xq[7][128] xq[15][128] ... xq[63][128]
        for dest_channel in range(channels_required):
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + data.shape[-1] // self.burst_length
            for bank in range(self.dim//self.DRAM_column):
                if self.GQA:
                    data_tmp = torch.cat([data[i * self.DRAM_column + bank * self.head_dim : i * self.DRAM_column + (bank + 1) * self.head_dim] for i in range(self.n_kv_heads)])
                else:
                    data_tmp = data[bank * self.DRAM_column : (bank + 1) * self.DRAM_column]
                self.store_to_DRAM_single_bank(0, dest_channel, bank, dest_row_index, 0, self.DRAM_column, data_tmp, op_trace)

    def broadcast_load_query(self, dic, channels_required, row_index):
        for channel in range(channels_required):
            load_tmp = []
            load_reorder = []
            if self.GQA:
                for bank in range(self.dim//self.DRAM_column):
                    load_tmp.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index, 0, self.DRAM_column, False))
                for j in range(self.n_repeat):
                    reorder = []
                    for i in range(self.n_kv_heads):
                        reorder.append(load_tmp[i][j*128:(j+1)*128])
                    load_reorder.append(torch.cat(reorder))
            else:
                for bank in range(self.dim//self.DRAM_column):
                    load_reorder.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index, 0, self.DRAM_column, False))
            dic[channel] = torch.cat(load_reorder)

    def Vector_Matrix_Mul_weight_pim_only_trace(self, channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, timing):
        matrix_col_per_bank = (matrix_col - 1) // total_banks + 1
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        if self.GEMV_order == "no-reuse":
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for vector_index_per_bank in range(matrix_col_per_bank):
                    self.WR_BIAS_only_trace(channel_lst)
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    self.RD_MAC_only_trace(channel_lst)
        elif self.GEMV_order == "reuse-GB":
            num_reuse_groups = (matrix_col_per_bank - 1) // self.reuse_size + 1
            reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for reuse_group_index in range(num_reuse_groups):
                    if reuse_group_index < num_reuse_groups - 1:
                        num_left_maxtrix_col = reuse_group_size
                    else:
                        num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                    for latch_index in range(num_left_maxtrix_col):
                        self.WR_BIAS_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    for latch_index in range(num_left_maxtrix_col):
                        self.RD_MAC_only_trace(channel_lst)

    def Vector_Matrix_Mul_weight_af_pim_only_trace(self, channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, timing):
        matrix_col_per_bank = (matrix_col - 1) // total_banks + 1
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        if self.GEMV_order == "no-reuse":
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for vector_index_per_bank in range(matrix_col_per_bank):
                    self.WR_BIAS_only_trace(channel_lst)
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    if row_index == rows_per_vector - 1:
                        self.AF_only_trace(channel_lst)
                    self.RD_MAC_only_trace(channel_lst)
        elif self.GEMV_order == "reuse-GB":
            num_reuse_groups = (matrix_col_per_bank - 1) // (self.reuse_size // 2) + 1
            reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for reuse_group_index in range(num_reuse_groups):
                    if reuse_group_index < num_reuse_groups - 1:
                        num_left_maxtrix_col = reuse_group_size
                    else:
                        num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                    for latch_index in range(num_left_maxtrix_col):
                        self.WR_BIAS_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    if row_index == rows_per_vector - 1:
                        for latch_index in range(num_left_maxtrix_col):
                            self.AF_only_trace(channel_lst)
                            self.RD_AF_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        self.RD_MAC_only_trace(channel_lst)
            
    def Vector_Matrix_Mul_score_pim_only_trace(self, row_index_matrix, seqlen, timing):
        rows_per_vector = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
        WR_GB_op_size = self.DRAM_column // self.burst_length
        MAC_op_size = self.head_dim // self.burst_length
        heads_per_row = self.DRAM_column // self.head_dim
        # channels_required = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        seq_iterations = (seqlen - 1) // self.FC_total_banks + 1
        for row_index in range(rows_per_vector):
            for seq_iter in range(seq_iterations):
                if seq_iter == seq_iterations - 1:
                    left_channels = (seqlen - self.FC_total_banks * seq_iter - 1) // self.num_banks + 1
                else:
                    left_channels = channels_required_all_devices
                channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // left_channels * left_channels
                channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
                for repeat in range(self.n_repeat):
                    self.WR_GB_only_trace(channel_lst, WR_GB_op_size)
                    for head_iter in range(heads_per_row):
                        self.WR_BIAS_only_trace(channel_lst)
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + seq_iter * rows_per_vector + row_index, MAC_op_size, timing)
                        self.RD_MAC_only_trace(channel_lst)
    
    def Vector_Matrix_Mul_output_pim_only_trace(self, row_index_matrix, seqlen, timing):
        if self.intra_device_attention:
            rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
            rows_per_dim = (self.max_seq_len - 1) // self.DRAM_column + 1
            left_banks = self.num_banks
            num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
            channel_lst = [i for i in range(self.channels_per_block)]
            channel_multi_transformer_block_required = self.num_channels // self.channels_per_block * self.channels_per_block
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            dim_iterations = self.head_dim // left_banks
            for head_index_per_bank in range(num_heads_per_bank):
                row_current_head = row_index_matrix + (rows_per_dim * dim_iterations) * head_index_per_bank
                for repeat in range(self.n_repeat):
                    for row_offset in range(rows_per_seq):
                        if row_offset == rows_per_seq - 1:
                            op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                        else:
                            op_size = self.DRAM_column // self.burst_length
                        self.WR_GB_only_trace(channel_lst, op_size)
                        for dim_iter in range(dim_iterations):
                            self.WR_BIAS_only_trace(channel_lst)
                            self.MAC_ABK_only_trace(channel_lst, row_current_head + dim_iter * rows_per_dim + row_offset, op_size, timing)
                            self.RD_MAC_only_trace(channel_lst)
        else:
            channels_required_all_devices = self.FC_total_banks // self.num_banks
            channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
            # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
            # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
            # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                # seqlen = 32k, head_dim = 128
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 8,  32, 256, 2k
            channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 1,  2,  16,  128
            devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,  1,   4
            # iteration along the head dimension
            dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 16, 4,  1,   1
            # iteration along the sequence dimension or rows per sequence
            rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  1,  2,   16
            seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 32, 16,  2
            rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32, 32,  32
            channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1  

            if banks_per_head < self.num_banks:
                raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
            for repeat in range(self.n_repeat):
                for row_offset in range(rows_per_seq):
                    if row_offset == rows_per_seq - 1:
                        op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                    else:
                        op_size = self.DRAM_column // self.burst_length
                    self.WR_GB_only_trace(channel_lst, op_size)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            self.WR_BIAS_only_trace(channel_lst)
                            self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset * dim_iterations + dim_iter, op_size, timing)
                            self.RD_MAC_only_trace(channel_lst)
                    else:
                        self.WR_BIAS_only_trace(channel_lst)
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset // rows_per_seq_iteration, op_size, timing)
                        self.RD_MAC_only_trace(channel_lst)


    def Vector_Matrix_Mul_weight_pim(self, vector, row_index_matrix, vector_dim, matrix_col, FC_total_banks, op_trace_input, timing):
        matrix_col_per_bank = (matrix_col - 1) // FC_total_banks + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        accumulator = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        channels_utilized = self.channels_per_block
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        # print(matrix_col_per_bank, utilized_banks, channels_utilized)
        for channel in range(channels_required_all_devices):
            op_trace = channel == 0 and op_trace_input
            if channel == channels_required_all_devices - 1:
                left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
            else:
                left_banks = self.num_banks
            if self.GEMV_order == "no-reuse":
                # keep vector static in GB and frequently write BIAS registers
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1

                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for vector_index_per_bank in range(matrix_col_per_bank):
                        bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                        self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                        self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, 0, op_size, op_trace, timing)
                        accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                        for bank in range(left_banks):
                            accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
            elif self.GEMV_order == "reuse-GB":
                # keep vector static in GB and frequently write BIAS registers
                # write BIAS registers multiple times, then MAC_BK_GB multiple times, finally RD_MAC multiple times, saving the mode switching time between manipulating BIAS register and MAC_BK_GB
                num_reuse_groups = (matrix_col_per_bank - 1) // self.reuse_size + 1
                reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for reuse_group_index in range(num_reuse_groups):
                        if reuse_group_index < num_reuse_groups - 1:
                            num_left_maxtrix_col = reuse_group_size
                        else:
                            num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, bias, op_trace)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, latch_index, op_size, op_trace, timing)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                            for bank in range(left_banks):
                                accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
        # print(self.pim_device["dimm_" + str(0)].dimm["channel_" + str(31)].channel["bank_" + str(15)].latch[0])
        # print(self.pim_device["dimm_" + str(0)].dimm["channel_" + str(31)].channel["bank_" + str(15)].arrays[18])
        # print(self.pim_device["dimm_" + str(1)].dimm["channel_" + str(31)].channel["bank_" + str(15)].latch[0])
        # print(self.pim_device["dimm_" + str(1)].dimm["channel_" + str(31)].channel["bank_" + str(15)].arrays[18])
        return torch.tensor(accumulator[:matrix_col])
    
    def Vector_Matrix_Mul_weight_af_pim(self, vector, row_index_matrix, vector_dim, matrix_col, FC_total_banks, op_trace_input, timing):
        matrix_col_per_bank = (matrix_col - 1) // FC_total_banks + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        accumulator = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        accumulator_af = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        channels_utilized = self.channels_per_block
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        for channel in range(channels_required_all_devices):
            op_trace = channel == 0 and op_trace_input
            if channel == channels_required_all_devices - 1:
                left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
            else:
                left_banks = self.num_banks
            if self.GEMV_order == "no-reuse":
                # keep vector static in GB and frequently write BIAS registers
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for vector_index_per_bank in range(matrix_col_per_bank):
                        bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                        self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                        self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, 0, op_size, op_trace, timing)
                        if row_index == rows_per_vector - 1:
                            self.AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                            accumulator_af_loaded = self.RD_AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, op_trace)
                            for bank in range(left_banks):
                                accumulator_af[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_af_loaded[bank]
                        accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                        for bank in range(left_banks):
                            accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
            elif self.GEMV_order == "reuse-GB":
                # keep vector static in GB and frequently write BIAS registers
                # write BIAS registers multiple times, then MAC_BK_GB multiple times, finally RD_MAC multiple times, saving the mode switching time between manipulating BIAS register and MAC_BK_GB
                num_reuse_groups = (matrix_col_per_bank - 1) // (self.reuse_size // 2) + 1
                reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for reuse_group_index in range(num_reuse_groups):
                        if reuse_group_index < num_reuse_groups - 1:
                            num_left_maxtrix_col = reuse_group_size
                        else:
                            num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, bias, op_trace)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, latch_index, op_size, op_trace, timing)
                        if row_index == rows_per_vector - 1:
                            for latch_index in range(num_left_maxtrix_col):
                                vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                                self.AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                                accumulator_af_loaded = self.RD_AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, op_trace)
                                for bank in range(left_banks):
                                    accumulator_af[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_af_loaded[bank]
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                            for bank in range(left_banks):
                                accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
        return (torch.tensor(accumulator), torch.tensor(accumulator_af))

    def Vector_Matrix_Mul_score_pim(self, row_index_vector, row_index_matrix, op_trace_input, timing):
        bsz, seqlen, _, _ = self.cache_k.shape
        rows_per_vector = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
        WR_GB_op_size = self.DRAM_column // self.burst_length
        MAC_op_size = self.head_dim // self.burst_length
        accumulator = torch.zeros(torch.Size([bsz, self.n_heads, 1, seqlen]))
        heads_per_row = self.DRAM_column // self.head_dim
        # channels_required = self.channels_per_block
        channels_utilized = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        seq_iterations = (seqlen - 1) // self.FC_total_banks + 1
        for row_index in range(rows_per_vector):
            for seq_iter in range(seq_iterations):
                if seq_iter == seq_iterations - 1:
                    left_channels = (seqlen - self.FC_total_banks * seq_iter - 1) // self.num_banks + 1
                else:
                    left_channels = channels_required_all_devices
                for channel in range(left_channels):
                    op_trace = channel == 0 and op_trace_input
                    if (seq_iter == seq_iterations - 1) and (channel == left_channels - 1):
                        left_banks = seqlen - self.FC_total_banks * seq_iter - self.num_banks * channel
                    else:
                        left_banks = self.num_banks
                    for repeat in range(self.n_repeat):
                        vector = self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, row_index * self.n_repeat + repeat, row_index_vector, 0, WR_GB_op_size * self.burst_length, False)
                        self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, WR_GB_op_size, vector, op_trace)
                        for head_iter in range(heads_per_row):
                            bias = [accumulator[0][row_index * heads_per_row + head_iter * self.n_repeat + repeat][0][seq_iter * self.FC_total_banks + channel * self.num_banks + bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + seq_iter * rows_per_vector + row_index, head_iter * self.head_dim, 0, MAC_op_size, op_trace, timing)
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                            for bank in range(left_banks):
                                accumulator[0][row_index * heads_per_row + head_iter * self.n_repeat + repeat][0][seq_iter * self.FC_total_banks + channel * self.num_banks + bank] = accumulator_loaded[bank]
        return accumulator

    def Vector_Matrix_Mul_output_pim(self, scores, row_index_matrix, op_trace_input, timing):
        bsz, seqlen, _, _ = self.cache_k.shape
        accumulator = torch.zeros(torch.Size([bsz, self.n_heads, 1, self.head_dim]))
        if self.intra_device_attention:
            channels_required = self.channels_per_block
            rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
            rows_per_dim = (self.max_seq_len - 1) // self.DRAM_column + 1
            num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
            for channel in range(channels_required):
                op_trace = channel == 0 and op_trace_input
                if channel == channels_required - 1:
                    num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                else:
                    num_heads_iteration = num_heads_per_bank
                left_banks = self.num_banks
                dim_iterations = self.head_dim // left_banks
                for head_index_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains left_banks heads
                    row_current_head = row_index_matrix + (rows_per_dim * dim_iterations) * head_index_per_bank
                    for repeat in range(self.n_repeat):
                        head = channel * num_heads_per_bank + head_index_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        for row_offset in range(rows_per_seq):
                            if row_offset == rows_per_seq - 1:
                                op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                                pad_zero = torch.tensor([0 for _ in range(op_size * self.burst_length - (seqlen - row_offset * self.DRAM_column))])
                                vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column :]
                                vector = torch.cat((vector, pad_zero))
                            else:
                                op_size = self.DRAM_column // self.burst_length
                                vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column : (row_offset+1) * self.DRAM_column]
                            # vector = self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, head_index_per_bank * self.n_repeat + repeat, row_index_vector + row_offset, 0, op_size * self.burst_length, False)
                            self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_required, 0, op_size, vector, op_trace)
                            for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][dim_iter * left_banks + bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_required, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_required, row_current_head + dim_iter * rows_per_dim + row_offset, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_required, 0, op_trace)
                                for bank in range(left_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][dim_iter * left_banks + bank] = accumulator_loaded[bank]
        else:
            channels_utilized = self.channels_per_block
            channels_required_all_devices = self.FC_total_banks // self.num_banks
            # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
            # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
            # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
            # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                # seqlen = 32k, head_dim = 128
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 8,  32, 256, 2k
            channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 1,  2,  16,  128
            devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,  1,   4
            # iteration along the head dimension
            dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 16, 4,  1,   1
            # iteration along the sequence dimension or rows per sequence
            rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  1,  2,   16
            seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 32, 16,  2
            rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32, 32,  32
            channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
            for channel in range(channels_required_all_devices):
                op_trace = channel == 0 and op_trace_input
                if banks_per_head < self.num_banks:
                    raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                for repeat in range(self.n_repeat):
                    head = channel // (banks_per_head // self.num_banks)
                    print("repeat", repeat, "kv head", head, "head", head * self.n_repeat + repeat)
                    for row_offset in range(rows_per_seq):
                        if row_offset == rows_per_seq - 1:
                            op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                            pad_zero = torch.tensor([0 for _ in range(op_size * self.burst_length - (seqlen - row_offset * self.DRAM_column))])
                            vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column :]
                            vector = torch.cat((vector, pad_zero))
                        else:
                            op_size = self.DRAM_column // self.burst_length
                            vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column : (row_offset+1) * self.DRAM_column]
                        self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector, op_trace)
                        if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                            for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank] for bank in range(self.num_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset * dim_iterations + dim_iter, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                                for bank in range(self.num_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank] = accumulator_loaded[bank]
                        else:   
                            # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                            # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                            # each head is mapped on multiple devices
                            # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                            if channel // channels_per_row_offset == row_offset:    
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank] for bank in range(self.num_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset // rows_per_seq_iteration, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                                for bank in range(self.num_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank] = accumulator_loaded[bank]
                    if head > 0:
                        print(vector)
        return accumulator
    
    def store_for_neighbor_bank_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*2+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def store_for_input_only_trace(self, channels_required, total_banks, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)

    
    def store_for_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def load_from_input_only_trace(self, channels_required, total_banks, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank)
                for tb in range(num_transformer_blocks_per_device):
                    self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def load_from_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    
    def store_for_EWMUL_score_only_trace(self, channels_required, row_index, total_banks, bank_group_index, seqlen):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        # size_per_bank = (self.n_heads * seqlen - 1) // (total_banks // 4) + 1
        # rows_per_bank = (size_per_bank - 1) // self.DRAM_column + 1
        rows_per_score = (seqlen - 1) // self.DRAM_column + 1
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        for k in range(rows_per_score):
            if k == rows_per_score - 1:
                # size = (size_per_bank - 1) % self.DRAM_column + 1
                size = seqlen - self.DRAM_column * (rows_per_score - 1)
            else:
                size = self.DRAM_column
            for j in range(num_heads_per_bank):
                for i in range((size - 1) // self.burst_length + 1):
                    for bank in range(total_banks):
                        if bank % 4 == bank_group_index:
                            head = (bank // 4) * num_heads_per_bank + j
                            if head > self.n_heads - 1:
                                break
                            dimm_index, channel_index, bank_index = self.bank_index(bank)
                            for tb in range(num_transformer_blocks_per_device):
                                self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index + j * rows_per_score + k, self.burst_length)  
    
    def load_from_EWMUL_score_only_trace(self, channels_required, row_index, total_banks, bank_group_index, seqlen):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        # size_per_bank = (self.n_heads * seqlen - 1) // (total_banks // 4) + 1
        # rows_per_bank = (size_per_bank - 1) // self.DRAM_column + 1
        rows_per_score = (seqlen - 1) // self.DRAM_column + 1
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        for k in range(rows_per_score):
            if k == rows_per_score - 1:
                # size = (size_per_bank - 1) % self.DRAM_column + 1
                size = seqlen - self.DRAM_column * (rows_per_score - 1)
            else:
                size = self.DRAM_column
            for j in range(num_heads_per_bank):
                for i in range((size - 1) // self.burst_length + 1):
                    for bank in range(total_banks):
                        if bank % 4 == bank_group_index:
                            head = (bank // 4) * num_heads_per_bank + j
                            if head > self.n_heads - 1:
                                break
                            dimm_index, channel_index, bank_index = self.bank_index(bank)
                            for tb in range(num_transformer_blocks_per_device):
                                self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index + j * rows_per_score + k, self.burst_length)

    def store_for_score_only_trace(self, row_index, FC_total_banks, seqlen):
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        num_transformer_blocks_per_device = max(self.num_channels // channels_required_all_devices, 1)
        rows = max(1, (seqlen * self.n_heads - 1) // self.DRAM_column // self.num_banks // channels_required_all_devices + 1)
        columns = (seqlen * self.n_heads - 1) // rows // self.num_banks // channels_required_all_devices + 1
        # print(rows, columns, num_transformer_blocks_per_device)
        for row in range(rows):
            for burst in range((columns - 1) // self.burst_length + 1):
                for bank_index in range(self.num_banks):
                    for channel_index in range(channels_required_all_devices):
                        if channel_index > self.num_channels - 1:
                            break
                        for tb in range(num_transformer_blocks_per_device):
                            self.W_MEM_only_trace(channel_index + channels_required_all_devices * tb, bank_index, row_index + row * self.DRAM_column, self.burst_length)

    def load_for_score_only_trace(self, row_index, FC_total_banks, seqlen):
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        num_transformer_blocks_per_device = max(self.num_channels // channels_required_all_devices, 1)
        rows = max(1, (seqlen * self.n_heads - 1) // self.DRAM_column // self.num_banks // channels_required_all_devices + 1)
        columns = (seqlen * self.n_heads - 1) // rows // self.num_banks // channels_required_all_devices + 1
        for row in range(rows):
            for burst in range((columns - 1) // self.burst_length + 1):
                for bank_index in range(self.num_banks):
                    for channel_index in range(channels_required_all_devices):
                        if channel_index > self.num_channels - 1:
                            break
                        for tb in range(num_transformer_blocks_per_device):
                            self.R_MEM_only_trace(channel_index + channels_required_all_devices * tb, bank_index, row_index + row * self.DRAM_column, self.burst_length)
                            
                            
                            
    def apply_rotary_emb_pim_only_trace(self, row_index_xq, row_index_xk, op_trace_input):
        """
        Executes Rotary Positional Embedding entirely within the PIM channels.
        Replaces the legacy PNM formatting by utilizing the Global Buffer MIN.
        """
        if not op_trace_input:
            return

        # Calculate operation size based on head dimension and bus width
        base_op_size = self.head_dim // self.burst_length
        channels_utilized = self.channels_per_block
        channel_lst = [channel for channel in range(channels_utilized)]
        
        is_decode = (self.seqlen == 1)
        op_size_xq = base_op_size if is_decode else base_op_size * self.seqlen

        # GQA - Keys only exist for n_kv_heads. 
        # Calculate the number of banks holding KV heads based on n_repeat
        kv_banks = self.num_banks // self.n_repeat if self.GQA else self.num_banks
        op_size_xk = base_op_size if is_decode else base_op_size * self.seqlen
        
        
        # Rotate the Query Vectors (xq)
        for bank in range(self.num_banks):
            # unformatted real numbers (a, b, c, d) to the Global Buffer
            self.COPY_BK_GB_only_trace(channel_lst, bank, row_index_xq, op_size_xq)
            
            # use 256x256 MIN to cross wires and form (a+jb) pairs
            # This triggers the strict 8-cycle tSHUFFLE hardware delay in Ramulator
            self.MULINT_only_trace(channel_lst, op_size_xq)
            
            # write the aligned complex pairs back to bank
            self.COPY_GB_BK_only_trace(channel_lst, bank, row_index_xq, op_size_xq)
            
            # complex multiplication against pre-loaded freqs_cis weights
            # (Assuming EWMUL is mapped to operate on the target bank group)
            self.EWMUL_only_trace(channel_lst, row_index_xq, op_size_xq)

        # Rotate the Key Vectors (xk)
        for bank in range(kv_banks):
            self.COPY_BK_GB_only_trace(channel_lst, bank, row_index_xk, op_size_xk)
            self.MULINT_only_trace(channel_lst, op_size_xk)
            self.COPY_GB_BK_only_trace(channel_lst, bank, row_index_xk, op_size_xk)
            self.EWMUL_only_trace(channel_lst, row_index_xk, op_size_xk)


        # Inverse Shuffle
        # Q Vectors Inverse
        for bank in range(self.num_banks):
            self.COPY_BK_GB_only_trace(channel_lst, bank, row_index_xq, op_size_xq)
            self.time["MULINT"] += self.timing_constant["MULINT"] + op_size_xq
            self.file.write("AiM MULINT {} {} 0\n".format(op_size_xq, self.hex_channel_mask(channel_lst)))
            self.COPY_GB_BK_only_trace(channel_lst, bank, row_index_xq, op_size_xq)

        # K Vectors Inverse (Using GQA kv_banks)
        for bank in range(kv_banks):
            self.COPY_BK_GB_only_trace(channel_lst, bank, row_index_xk, op_size_xk)
            self.time["MULINT"] += self.timing_constant["MULINT"] + op_size_xk
            self.file.write("AiM MULINT {} {} 0\n".format(op_size_xk, self.hex_channel_mask(channel_lst)))
            self.COPY_GB_BK_only_trace(channel_lst, bank, row_index_xk, op_size_xk)
