import torch
torch.multiprocessing.set_sharing_strategy('file_system')

class Bank():
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.arrays = 0 if args.only_trace else torch.zeros(torch.Size([self.DRAM_row, self.DRAM_column]))
        self.latch = 0 if args.only_trace else [0 for _ in range(args.reuse_size)]
        self.activation_function_register = 0

class Channel(Bank):
    def __init__(self, args):
        super().__init__(args)
        self.num_banks = args.num_banks
        self.GB = torch.zeros(torch.Size([self.DRAM_column]))
        bank_lst = ["bank_" + str(i) for i in range(self.num_banks)]
        self.channel = {}
        for bank in bank_lst:
            self.channel[bank] = Bank(args)

class DIMM(Channel):
    """
    DIMM Class inherits DRAM topology from Channel and Bank class
    """
    def __init__(self, args):
        super().__init__(args)
        self.num_channels = args.num_channels
        channel_lst = ["channel_" + str(i) for i in range(self.num_channels)]
        self.dimm = {}
        for channel in channel_lst:
            self.dimm[channel] = Channel(args)

class PIM():
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.num_banks = args.num_banks
        self.num_channels = args.num_channels
        self.threads = args.threads
        self.pim_device = {}
        if not args.only_trace:
            if args.model_parallel:
                for i in range(args.FC_devices):
                    self.pim_device["dimm_{}".format(i)] = DIMM(args)
            else:
                self.pim_device["dimm_0"] = DIMM(args)
        self.op_trace = args.op_trace
        self.trace_file = args.trace_file
        self.file = open(self.trace_file, "w")
        # print(torch.linspace(-10, 10, 512))
        self.sigmoid_LUT = torch.sigmoid(torch.linspace(-10, 10, 512))
        self.sigmoid_LUT = torch.cat((self.sigmoid_LUT, torch.tensor([1])))
        # print(self.sigmoid_LUT)
        self.time = {
            "COPY_GB_BK": 0,
            "COPY_BK_GB": 0,
            "WR_GB": 0,
            "MAC_ABK": 0,
            "MAC_BK_BK": 0,
            "MAC_BK_GB": 0,
            "EWMUL": 0,
            "EWADD": 0,
            "AF": 0,
            "RD_MAC": 0,
            "RD_AF": 0,
            "WR_BIAS": 0,
            "RD_SBK": 0,
            "WR_SBK": 0,
            "MULINT": 0, #NEW
            "breakdown_sa_pow": 0,
            "breakdown_sa_weight": 0,
            "breakdown_sa_score": 0,
            "breakdown_sa_output": 0,
            "breakdown_ffn_weight": 0,
            "breakdown_embedding_weight": 0,
        }
        self.timing_constant = {
            "COPY_GB_BK": 45.5,
            "COPY_BK_GB": 42.5,
            "WR_GB": 32,
            "MAC_ABK": 49,
            "MAC_BK_BK": 49,
            "MAC_BK_GB": 49,
            "EWMUL": 47,
            "EWADD": 0,
            "AF": 60,
            "RD_MAC": 37.5,
            "RD_AF": 37.5,
            "WR_BIAS": 37.5,
            "RD_SBK": 30.5,
            "WR_SBK": 45.5,
            "MULINT": 8, #NEW
        }

    def hex_channel_mask(self, channel):
        mask = ["0"] * self.num_channels
        if isinstance(channel, list):
            for c in channel:
                mask[c] = "1"
        else:
            mask[channel] = "1"
        binary = "0b" + ''.join(mask)
        num = int(binary, 2)
        
        # convert int to hexadecimal
        hex_num = hex(num)
        return hex_num

    def address(self, dimm_index, channel_index, bank_index, row_index, col):
        bank_size = self.DRAM_column * self.DRAM_row
        channel_size = bank_size * self.num_banks
        dimm_size = channel_size * self.num_channels
        addr = dimm_index * dimm_size + channel_index * channel_size + bank_index * bank_size + row_index * self.DRAM_column + col
        return addr
    
    def store_to_DRAM_single_bank(self, dimm_index, channel_index, bank_index, row_index, col_index, size, data, op_trace):
        # GDDR6 stores with 32B granularity
        if op_trace and dimm_index == 0:
            for i in range((size - 1) // self.burst_length + 1):
                self.file.write("W MEM {} {} {}\n".format(channel_index, bank_index, row_index))
        self.pim_device["dimm_" + str(dimm_index)].dimm["channel_" + str(channel_index)].channel["bank_" + str(bank_index)].arrays[row_index][col_index : col_index + size] = data
    
    def store_to_DRAM_all_banks(self, dim_iter, channel, row_current_head, seq, head, xv_data, num_rows_per_seq, rows_per_dim):
        for bank in range(self.num_banks):
            dim = dim_iter * self.num_banks + bank
            row_offset = num_rows_per_seq - 1
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + 1
            self.store_to_DRAM_single_bank(0, channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
    
    def load_from_DRAM_single_bank(self, dimm_index, channel_index, bank_index, row_index, col_index, size, op_trace):
        if op_trace and dimm_index == 0:
            for i in range((size - 1) // self.burst_length + 1):
                self.file.write("R MEM {} {} {}\n".format(channel_index, bank_index, row_index))
        return self.pim_device["dimm_" + str(dimm_index)].dimm["channel_" + str(channel_index)].channel["bank_" + str(bank_index)].arrays[row_index][col_index : col_index + size]

    def WR_BIAS(self, dimm, channel, utilized_channels, latch_index, bias, op_trace):
        self.time["WR_BIAS"] += self.timing_constant["WR_BIAS"]
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM WR_BIAS 0 {}\n".format(self.hex_channel_mask(channel_lst)))
        for bank in range(self.num_banks):
            self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].latch[latch_index] = bias[bank]

    def MAC_BK_GB(self, dimm, channel, utilized_channels, row_index, col_index, latch_index, op_size, op_trace, timing):
        self.time[timing] += self.timing_constant["MAC_BK_GB"] + op_size
        self.time["MAC_BK_GB"] += self.timing_constant["MAC_BK_GB"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM MAC_ABK {} {} {}\n".format(op_size, self.hex_channel_mask(channel_lst), row_index))
        for bank in range(self.num_banks):
            for i in range(op_size):
                A = self.load_from_DRAM_single_bank(dimm, channel, bank, row_index, col_index + i * self.burst_length, self.burst_length, False)
                B = self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB[col_index + i * self.burst_length : col_index + (i+1) * self.burst_length]
                self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].latch[latch_index] += self.MAC(A, B, False)

    def MAC_BK_BK(self, dimm, channel, utilized_channels, row_index, col_index, latch_index, op_size, op_trace):
        self.time["MAC_BK_BK"] += self.timing_constant["MAC_BK_BK"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM MAC_ABK {} {} {}\n".format(op_size, self.hex_channel_mask(channel_lst), row_index))
        for bank in range(self.num_banks//2):
            for i in range(op_size):
                A = self.load_from_DRAM_single_bank(dimm, channel, bank*2, row_index, col_index + i * self.burst_length, self.burst_length, False)
                B = self.load_from_DRAM_single_bank(dimm, channel, bank*2+1, row_index, col_index + i * self.burst_length, self.burst_length, False)
                self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank*2)].latch[latch_index] += self.MAC(A, B, False)
    
    def RD_MAC(self, dimm, channel, utilized_channels, latch_index, op_trace):
        self.time["RD_MAC"] += self.timing_constant["RD_MAC"]
        result = []
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM RD_MAC 0 {}\n".format(self.hex_channel_mask(channel_lst)))
        for bank in range(self.num_banks):
            result.append(self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].latch[latch_index])
        return result
    
    def EWMUL(self, dimm, channel, utilized_channels, row_index, col_index, op_size, op_trace):
        # parallel in 4 bank groups, src bank 0 and 1, dest bank 2
        self.time["EWMUL"] += self.timing_constant["EWMUL"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM EWMUL {} {} {}\n".format(op_size, self.hex_channel_mask(channel_lst), row_index))
        for bank in range(self.num_banks//4):
            for i in range(op_size):
                A = self.load_from_DRAM_single_bank(dimm, channel, bank*4, row_index, col_index + i * self.burst_length, self.burst_length, False)
                B = self.load_from_DRAM_single_bank(dimm, channel, bank*4+1, row_index, col_index + i * self.burst_length, self.burst_length, False)
                self.store_to_DRAM_single_bank(dimm, channel, bank*4+2, row_index, col_index + i * self.burst_length, self.burst_length, A * B, False)

    def WR_GB(self, dimm, channel, utilized_channels, col_index, op_size, data, op_trace):
        self.time["WR_GB"] += self.timing_constant["WR_GB"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM WR_GB {} 0 {}\n".format(op_size, self.hex_channel_mask(channel_lst)))
        self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB[col_index : col_index + op_size * self.burst_length] = data

    def COPY_BK_GB(self, dimm, channel, utilized_channels, bank, row_index, col_index, op_size, op_trace):
        self.time["COPY_BK_GB"] += self.timing_constant["COPY_BK_GB"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM COPY_BKGB {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel_lst), bank, row_index))
        self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB[col_index : col_index + op_size * self.burst_length] = self.load_from_DRAM_single_bank(dimm, channel, bank, row_index, col_index, op_size * self.burst_length, False)

    def COPY_GB_BK(self, dimm, channel, utilized_channels, bank, row_index, col_index, op_size, op_trace):
        self.time["COPY_GB_BK"] += self.timing_constant["COPY_GB_BK"] + op_size
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM COPY_GBBK {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel_lst), bank, row_index))
        data = self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB[col_index : col_index + op_size * self.burst_length]
        self.store_to_DRAM_single_bank(dimm, channel, bank, row_index, col_index, op_size * self.burst_length, data, False)
    
    def AF(self, dimm, channel, utilized_channels, latch_index, op_trace):
        self.time["AF"] += self.timing_constant["AF"]
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM AF {}\n".format(self.hex_channel_mask(channel_lst)))
        interval = 20 / 512
        for bank in range(self.num_banks):
            x = self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].latch[latch_index]
            A = int(((x - (-10)) // interval).item())
            if A > 511:
                A = 511
            elif A < 0:
                A = 0
            interpolation = self.sigmoid_LUT[A] + (x - (-10 + A * interval)) * (self.sigmoid_LUT[A+1] - self.sigmoid_LUT[A])
            # print(x, torch.sigmoid(x), self.sigmoid_LUT[A], interpolation, (torch.sigmoid(x) - interpolation) / torch.sigmoid(x))
            self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].activation_function_register = interpolation

    def RD_AF(self, dimm, channel, utilized_channels, op_trace):
        self.time["RD_AF"] += self.timing_constant["RD_AF"]
        result = []
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            self.file.write("AiM RD_AF 0 {}\n".format(self.hex_channel_mask(channel_lst)))
        for bank in range(self.num_banks):
            result.append(self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].channel["bank_" + str(bank)].activation_function_register)
        return result
    
    def W_MEM_only_trace(self, channel_index, bank_index, row_index, size):
        for i in range((size - 1) // self.burst_length + 1):
            self.file.write("W MEM {} {} {}\n".format(channel_index, bank_index, row_index))
    
    def R_MEM_only_trace(self, channel_index, bank_index, row_index, size):
        for i in range((size - 1) // self.burst_length + 1):
            self.file.write("R MEM {} {} {}\n".format(channel_index, bank_index, row_index))
    
    def WR_ABK_only_trace(self, channel, row_index, op_size):
        self.file.write("AiM WR_ABK 0 {} {}\n".format(self.hex_channel_mask(channel), row_index))

    def WR_BIAS_only_trace(self, channel):
        self.time["WR_BIAS"] += self.timing_constant["WR_BIAS"]
        self.file.write("AiM WR_BIAS 0 {}\n".format(self.hex_channel_mask(channel)))

    def MAC_ABK_only_trace(self, channel, row_index, op_size, timing):
        self.time[timing] += self.timing_constant["MAC_ABK"] + op_size
        self.time["MAC_ABK"] += self.timing_constant["MAC_ABK"] + op_size
        self.file.write("AiM MAC_ABK {} {} {}\n".format(op_size, self.hex_channel_mask(channel), row_index))
    
    def RD_MAC_only_trace(self, channel):
        self.time["RD_MAC"] += self.timing_constant["RD_MAC"]
        self.file.write("AiM RD_MAC 0 {}\n".format(self.hex_channel_mask(channel)))
    
    def EWMUL_only_trace(self, channel, row_index, op_size):
        # parallel in 4 bank groups, src bank 0 and 1, dest bank 2
        self.time["EWMUL"] += self.timing_constant["EWMUL"] + op_size
        self.file.write("AiM EWMUL {} {} {}\n".format(op_size, self.hex_channel_mask(channel), row_index))
    
    def EWADD_only_trace(self, op_size):
        self.time["EWADD"] += self.timing_constant["EWADD"] + op_size
        self.file.write("AiM EWADD {} 0 0\n".format(op_size))

    def WR_GB_only_trace(self, channel, op_size):
        self.time["WR_GB"] += self.timing_constant["WR_GB"] + op_size
        self.file.write("AiM WR_GB {} 0 {}\n".format(op_size, self.hex_channel_mask(channel)))

    def COPY_BK_GB_only_trace(self, channel, bank, row_index, op_size):
        assert bank < self.num_banks
        self.time["COPY_BK_GB"] += self.timing_constant["COPY_BK_GB"] + op_size
        self.file.write("AiM COPY_BKGB {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))

    def COPY_GB_BK_only_trace(self, channel, bank, row_index, op_size):
        assert bank < self.num_banks
        self.time["COPY_GB_BK"] += self.timing_constant["COPY_GB_BK"] + op_size
        self.file.write("AiM COPY_GBBK {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))
    
    def AF_only_trace(self, channel):
        self.time["AF"] += self.timing_constant["AF"]
        self.file.write("AiM AF {}\n".format(self.hex_channel_mask(channel)))
        
    def RD_AF_only_trace(self, channel):
        self.time["RD_AF"] += self.timing_constant["RD_AF"]
        self.file.write("AiM RD_AF 0 {}\n".format(self.hex_channel_mask(channel)))
    
    def SYNC_only_trace(self):
        self.file.write("AiM SYNC\n")
    
    def finish(self):
        self.file.write("AiM EOC\n")

    def MAC(self, A, B, profile: bool):
        result = A * B
        return result.sum()
    
    def Vector_Vector_Mul_Row(self, A, B, profile: bool):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [self.MAC(
                    A[i*self.burst_length:(i+1)*self.burst_length], 
                    B[i*self.burst_length:(i+1)*self.burst_length], profile) for i in range(n-1)]
        lst.append(self.MAC(A[(n-1)*self.burst_length:], B[(n-1)*self.burst_length:], profile))
        return sum(lst)

    def Vector_Vector_Mul(self, A, B, profile: bool):
        n = (A.shape[0] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_Mul_Row(
                A[i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[i*self.DRAM_column:(i+1)*self.DRAM_column], profile) for i in range(n-1)]
        lst.append(self.Vector_Vector_Mul_Row(A[(n-1)*self.DRAM_column:], B[(n-1)*self.DRAM_column:], profile))
        if profile:
            print(lst)
        return sum(lst)

    def Vector_Matrix_Mul(self, vector, matrix, profile=False):
        matrix_dim = matrix.shape[1]
        result = []
        for i in range(matrix_dim):
            result.append(self.Vector_Vector_Mul(vector, matrix[:, i], profile))
        return result
    
    def Vector_Matrix_Mul_multithreads(self, vector, matrix):
        assert vector.dim() == 1
        vector_dim = vector.shape[0]
        assert vector_dim == matrix.shape[0]
        matrix_dim = matrix.shape[1]
        matrix_parallel_n = (matrix_dim-1) // self.threads + 1
        pool = torch.multiprocessing.Pool(processes=self.threads)
        arguments = [(vector, matrix[:, i * matrix_parallel_n : (i+1) * matrix_parallel_n], False) for i in range(self.threads)]
        results = pool.starmap(self.Vector_Matrix_Mul, arguments)
        pool.close()
        pool.join()
        for i in range(len(results)):
            results[i] = torch.tensor(results[i])
        return torch.cat(results)


    def Vector_Vector_EWMUL_Row(self, A, B):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [A[i*self.burst_length:(i+1)*self.burst_length] *  
               B[i*self.burst_length:(i+1)*self.burst_length] for i in range(n-1)]
        lst.append(A[(n-1)*self.burst_length:] * B[(n-1)*self.burst_length:])
        return torch.cat(lst)

    def Vector_Vector_EWMUL(self, A, B):
        n = (A.shape[0] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_EWMUL_Row(
                A[i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[i*self.DRAM_column:(i+1)*self.DRAM_column]) for i in range(n-1)]
        lst.append(self.Vector_Vector_EWMUL_Row(A[(n-1)*self.DRAM_column:], B[(n-1)*self.DRAM_column:]))
        return torch.cat(lst)
    
    def EWADD(self, A, B):
        return A + B

    def Vector_Vector_EWADD_Row(self, A, B):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [self.EWADD(
                    A[i*self.burst_length:(i+1)*self.burst_length], 
                    B[i*self.burst_length:(i+1)*self.burst_length]) for i in range(n-1)]
        lst.append(self.EWADD(A[(n-1)*self.burst_length:], B[(n-1)*self.burst_length:]))
        return torch.cat(lst)

    def Vector_Vector_EWADD(self, A, B):
        if self.op_trace:
            self.file.write("AiM EWADD {} 0 0\n".format(A.shape[-1] // self.burst_length))
        n = (A.shape[-1] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_EWADD_Row(
                A[0][0][i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[0][0][i*self.DRAM_column:(i+1)*self.DRAM_column]) for i in range(n-1)]
        lst.append(self.Vector_Vector_EWADD_Row(A[0][0][(n-1)*self.DRAM_column:], B[0][0][(n-1)*self.DRAM_column:]))
        return torch.cat(lst).reshape(A.shape)
        
    def PERMUTE_GB(self, dimm, channel, utilized_channels, col_index, op_size, op_trace):
        """
        Simulates swapping adjacent pairs in the Global Buffer: [x1, x2] -> [-x2, x1]
        This assumes the Global Buffer has a simple shuffle network.
        """
        # 1. Update Timing Stats
        # Permutation is purely logic, usually fast (1-2 cycles). 
        # We can reuse a cheap timing constant or define a new one.
        PERMUTE_COST = 2 # cycles
        self.time["MAC_BK_GB"] += PERMUTE_COST # attributing to logic time
        
        if op_trace and dimm == 0:
            channel_multi_transformer_block_required = self.num_channels // utilized_channels * utilized_channels
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            # Write a new trace entry for debug
            self.file.write("AiM PERMUTE_GB {} {}\n".format(op_size, self.hex_channel_mask(channel_lst)))

        # 2. Perform the Swap Logic (Functional Simulation)
        # Assuming data is in GB. We need to handle the PyTorch tensor.
        gb_data = self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB
        
        # Extract the vector
        vector = gb_data[col_index : col_index + op_size * self.burst_length]
        
        # Reshape to pairs to handle the swap
        # Shape: (N/2, 2)
        pairs = vector.view(-1, 2)
        
        # Swap: [x, y] -> [-y, x]
        # column 0 gets -1 * column 1
        # column 1 gets column 0
        swapped = torch.stack((-1 * pairs[:, 1], pairs[:, 0]), dim=1).view(-1)
        
        # Write back to GB
        self.pim_device["dimm_" + str(dimm)].dimm["channel_" + str(channel)].GB[col_index : col_index + op_size * self.burst_length] = swapped
        
    def MULINT_only_trace(self, channel, op_size):
        """
        Executes the in-memory shuffle network (MIN) to pair (a,b) elements 
        into complex numbers directly within the Global Buffer.
        """
        # Track the time based on the base hardware latency plus the operation size
        self.time["MULINT"] += self.timing_constant["MULINT"] + op_size
        
        # Write the trace instruction. Format: "AiM MULINT <op_size> <channel_mask> <dummy_row>"
        self.file.write("AiM ISR_MIN {} {} 0\n".format(op_size, self.hex_channel_mask(channel)))
