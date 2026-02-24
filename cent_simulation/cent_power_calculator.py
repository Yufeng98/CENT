import sys
import argparse
import math

KILO = 1000
MEGA = 1000000
GIGA = 1000000000
FREQ = 2.00 * GIGA

CH_PER_DV = 32.00

commands = [ "ACT",
            "PREA",
            "PRE",
            "RD",
            "WR",
            "RDA",
            "WRA",
            "REFab",
            "REFpb",
            "ACT4",
            "ACT16",
            "PRE4",
            "MAC",
            "MAC16",
            "AF16",
            "EWMUL16",
            "RDCP",
            "WRCP",
            "WRGB",
            "RDMAC16",
            "RDAF16",
            "WRMAC16",
            "WRA16",
            "TMOD",
            "SYNC",
            "EOC"]

isrs =  ["WR_SBK",
        "WR_GB",
        "WR_BIAS",
        "WR_AFLUT",
        "RD_MAC",
        "RD_AF",
        "RD_SBK",
        "COPY_BKGB",
        "COPY_GBBK",
        "MAC_SBK",
        "MAC_ABK",
        "AF",
        "EWMUL",
        "EWADD",
        "WR_ABK",
        "EOC",
        "SYNC"]

tRC = 44.5
tBL = 1.25
tCCDL = 1.0

# DRAM_POWER = {  "ACT_STBY": 415,
#                 "PRE_STBY": 317.5,
#                 "ACT": 93.9,
#                 "WR": 915,
#                 "RD": 525}

DRAM_POWER = {  "ACT_STBY": 527.5 / 2.00, # 415,
                "PRE_STBY": 366.3 / 2.00, # 317.5,
                "ACT": 132.6 / 2.00, # 93.9,
                "WR": 1106.3 / 2.00, # 915,
                "RD": 876.3 / 2.00} # 525}

# RED: Reduction Tree
# EXP: Exponent
# VEC: Vector Add
# SFT: Softmax
# CTR: PIMDispatcher + CXLController + Decoder + pim_ld_st + accel_controller
# DYN: Dynamic
# STT: Static
# Power in mW
# TODO: Change SFT
ACCEL_POWER = { "RED": {"SWITCH": 5.62E-02, "INT": 2.32E-01, "LEAK": 5.62E+03},
                "EXP": {"SWITCH": 3.39E-02, "INT": 5.60E-01, "LEAK": 1.28E+04},
                "VEC": {"SWITCH": 1.33E-01, "INT": 2.48E-01, "LEAK": 6.54E+03},
                "CTR": {"SWITCH": 3.18E-03 + 3.36E-03 + 2.89E-03 + 2.37E-01 + 1.30E-04, 
                        "INT": 2.49E-02 + 2.37E-01 + 4.51E-02 + 5.42E+00 + 8.77E-03, 
                        "LEAK": 2.87E+02 + 1.48E+03 + 2.94E+02 + 4.36E+04 + 8.78E+01},
                "RV": 3.96}   # 1 RV core
for accel_name in ["RED", "EXP", "VEC", "CTR"]:
    ACCEL_POWER[accel_name]["DYN"] = float(ACCEL_POWER[accel_name]["SWITCH"] + ACCEL_POWER[accel_name]["INT"])
    ACCEL_POWER[accel_name]["STT"] = float(ACCEL_POWER[accel_name]["LEAK"]) / float(GIGA)

RV_COUNT = 8
# Latency of 1 SIMD operation
SB_RD_CYCLE = 1.00
SB_WR_CYCLE = 1.00
EXP_LANE_CYCLE = 11.00
RV_RMSNorm_CYCLE = 26.00
RV_ROTEmbed_CYCLE = 3.00 / RV_COUNT
RV_SFT_CYCLE_PIPELINE = 16.00 * SB_WR_CYCLE + 2.00 / RV_COUNT + 1.00 * SB_RD_CYCLE
RV_SFT_CYCLE_SINGLE = 16.00 * SB_WR_CYCLE + 2.00 + 1.00 * SB_RD_CYCLE

# latency of pipelining 32 accelerators
# each having 16 SIMD lanes
ACCEL_CYCLE = { "EXP": CH_PER_DV * SB_RD_CYCLE + EXP_LANE_CYCLE + SB_WR_CYCLE,
                "VEC": CH_PER_DV * 2.00 * SB_RD_CYCLE + 1.00 + SB_WR_CYCLE,
                "COPY_BK_GB": 43,   # Cycles to copy Bank -> Global Buffer
                "COPY_GB_BK": 46,   # Cycles to copy Global Buffer -> Bank
                "MAC_BK_GB": 49,    # Cycles for MAC (Bank * Global Buffer)
                "PERMUTE_GB": 2     # Cycles for the shuffle logic
                }

# GB: Global Buffer
# SB: Shared Buffer
# STT: Static
# RD: Read
# WR: Write
SRAM_POWER = {  "GB": {"STT": 0.06702101898, "RD": 0.2785010052, "WR": 0.3254884575},
                "SB": {"STT": 0.6917736525, "RD": 3.207188769, "WR": 3.754155771},
                "IB": {"STT": 18.81731768, "RD": 70.13266856, "WR": 92.43730523}}

# TRX: Transaction Engine
# PHY: Physical Interface
# TODO: make sure PHY is not DQ
CTRL_POWER = {  "TRX": 267.7082056,
                "PHY": 381.0445262}
CH_PER_CTRL = 2.00

# pJ/bit
DQ_ENERGY = 5.5
PCIE_ENERGY = 4.4

WORD_SIZE = 256

def command_processor(stat_path):
    file = open(stat_path, 'r')
    lines = file.readlines()
    file.close()
    stat = {}
    for command in commands:
        stat[command] = 0.00
    for isr in isrs:
        stat[isr] = 0.00
    stat["cycles"] = 0.00
    stat["idle_cycles"] = 0.00
    stat["active_cycles"] = 0.00
    stat["precharged_cycles"] = 0.00
    
    for line in lines:
        words = line.split(' ')
        while len(words) > 0 and words[0] == "":
            words.pop(0)
        if words[0] == "Processing":
            if len(stat.keys) > 0:
                print("Error: multiple files in the same log")
                exit(1)
        if "memory_system_cycles" in words[0]:
            assert stat["cycles"] == 0
            stat["cycles"] = float(words[1])
        if "idle_cycles" in words[0]:
            stat["idle_cycles"] += float(words[1])
        if "active_cycles" in words[0]:
            stat["active_cycles"] += float(words[1])
        if "precharged_cycles" in words[0]:
            stat["precharged_cycles"] += float(words[1])
        for command in commands:
            if "num_" + command + "_commands" in words[0]:
                stat[command] += float(words[1])
        for isr in isrs:
            if "total_num_AiM_ISR_" + isr + "_requests" in words[0]:
                stat[isr] += float(words[1])
    stat["idle_cycles"] = stat["idle_cycles"] / CH_PER_DV
    stat["active_cycles"] = stat["active_cycles"] / CH_PER_DV
    stat["precharged_cycles"] = stat["precharged_cycles"] / CH_PER_DV
    # print(stat["cycles"], stat["idle_cycles"], stat["active_cycles"], stat["precharged_cycles"])
    # ms (average of all channels)
    stat["latency"] = stat["cycles"] * KILO / FREQ
    # ms (average of all channels)
    stat["active_latency"] = stat["active_cycles"] / CH_PER_DV * KILO / FREQ
    # ms (average of all channels)
    stat["precharged_latency"] = stat["precharged_cycles"] / CH_PER_DV * KILO / FREQ
    # % (average of all channels)
    if stat["cycles"] == 0:
        print(stat_path)
    stat["utilization"] = 100.00 - (stat["idle_cycles"] / CH_PER_DV / stat["cycles"]) * 100.00
    return stat

def power_calculator(stat, PCIE_bits, Head, HiddenDim, Tokens, GQA):
    energy = {}
    latency = {}
    # TODO: should we use the tRC or tRCD?
    energy["ACT/PRE"] = DRAM_POWER["ACT"] * (stat["ACT"] + 4.00 * stat["ACT4"] + 16.00 * stat["ACT16"]) * tRC / GIGA
    energy["RD"] = DRAM_POWER["RD"] * (stat["RDCP"] + stat["RD"] + stat["RDA"] + 16.00 * stat["AF16"] + stat["RDMAC16"] + stat["RDAF16"]) * tBL / GIGA
    energy["WR"] = DRAM_POWER["WR"] * (stat["WRCP"] + stat["WR"] + stat["WRA"] + stat["WRMAC16"] + 16.00 * stat["WRA16"]) * tBL / GIGA
    energy["PIM"] = 3 * DRAM_POWER["RD"] * (stat["MAC"] / 16.00 + stat["MAC16"] + stat["EWMUL16"] / 4.00) * tCCDL / GIGA
    energy["ACT_STBY"] = DRAM_POWER["ACT_STBY"] * CH_PER_DV * stat["active_latency"] / KILO
    energy["PRE_STBY"] = DRAM_POWER["PRE_STBY"] * CH_PER_DV * stat["precharged_latency"] / KILO
    energy["DQ"] = DQ_ENERGY * WORD_SIZE * (stat["RD"] + stat["WR"] + stat["RDA"] + stat["WRA"] + stat["WRGB"] + stat["RDMAC16"] + stat["RDAF16"] + stat["WRMAC16"] + stat["WRA16"]) / GIGA
    energy["PCIe"] = PCIE_bits * PCIE_ENERGY / GIGA

    ISR_COUNT = stat["RD"] + stat["WR"] + stat["RDA"] + stat["WRA"] + stat["MAC"] + stat["MAC16"] + stat["AF16"] + stat["EWMUL16"] + stat["RDCP"] + stat["WRCP"] + stat["WRGB"] + stat["RDMAC16"] + stat["RDAF16"] + stat["WRMAC16"] + stat["WRA16"]
    CMD_COUNT = sum(stat[x] for x in commands)
    energy["MEM_CTR"] = (CTRL_POWER["TRX"] * ISR_COUNT + CTRL_POWER["PHY"] * CMD_COUNT) / CH_PER_CTRL / FREQ

    GQA_factor = 1.00 + 1.00 / GQA
    latency["RMSNorm_latency"] =  HiddenDim / 16.00 / 16.00 / CH_PER_DV * ACCEL_CYCLE["VEC"]    # EMB /16.00 /16.00 ADD
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00                              # 1 RED
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE                                              # 1 RISCV
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] =  Tokens * Head / 16.00 / CH_PER_DV * ACCEL_CYCLE["EXP"]        # TOK*HEAD /16.00 EXP
    latency["Softmax_latency"] += Tokens * Head / 16.00 / CH_PER_DV * ACCEL_CYCLE["VEC"]        # TOK*HEAD /16.00 ADD
    latency["Softmax_latency"] += Head * 1.00 * SB_RD_CYCLE                                     # HEAD RED
    latency["Softmax_latency"] += Head * RV_SFT_CYCLE_PIPELINE                                  # HEAD RISCV
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = HiddenDim * RV_ROTEmbed_CYCLE                                 # EMB RISCV
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)

    # Static
    energy["GB_STT"] = stat["latency"] * SRAM_POWER["GB"]["STT"] * CH_PER_DV / KILO
    energy["SB_STT"] = stat["latency"] * SRAM_POWER["SB"]["STT"] / KILO
    energy["IB_STT"] = stat["latency"] * SRAM_POWER["IB"]["STT"] / KILO
    energy["RED_STT"] = stat["latency"] * ACCEL_POWER["RED"]["STT"] * CH_PER_DV / KILO
    energy["EXP_STT"] = stat["latency"] * ACCEL_POWER["EXP"]["STT"] * CH_PER_DV / KILO
    energy["VEC_STT"] = stat["latency"] * ACCEL_POWER["VEC"]["STT"] * CH_PER_DV / KILO

    # SRAM Power
    energy["GB_RD"] = SRAM_POWER["GB"]["RD"] * (stat["WRCP"]) / FREQ
    energy["GB_WR"] = SRAM_POWER["GB"]["WR"] * (stat["WRGB"] + stat["RDCP"]) / FREQ
    energy["SB_DYN"] = 2.00 * (HiddenDim / 16.00 / 16.00 + 1.00 + 1.00) * (2.00 * SRAM_POWER["SB"]["RD"] + 1.00 * SRAM_POWER["SB"]["WR"]) / FREQ    # RMSNorm: EMB /16.00 /16.00 ADD + 1 RED + 1 RV [First 16.00 is #SIMD lanes, Second is because of PU 16.00-to-1 MAC]
    energy["SB_DYN"] += ((Tokens * Head / 16.00 * 3 + Head * 2.00) * SRAM_POWER["SB"]["RD"]) / FREQ                                                 # Softmax: TOK * HEAD / 16.00 EXP and ADD + HEAD RED
    energy["SB_DYN"] += ((Tokens * Head / 16.00 * 2.00 + Head * 2.00) * SRAM_POWER["SB"]["WR"]) / FREQ                                              # Softmax: TOK * HEAD / 16.00 EXP and ADD + HEAD RED
    energy["SB_DYN"] += GQA_factor * HiddenDim / 16.00 * (SRAM_POWER["SB"]["RD"] + 2.00 * SRAM_POWER["SB"]["WR"]) / FREQ                            # RotEmbed: EMB /16.00 RV (1 ld + 2.00 st)
    
    ISR_COUNT = 0
    for isr in isrs:
        ISR_COUNT += stat[isr]
    ISR_COUNT += 2.00 * (HiddenDim / 16.00 / 16.00 + 2.00)              # RMSNorm
    ISR_COUNT += (Tokens * Head / 16.00 * 2.00 + Head * 2.00)           # Softmax
    ISR_COUNT += GQA_factor * HiddenDim                                 # RotEmbed
    energy["IB_DYN"] = ISR_COUNT * SRAM_POWER["IB"]["RD"] / FREQ

    # Accelerator Power
    energy["RV_DYN"] = 2.00 * RV_RMSNorm_CYCLE * ACCEL_POWER["RV"] / FREQ                       # RMSNorm: 1 RISCV
    energy["RV_DYN"] += Head * RV_SFT_CYCLE_SINGLE * ACCEL_POWER["RV"] / FREQ                   # Softmax: HEAD RISCV
    energy["RV_DYN"] += GQA_factor * HiddenDim * RV_ROTEmbed_CYCLE * ACCEL_POWER["RV"] / FREQ   # RotEmbed: EMB RISCV

    energy["RED_DYN"] = 2.00 * (1.00 * ACCEL_POWER["RED"]["DYN"]) / FREQ                    # 1 RED (RMSNorm)
    energy["RED_DYN"] +=(Head * ACCEL_POWER["RED"]["DYN"]) / FREQ                           # HEAD RED (Softmax)

    energy["EXP_DYN"] = (Tokens * Head / 16.00 * ACCEL_POWER["EXP"]["DYN"]) / FREQ          # TOK*HEAD /16.00 EXP (Softmax)

    energy["VEC_DYN"] = 2.00 * HiddenDim / 16.00 / 16.00 * ACCEL_POWER["VEC"]["DYN"] / FREQ     # EMB /16.00 /16.00 ADD (RMSNorm) [First 16.00 is #SIMD lanes, Second is because of PU 16.00-to-1 MAC]
    energy["VEC_DYN"] +=(Tokens * Head / 16.00 * ACCEL_POWER["VEC"]["DYN"]) / FREQ              # TOK*HEAD /16.00 ADD (Softmax)

    # We simply assume all the other components have a switching activity of 0.5
    energy["DV_CTR"] = stat["latency"] * 0.5 * (ACCEL_POWER["CTR"]["STT"] + ACCEL_POWER["CTR"]["DYN"]) / KILO

    return energy, latency

def get_args():
    parser = argparse.ArgumentParser(description="CENT Power Calculator")
    parser.add_argument("--mlog", help="path of the main ramulator log", type=str, required=True)
    parser.add_argument("--plog", help="path of the pim ramulator log (only if ch_per_bl > ch_per_dv)", type=str)
    parser.add_argument("--head", help="Number of heads", type=int, required=True)
    parser.add_argument("--hidden", help="Hidden dimension (embedding size)", type=int, required=True)
    parser.add_argument("--fc", help="FC layer embedding dimension", type=int, required=True)
    parser.add_argument("--token", help="Number of tokens", type=int, required=True)
    parser.add_argument("--block", help="Number of blocks", type=int, required=True)
    parser.add_argument("--ch_per_bl", help="Number of channels per block", type=int, required=True)
    parser.add_argument("--dv", help="Total number of devices (default = 32)", default=32, type=int)
    parser.add_argument("--ch_per_dv", help="Number of channels per device (default = 32)", default=32, type=int)
    parser.add_argument("--gqa", help="Factor of group query attention (default = 1)", default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    mlog = args.mlog
    plog = args.plog
    fc = args.fc
    head = args.head
    hidden = args.hidden
    token = args.token
    block = args.block
    CH_PER_BL = args.ch_per_bl
    DV = args.dv
    gqa = args.gqa

    if args.ch_per_dv != CH_PER_DV:
        CH_PER_DV = args.ch_per_dv
        # latency of pipelining 32 accelerators
        # each having 16 SIMD lanes
        ACCEL_CYCLE = { "EXP": CH_PER_DV * SB_RD_CYCLE + EXP_LANE_CYCLE + SB_WR_CYCLE,
                        "VEC": CH_PER_DV * 2.00 * SB_RD_CYCLE + 1.00 + SB_WR_CYCLE}

    energy_token = {}
    power_alldv = {}
    stat_main = command_processor(mlog)
    PCIE = hidden if CH_PER_BL <= CH_PER_DV else hidden * 10 + fc * 2.00
    energy_main, latency_main = power_calculator(stat_main, PCIE, head, hidden, token, gqa)

    total_ch_used = block * CH_PER_BL
    total_dv_need = 0
    if CH_PER_BL >= CH_PER_DV:
        DV_PER_BL = math.ceil(float(CH_PER_BL) / float(CH_PER_DV))
        assert DV_PER_BL >= 1.00
        PIPE_STAGES = DV / DV_PER_BL
        assert DV % DV_PER_BL == 0
        stat_pim = command_processor(plog)
        # print(stat_main)
        # print(stat_pim)
        energy_pim, latency_pim = power_calculator(stat_pim, PCIE, head, hidden, token, gqa)
        total_dv_need = block * DV_PER_BL
        for comp in energy_main.keys():
            energy_token[comp] = (energy_main[comp] + energy_pim[comp] * (DV_PER_BL - 1.00)) * block
            power_alldv[comp] = (energy_main[comp] + energy_pim[comp] * (DV_PER_BL - 1.00)) * PIPE_STAGES / stat_main["latency"]
    else:
        BL_PER_DV = int(CH_PER_DV / CH_PER_BL)
        total_dv_need = math.ceil(float(block) / float(BL_PER_DV))
        assert total_dv_need <= DV
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * total_dv_need
            power_alldv[comp] = energy_main[comp] * total_dv_need / stat_main["latency"]
        for comp in latency_main.keys():
            latency_main[comp] = latency_main[comp] * float(BL_PER_DV)
    total_ch_need = total_dv_need * CH_PER_DV

    # print("Configuration:")
    # print("CH/DV,CH-used,CH-needed,DV-needed")
    # print(f"{CH_PER_DV},{total_ch_used},{total_ch_need},{total_dv_need}")

    # print(",\nlatency (ms)")
    # print("pim,RMS,SFT,ROT,Total Acc,Total,utilization(%)")
    total_acc_latency = latency_main["RMSNorm_latency"] + latency_main["Softmax_latency"] + latency_main["RotEmbed_latency"]
    total_latency = stat_main["latency"] + latency_main["RMSNorm_latency"] + latency_main["Softmax_latency"] + latency_main["RotEmbed_latency"]
    print(f"{stat_main['latency']},{latency_main['RMSNorm_latency']},{latency_main['Softmax_latency']},{latency_main['RotEmbed_latency']},{total_acc_latency},{total_latency},{stat_main['utilization']}")
    print(total_acc_latency)

    print(",\nenergy 1 token detailed (mJ):")
    for comp in energy_token.keys():
        print(comp, end=",")
    print()
    for comp in energy_token.keys():
        print(energy_token[comp], end=",")
    print()

    # print(",\nenergy 1 token summary (mJ):") 
    # print("DRAM,ctrl,DQ,DV,PCIe,Total")
    # print(energy_token["ACT/PRE"] + energy_token["RD"] + energy_token["WR"] + energy_token["PIM"] + energy_token["ACT_STBY"] + energy_token["PRE_STBY"] + energy_token["GB_STT"] + energy_token["GB_RD"] + energy_token["GB_WR"], end=",")
    # print(energy_token["MEM_CTR"], end=",")
    # print(energy_token["DQ"], end=",")
    # print(energy_token["IB_STT"] + energy_token["SB_STT"] + energy_token["RED_STT"] + energy_token["EXP_STT"] + energy_token["VEC_STT"] + energy_token["IB_DYN"] + energy_token["SB_DYN"] + energy_token["RV_DYN"] + energy_token["RED_DYN"] + energy_token["EXP_DYN"] + energy_token["VEC_DYN"] + energy_token["DV_CTR"], end=",")
    # print(energy_token["PCIe"], end=",")
    total_energy = 0
    for comp in energy_token.keys():
        total_energy += energy_token[comp]
    print(total_energy)

    # print(",\nenergy 1 query summary (J):", total_energy * 4096 / 1000) 
    # print("energy 1 query summary (J):", total_energy * 4096 / 1000) 

    # print(",\npower all devices detailed (W):")
    # for comp in power_alldv.keys():
    #     print(comp, end=",")
    # print()
    # for comp in power_alldv.keys():
    #     print(power_alldv[comp], end=",")
    # print()

    # print(",\npower all devices summary (W):") 
    # print("DRAM,ctrl,DQ,DV,PCIe,Total")
    # print(power_alldv["ACT/PRE"] + power_alldv["RD"] + power_alldv["WR"] + power_alldv["PIM"] + power_alldv["ACT_STBY"] + power_alldv["PRE_STBY"] + power_alldv["GB_STT"] + power_alldv["GB_RD"] + power_alldv["GB_WR"], end=",")
    # print(power_alldv["MEM_CTR"], end=",")
    # print(power_alldv["DQ"], end=",")
    # print(power_alldv["IB_STT"] + power_alldv["SB_STT"] + power_alldv["RED_STT"] + power_alldv["EXP_STT"] + power_alldv["VEC_STT"] + power_alldv["IB_DYN"] + power_alldv["SB_DYN"] + power_alldv["RV_DYN"] + power_alldv["RED_DYN"] + power_alldv["EXP_DYN"] + power_alldv["VEC_DYN"] + power_alldv["DV_CTR"], end=",")
    # print(power_alldv["PCIe"], end=",")
    total_power = 0
    for comp in power_alldv.keys():
        total_power += power_alldv[comp]
    print(total_power)

    one_device = total_power/DV
    PIM_power = power_alldv["PIM"]/DV
    standby_power = (power_alldv["ACT_STBY"] + power_alldv["PRE_STBY"])/DV
    ACT_PRE_power = power_alldv["ACT/PRE"]/DV
    print("1 device", one_device, "PIM", PIM_power/one_device, "standby", standby_power/one_device, "ACT/PRE", ACT_PRE_power/one_device)

    # # Using write DRAM command
    # print(",\npower cap all devices (W):")
    # print("DRAM,ctrl,DQ,DV,PCIe,Total")
    # DRAM_power_cap = ((DRAM_POWER["ACT_STBY"] + DRAM_POWER["RD"] * 3 + SRAM_POWER["GB"]["STT"])) * DV * CH_PER_DV / KILO
    # ctrl_power_cap = (CTRL_POWER["TRX"] + CTRL_POWER["PHY"]) * DV * CH_PER_DV / CH_PER_CTRL / KILO
    # DQ_power_cap = (DQ_ENERGY * WORD_SIZE / tBL) * DV * CH_PER_DV / KILO
    # DV_power_cap = SRAM_POWER["IB"]["STT"] + SRAM_POWER["SB"]["STT"] 
    # DV_power_cap += (ACCEL_POWER["RED"]["STT"] + ACCEL_POWER["EXP"]["STT"] + ACCEL_POWER["VEC"]["STT"] + ACCEL_POWER["CTR"]["STT"]) * CH_PER_DV
    # DV_power_cap += SRAM_POWER["IB"]["WR"] + SRAM_POWER["SB"]["WR"]
    # DV_power_cap += (ACCEL_POWER["RED"]["DYN"] + ACCEL_POWER["EXP"]["DYN"] + ACCEL_POWER["VEC"]["DYN"] + ACCEL_POWER["CTR"]["DYN"]) * CH_PER_DV
    # DV_power_cap += ACCEL_POWER["RV"] * RV_COUNT
    # DV_power_cap = DV_power_cap * DV / KILO
    # PCIe_power_cap = (PCIE_ENERGY * WORD_SIZE / 1.00) * DV / KILO
    # print(DRAM_power_cap, end=",")
    # print(ctrl_power_cap, end=",")
    # # print(DQ_power_cap, end=",")
    # print(0, end=",")
    # print(DV_power_cap, end=",")
    # print(PCIe_power_cap, end=",")
    # # print(DRAM_power_cap + ctrl_power_cap + DQ_power_cap + DV_power_cap + PCIe_power_cap)
    # print(DRAM_power_cap + ctrl_power_cap + DV_power_cap + PCIe_power_cap)
