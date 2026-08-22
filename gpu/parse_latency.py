import sys

# parse prefill=512 decoding=3584 throughput
filename = sys.argv[1]
with open(filename, 'r') as f:
    lines = f.readlines()

prefill_latency = 0
decoding_latency = 0
prefill = True

for line in lines:
    if "Latency:" in line:
        if "Prefill: ()" in line:
            prefill = False
            decoding_latency += float(line.split()[3][:-1])
        # elif prefill:
        else:
            prefill_latency += float(line.split()[3][:-1])

print("Prefill latency: ", prefill_latency)
print("Decoding latency: ", decoding_latency)

batch = int(filename.split('_')[-6])
prefill_size = int(filename.split('_')[-4])
decoding_size = int(filename.split('_')[-2])
throughput = batch * (prefill_size + decoding_size) / (prefill_latency + decoding_latency) * 1000
print(throughput)
# print(decoding_latency / (prefill_latency + decoding_latency))