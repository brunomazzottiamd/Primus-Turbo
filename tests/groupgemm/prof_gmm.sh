#!/usr/bin/env bash

rm -rf gmm_agent_info.csv gmm_kernel_trace.csv

rocprofv3 \
    --kernel-trace --truncate-kernels \
    --output-format csv --output-file gmm \
    -- python test_gmm.py &> /dev/null

python<<EOF
import pandas as pd
df = pd.read_csv("gmm_kernel_trace.csv", usecols=["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
kernel_filter = (df["Kernel_Name"] == "_grouped_bf16_persistent_gemm_kernel") | (df["Kernel_Name"] == "gmm_kernel")
df = df[kernel_filter]
df["Time_ms"] = (df["End_Timestamp"] - df["Start_Timestamp"]) * 1e-6
df = df[["Kernel_Name", "Time_ms"]]
df["Kernel_Name"] = df["Kernel_Name"].replace({"_grouped_bf16_persistent_gemm_kernel": "Primus-Turbo", "gmm_kernel": "AITER"})
print(df.to_string(index=False))
EOF

rm -rf gmm_agent_info.csv gmm_kernel_trace.csv
