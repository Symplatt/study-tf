"""
Transformer 全流程完整手搓【第二轮】（输入为一串数字，输出为其倒序）
使用PyTorch实现

为了易于理解：
- 使用 Post-LN
- dropout 暂时省略
- 暂时只考虑 <BOS>、<EOS> 两个特殊 Token，即使 <PAD>、<UNK> 也不考虑
- Encoder 和 Decoder 共用一张词表，注意力头数相同，max_len 相同
"""

import torch

torch.manual_seed(42)  # XXX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary
V = 12

BOS_ID = 10
EOS_ID = 11

# Encoder 输入 [B, N]
source_token_ids = torch.tensor(
    [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 3, 5, 7],
        [2, 4, 6, 8],
        [9, 8, 7, 6],
        [3, 6, 9, 2],
        [5, 1, 4, 8],
        [7, 2, 5, 9],
    ],
    dtype=torch.long,
    device=device,
)

# 目标内容：将输入序列逆序
target_content_ids = torch.flip(source_token_ids, dims=[1])

# BOS & EOS
B = source_token_ids.shape[0]

bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
eos = torch.full((B, 1), EOS_ID, dtype=torch.long, device=device)

# Decoder 输入 [B,M]
decoder_input_ids = torch.cat([bos, target_content_ids], dim=-1)

# 训练标签 [B, M]
target_token_ids = torch.cat([target_content_ids, eos], dim=-1)

print("source_token_ids:", source_token_ids.shape)
print("target_content_ids:", target_content_ids.shape)
print("decoder_input_ids:", decoder_input_ids.shape)
print("target_token_ids:", target_token_ids.shape)

print("\n第一条训练数据：")
print("Encoder 输入:", source_token_ids[0])
print("Decoder 输入:", decoder_input_ids[0])
print("训练标签:", target_token_ids[0])
