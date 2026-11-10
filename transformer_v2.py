"""
Transformer 全流程完整手搓【第二轮】
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
