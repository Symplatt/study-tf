"""
Transformer 全流程完整手搓【第二轮】（输入为一串数字，输出为其倒序）
使用PyTorch实现

为了易于理解：
- 使用 Post-LN
- dropout 暂时省略
- 暂时只考虑 <BOS>、<EOS> 两个特殊 Token，即使 <PAD>、<UNK> 也不考虑
- Encoder 和 Decoder 共用一张词表，注意力头数相同，max_len 相同
- Encoder 和 Decoder 的 d_model 相等
"""

import torch

torch.manual_seed(42)

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

d_model = 64

# Encoder Token Embedding
embedding_token_en = torch.nn.Embedding(num_embeddings=V, embedding_dim=d_model).to(
    device
)

# Decoder Token Embedding
embedding_token_de = torch.nn.Embedding(num_embeddings=V, embedding_dim=d_model).to(
    device
)

# 查表
token_embedded_en = embedding_token_en(source_token_ids)
token_embedded_de = embedding_token_de(decoder_input_ids)
print(f"token id 嵌入：{token_embedded_en.shape}")

max_len = 1024

# Encoder Position Embedding
embedding_position_en = torch.nn.Embedding(
    num_embeddings=max_len, embedding_dim=d_model
).to(device)

# Decoder Position Embedding
embedding_position_de = torch.nn.Embedding(
    num_embeddings=max_len, embedding_dim=d_model
).to(device)

# Encoder和Decoder的位置ID
position_ids_en = torch.arange(
    source_token_ids.shape[1], dtype=torch.long, device=device
)
position_ids_de = torch.arange(
    target_token_ids.shape[1], dtype=torch.long, device=device
)

# 位置嵌入
# unsqueeze(0)为显示维度管理，只是为了让张量维度变化更明确，即使没有这一步，依然可以广播相加
position_embedded_en = embedding_position_en(position_ids_en)
position_embedded_de = embedding_position_de(position_ids_de)

# Encoder和Decoder的最终嵌入
input_embedded_en = token_embedded_en + position_embedded_en
input_embedded_de = token_embedded_de + position_embedded_de
