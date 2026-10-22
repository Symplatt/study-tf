"""
Transformer训练阶段全流程完整手搓【第一轮】
因为不会 PyTorch，先用基础 Python 语法实现。

为了易于理解：
- 使用Post-LN
- dropout 暂时省略
- 暂时只考虑 <BOS>、<EOS> 两个特殊 Token，即使 <PAD>、<UNK> 也不考虑
- Encoder和Decoder共用一张词表，注意力头数相同，max_len相同，
- 每次循环重新 randn 生成参数，不考虑实际反向传播情况
"""

import torch
import math
import torch.nn.functional as F

V = 12345
BOS_ID = V - 2
EOS_ID = V - 1
B = 24
N = 28
M = 36
H = 8
max_len = 1024
d_model_en = 256
d_model_de = 288
d_k_en = d_v_en = d_model_en // H
d_k_de = d_v_de = d_model_de // H
eps = 1e-5
d_ff_en = d_model_en * 4

# 超参数
num_encoder_layers = 6
num_decoder_layers = 8

# ================================== Encoder ==================================

# Token Embedding
origin_token_ids_en = torch.randint(low=0, high=V, size=(B, N))
weight_embedding_token_en = torch.randn(V, d_model_en)
# Positional Embedding
position_ids_en = torch.arange(N)
weight_embedding_position_en = torch.randn(max_len, d_model_en)

# 查表，最终输入 = Token Embedding + Positional Embedding → [B,N,D_en]
input_embedded_en = (
    weight_embedding_token_en[origin_token_ids_en]
    + weight_embedding_position_en[position_ids_en]
)

num = 0  # BLOCK 循环次数
x_en = input_embedded_en  # Encoder Block 循环的初始输入
while num < num_encoder_layers:
    num += 1

    # ————————— Attention —————————

    # 四大天王
    weight_Q_en = torch.randn(d_model_en, H * d_k_en)  # 东方持国天王，哈！ —— Q：Query
    weight_K_en = torch.randn(d_model_en, H * d_k_en)  # 南方增长天王，嘿！ —— K：Key
    weight_V_en = torch.randn(d_model_en, H * d_v_en)  # 西方广目天王，嚯！ —— V：Value
    weight_O_en = torch.randn(H * d_v_en, d_model_en)  # 北方多闻天王，哼！ —— O：Output

    # 上吧，KQV！[B, N, H * d_k]
    Q_en = x_en @ weight_Q_en
    K_en = x_en @ weight_K_en
    V_en = x_en @ weight_V_en

    # 拆成多头 [B, H, N, d_k]
    Q_en = Q_en.reshape(B, N, H, d_k_en).transpose(1, 2)
    K_en = K_en.reshape(B, N, H, d_k_en).transpose(1, 2)
    V_en = V_en.reshape(B, N, H, d_v_en).transpose(1, 2)

    # 1. 注意力分数 [B, H, N, N]
    scores_en = (Q_en @ K_en.transpose(-2, -1)) / math.sqrt(d_k_en)
    # 2. 注意力权重 [B, H, N, N]
    attn_weights_en = torch.softmax(scores_en, dim=-1)
    # 3. 每个头的注意力输出 [B, H, N, d_v]
    attn_output_en = attn_weights_en @ V_en
    # 4. 拼接多头 [B, N, H * d_v]
    concat_en = attn_output_en.transpose(1, 2).reshape(B, N, H * d_v_en)
    # 5. 输出投影 [B, N, D_en]
    output_en = concat_en @ weight_O_en

    # 残差连接
    residual_en_1 = x_en + output_en

    # ————————— 第一次 LayerNorm —————————

    # 生成 LN 的权重矩阵 [D]
    gamma_ln_en_1 = torch.ones(d_model_en)
    beta_ln_en_1 = torch.zeros(d_model_en)

    # 计算每个 Token 的均值与方差 [B,N,1]
    mean_ln_en_1 = residual_en_1.mean(dim=-1, keepdim=True)
    var_ln_en_1 = residual_en_1.var(dim=-1, keepdim=True, unbiased=False)

    # 计算 LN [B,N,D_en]
    normalized_ln_en_1 = (residual_en_1 - mean_ln_en_1) / torch.sqrt(var_ln_en_1 + eps)
    x_norm_en_1 = normalized_ln_en_1 * gamma_ln_en_1 + beta_ln_en_1

    # ————————— FFN —————————

    # 第一次线性变换 [B,N,D_ff]
    linear_A_en_1 = torch.randn(d_model_en, d_ff_en)
    bias_en_1 = torch.zeros(d_ff_en)  # 每列加一个bias，而不是每行
    x_linear_en_1 = x_norm_en_1 @ linear_A_en_1 + bias_en_1

    # 激活函数 [B,N,D_ff]
    x_GELU_en = F.gelu(x_linear_en_1)

    # 第二次线性变换 [B,N,D]
    linear_A_en_2 = torch.randn(d_ff_en, d_model_en)
    bias_en_2 = torch.zeros(d_model_en)
    x_linear_en_2 = x_GELU_en @ linear_A_en_2 + bias_en_2

    # 残差连接 [B,N,D]
    residual_en_2 = x_norm_en_1 + x_linear_en_2

    # ————————— 第二次 layernorm —————————

    # 生成 LN 的权重矩阵 [D]
    gamma_ln_en_2 = torch.ones(d_model_en)
    beta_ln_en_2 = torch.zeros(d_model_en)

    # 计算每个 Token 的均值与方差 [B,N,1]
    mean_ln_en_2 = residual_en_2.mean(dim=-1, keepdim=True)
    var_ln_en_2 = residual_en_2.var(dim=-1, keepdim=True, unbiased=False)

    # 懒得写了，直接二合一
    x_norm_en_2 = (
        gamma_ln_en_2 * (residual_en_2 - mean_ln_en_2) / torch.sqrt(var_ln_en_2 + eps)
        + beta_ln_en_2
    )

    # 赋值 x_en，用于下一次 while 循环
    x_en = x_norm_en_2

# Encoder最终输出
final_output_en = x_en

# ================================== Decoder ==================================

# 最初的Decoder输入 [B,M-1]
origin_token_ids_de = torch.randint(0, V, (B, M - 1))

bos = torch.full((B, 1), BOS_ID)
eos = torch.full((B, 1), EOS_ID)

# 输入/输出 Token IDs [B,M]
input_token_ids_de = torch.cat([bos, origin_token_ids_de], dim=1)
target_token_ids_de = torch.cat(
    [origin_token_ids_de, eos], dim=-1
)  # dim -1 和 1在此处完全等价

# Decoder の embedding
weight_embedding_token_de = torch.randn(V, d_model_de)
position_ids_de = torch.arange(M)
weight_embedding_position_de = torch.randn(max_len, d_model_de)
input_embedded_de = (
    weight_embedding_token_de[input_token_ids_de]
    + weight_embedding_position_de[position_ids_de]
)

num = 0  # BLOCK 循环次数
x_de = input_embedded_de  # Decoder Block 循环的初始输入

while num < num_decoder_layers:
    num += 1

    # ————————— 掩码多头自注意 —————————

    # QKVO权重
    weight_Q_de = torch.randn(d_model_de, H * d_k_de)
    weight_K_de = torch.randn(d_model_de, H * d_k_de)
    weight_V_de = torch.randn(d_model_de, H * d_v_de)
    weight_O_de = torch.randn(H * d_v_de, d_model_de)

    # 计算QKV [B, H, M, d_k/d_v]
    Q_de = (x_de @ weight_Q_de).reshape(B, M, H, d_k_de).transpose(1, 2)
    K_de = (x_de @ weight_K_de).reshape(B, M, H, d_k_de).transpose(1, 2)
    V_de = (x_de @ weight_V_de).reshape(B, M, H, d_v_de).transpose(1, 2)

    # 掩码前的计算 [B, H, M, M]
    scores_de = (Q_de @ K_de.transpose(-1, -2)) / math.sqrt(d_k_de)

    # 你的掩码。不，是你的掩码。 [B, H, M, M]
    mask = torch.triu(torch.full((M, M), float("-inf")), diagonal=1)
    scores_de += mask

    # 之后的Attention该咋算咋算，一模一样
    attn_output_de = torch.softmax(scores_de, dim=-1) @ V_de
    concat_de = attn_output_de.transpose(1, 2).reshape(B, M, H * d_v_de)
    output_de = concat_de @ weight_O_de

    # 残差连接 [B,M,D]
    residual_de_1 = x_de + output_de

    # ————————— Decoder 第一次 LayerNorm —————————

    # 那一堆参数
    gamma_ln_de_1 = torch.ones(d_model_de)  # [D]
    beta_ln_de_1 = torch.zeros(d_model_de)  # [D]
    mean_ln_de_1 = residual_de_1.mean(dim=-1, keepdim=True)  # [B,M,1]
    var_ln_de_1 = residual_de_1.var(dim=-1, keepdim=True)  # [B,M,1]

    # 算算算
    norm_1_de = (
        gamma_ln_de_1 * (residual_de_1 - mean_ln_de_1) / torch.sqrt(var_ln_de_1 + eps)
        + beta_ln_de_1
    )
