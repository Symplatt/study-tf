"""
Transformer全流程完整手搓

- 使用Post-LN
- dropout 暂时省略
- 暂时忽略Encoder embedding 时的 padding
- 暂不考虑反向传播
"""

import torch
import math
import torch.nn.functional as F

V = 12345
max_len = 1024
B = 24
N = 28
d_model_en = 256
H = 8
d_k_en = d_v_en = d_model_en // H
eps = 1e-5
d_ff_en = d_model_en * 4 # 通常如此

# 超参数
num_encoder_layers = 6
num_decoder_layers = 8


# ————————— embedding —————————
token_ids_en = torch.randint(low=0, high=V, size=(B, N))
weight_embedding_token_en = torch.randn(V, d_model_en)  # [V, D_en]

position_ids_en = torch.arange(N)  # 递增生成1维张量，内容为 [0,1,2,3……N-1]
weight_embedding_position_en = torch.randn(max_len, d_model_en)  # [max_len,D_en] 到时候用广播和token权重参数相加

# 查表
input_embedded_en = (
    weight_embedding_token_en[token_ids_en]
    + weight_embedding_position_en[position_ids_en]
)  # [B,N,D_en]

num = 0 # BLOCK 循环次数
x_en = input_embedded_en # 上一轮的最终结果
while(num < num_encoder_layers):
    num += 1
    
    # ————————— multi-head-self-attention —————————
    # 四大天王
    weight_Q_en = torch.randn(d_model_en, H * d_k_en)  # 东方持国天王 —— Q：Query
    weight_K_en = torch.randn(d_model_en, H * d_k_en)  # 南方增长天王 —— K：Key
    weight_V_en = torch.randn(d_model_en, H * d_v_en)  # 西方广目天王 —— V：Value
    weight_O_en = torch.randn(H * d_v_en, d_model_en)  # 北方多闻天王 —— O：Output

    # 上吧，KQV！
    Q_en = x_en @ weight_Q_en
    K_en = x_en @ weight_K_en
    V_en = x_en @ weight_V_en

    # 拆成多头
    Q_en = Q_en.reshape(B, N, H, d_k_en).transpose(1, 2)  # Q_en: [B, H, N, d_k]
    K_en = K_en.reshape(B, N, H, d_k_en).transpose(1, 2)  # K_en: [B, H, N, d_k]
    V_en = V_en.reshape(B, N, H, d_v_en).transpose(1, 2)  # V_en: [B, H, N, d_v]

    # Scaled Dot-Product Attention

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

    residual_en_1 = x_en + output_en # 残差连接

    # ————————— 第一次 LayerNorm —————————

    gamma_ln_en_1 = torch.ones(d_model_en)
    beta_ln_en_1 = torch.zeros(d_model_en)

    mean_ln_en_1 = residual_en_1.mean(dim=-1, keepdim=True)  # [B,N,1]
    var_ln_en_1 = residual_en_1.var(
        dim=-1, keepdim=True, unbiased=False
    )  # [B,N,1]

    normalized_ln_en_1 = (residual_en_1 - mean_ln_en_1) / torch.sqrt(
        var_ln_en_1 + eps
    )  # [B,N,D_en]

    x_norm_en_1 = normalized_ln_en_1 * gamma_ln_en_1 + beta_ln_en_1  # [B,N,D_en]

    # ————————— FFN —————————
    linear_A_en_1 = torch.randn(d_model_en, d_ff_en) # [D_en,D_ff]
    bias_en_1 = torch.zeros(d_ff_en) # [D_ff]

    x_linear_en_1 = x_norm_en_1 @ linear_A_en_1 + bias_en_1  # [B,N,D_ff]

    x_GELU_en = F.gelu(x_linear_en_1) 

    linear_A_en_2 = torch.randn(d_ff_en, d_model_en)
    bias_en_2 = torch.zeros(d_model_en)

    x_linear_en_2 = x_GELU_en @ linear_A_en_2 + bias_en_2

    residual_en_2 = x_norm_en_1 + x_linear_en_2 # [B,N,D_en]

    # ————————— 第二次 layernorm —————————

    gamma_ln_en_2 = torch.ones(d_model_en)
    beta_ln_en_2 = torch.zeros(d_model_en)

    mean_ln_en_2 = residual_en_2.mean(dim=-1, keepdim=True) # [B,N,1]
    var_ln_en_2 = residual_en_2.var(dim=-1, keepdim=True, unbiased=False) # [B,N,1]

    x_norm_en_2 = gamma_ln_en_2 * (residual_en_2 - mean_ln_en_2) / torch.sqrt(var_ln_en_2 + eps) + beta_ln_en_2 # 懒了，直接二合一

    x_en = x_norm_en_2

final_output_en = x_en
