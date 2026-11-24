"""
Transformer 全流程完整手搓【第二轮】（输入为一串数字，输出为其倒序）
使用PyTorch实现Transformer的训练阶段

为了易于理解：
- 使用 Post-LN
- dropout 暂时省略
- 暂时只考虑 <BOS>、<EOS> 两个特殊 Token，即使 <PAD>、<UNK> 也不考虑
- Encoder 和 Decoder 共用一张词表，注意力头数相同，max_len 相同
- Encoder 和 Decoder 的 d_model 相等
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderBlock(nn.Module):
    def __init__(self, d_model, H, d_ff):
        super().__init__()

        self.H = H
        self.d_k = d_model // H
        self.d_v = d_model // H

        # Multi-Head Self-Attention
        # 使用 self. 保存为对象属性，之后的forward()还得用d_k和d_v呢
        self.linear_Q = nn.Linear(d_model, H * self.d_k)
        self.linear_K = nn.Linear(d_model, H * self.d_k)
        self.linear_V = nn.Linear(d_model, H * self.d_v)
        self.linear_O = nn.Linear(H * self.d_v, d_model)

        # LayerNorm 1
        self.layer_norm_1 = nn.LayerNorm(d_model)

        # FFN
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

        # LayerNorm 2
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape

        # Multi-Head Self-Attention
        Q = self.linear_Q(x).reshape(B, N, self.H, self.d_k).transpose(1, 2)
        K = self.linear_K(x).reshape(B, N, self.H, self.d_k).transpose(1, 2)
        V = self.linear_V(x).reshape(B, N, self.H, self.d_v).transpose(1, 2)

        # Attention
        output_attn = self.linear_O(
            (torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1) @ V)
            .transpose(1, 2)
            .reshape(B, N, self.H * self.d_v)
        )

        # Residual + LayerNorm 1
        x_norm_1 = self.layer_norm_1(x + output_attn)

        # FFN
        ffn_output = self.linear_2(self.gelu(self.linear_1(x_norm_1)))

        # Residual + LayerNorm 2
        x_norm_2 = self.layer_norm_2(x_norm_1 + ffn_output)

        return x_norm_2


class Encoder(nn.Module):
    def __init__(self, d_model, H, d_ff, num_layers):
        super().__init__()

        # 堆叠多个Encoder Block
        # EncoderBlock(参数)就可以直接调用forward()方法
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, H, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, H, d_ff):
        super().__init__()

        self.H = H
        self.d_k = d_model // H
        self.d_v = d_model // H

        # Masked Multi-Head Self-Attention
        self.linear_Q_mask = nn.Linear(d_model, H * self.d_k)
        self.linear_K_mask = nn.Linear(d_model, H * self.d_k)
        self.linear_V_mask = nn.Linear(d_model, H * self.d_v)
        self.linear_O_mask = nn.Linear(H * self.d_v, d_model)

        # LayerNorm 1
        self.layer_norm_1 = nn.LayerNorm(d_model)

        # Multi-Head Cross-Attention
        self.linear_Q_corss = nn.Linear(d_model, H * self.d_k)
        self.linear_K_corss = nn.Linear(d_model, H * self.d_k)
        self.linear_V_corss = nn.Linear(d_model, H * self.d_v)
        self.linear_O_corss = nn.Linear(H * self.d_v, d_model)

        # LayerNorm 2
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # FFN
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

        # LayerNorm 3
        self.layer_norm_3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        B, M, D = x.shape
        N = encoder_output.shape[1]

        # Multi-Head Self-Attention
        Q_mask = self.linear_Q_mask(x).reshape(B, M, self.H, self.d_k).transpose(1, 2)
        K_mask = self.linear_K_mask(x).reshape(B, M, self.H, self.d_k).transpose(1, 2)
        V_mask = self.linear_V_mask(x).reshape(B, M, self.H, self.d_v).transpose(1, 2)

        # 计算 Attention scores_mask
        scores_mask = Q_mask @ K_mask.transpose(-1, -2) / math.sqrt(self.d_k)

        # Causal Mask
        # triu：来一个上三角矩阵（下三角全是0），diagonal=1:对角线也不要了，都设成0
        mask = torch.triu(
            torch.full(
                (M, M),
                float("-inf"),
                dtype=scores_mask.dtype,
                device=scores_mask.device,
            ),
            diagonal=1,
        )

        # Attention
        scores_mask += mask
        output_attn_mask = self.linear_O_mask(
            (torch.softmax(scores_mask, dim=-1) @ V_mask)
            .transpose(1, 2)
            .reshape(B, M, self.H * self.d_v)
        )

        # Residual + LayerNorm 1
        x = self.layer_norm_1(x + output_attn_mask)

        # Multi-Head Cross-Attention
        Q_corss = self.linear_Q_corss(x).reshape(B, M, self.H, self.d_k).transpose(1, 2)
        K_corss = (
            self.linear_K_corss(encoder_output)
            .reshape(B, N, self.H, self.d_k)
            .transpose(1, 2)
        )
        V_corss = (
            self.linear_V_corss(encoder_output)
            .reshape(B, N, self.H, self.d_v)
            .transpose(1, 2)
        )

        output_attn_corss = self.linear_O_corss(
            (
                torch.softmax(
                    Q_corss @ K_corss.transpose(-1, -2) / math.sqrt(self.d_k), dim=-1
                )
                @ V_corss
            )
            .transpose(1, 2)
            .reshape(B, M, self.H * self.d_v)
        )

        # Residual + LayerNorm 2
        x = self.layer_norm_2(x + output_attn_corss)

        # FFN
        output_ffn = self.linear_2(self.gelu(self.linear_1(x)))

        # Residual + LayerNorm 3
        x = self.layer_norm_3(x + output_ffn)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, H, d_ff, num_layers):
        super().__init__()

        # 堆叠多个 Decoder Block
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, H, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x


class Transformer(nn.Module):
    def __init__(self, V, H, d_model, d_ff, max_len, num_layers_en, num_layers_de):
        super().__init__()

        # Token Embedding
        self.embedding_token_en = nn.Embedding(V, d_model)
        self.embedding_token_de = nn.Embedding(V, d_model)

        # Positional Embedding
        self.embedding_position_en = nn.Embedding(max_len, d_model)
        self.embedding_position_de = nn.Embedding(max_len, d_model)

        # Encoder
        self.encoder = Encoder(d_model, H, d_ff, num_layers_en)

        # Decoder
        self.decoder = Decoder(d_model, H, d_ff, num_layers_de)

        # Output Projection
        self.linear_output = nn.Linear(d_model, V)

    def forward(self, source_token_ids, decoder_input_ids):
        