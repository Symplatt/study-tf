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
print(f"current device:{device}")

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

B = source_token_ids.shape[0]

bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
eos = torch.full((B, 1), EOS_ID, dtype=torch.long, device=device)

# Decoder 输入 [B,M]
decoder_input_ids = torch.cat([bos, target_content_ids], dim=-1)

# 训练标签 [B, M]
target_token_ids = torch.cat([target_content_ids, eos], dim=-1)


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
        B, N, _ = x.shape  # BMD中的D用不上

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
        self.linear_Q_cross = nn.Linear(d_model, H * self.d_k)
        self.linear_K_cross = nn.Linear(d_model, H * self.d_k)
        self.linear_V_cross = nn.Linear(d_model, H * self.d_v)
        self.linear_O_cross = nn.Linear(H * self.d_v, d_model)

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
        scores_mask = scores_mask + mask
        output_attn_mask = self.linear_O_mask(
            (torch.softmax(scores_mask, dim=-1) @ V_mask)
            .transpose(1, 2)
            .reshape(B, M, self.H * self.d_v)
        )

        # Residual + LayerNorm 1
        x = self.layer_norm_1(x + output_attn_mask)

        # Multi-Head Cross-Attention
        Q_cross = self.linear_Q_cross(x).reshape(B, M, self.H, self.d_k).transpose(1, 2)
        K_cross = (
            self.linear_K_cross(encoder_output)
            .reshape(B, N, self.H, self.d_k)
            .transpose(1, 2)
        )
        V_cross = (
            self.linear_V_cross(encoder_output)
            .reshape(B, N, self.H, self.d_v)
            .transpose(1, 2)
        )

        output_attn_cross = self.linear_O_cross(
            (
                torch.softmax(
                    Q_cross @ K_cross.transpose(-1, -2) / math.sqrt(self.d_k), dim=-1
                )
                @ V_cross
            )
            .transpose(1, 2)
            .reshape(B, M, self.H * self.d_v)
        )

        # Residual + LayerNorm 2
        x = self.layer_norm_2(x + output_attn_cross)

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

        N = source_token_ids.shape[1]
        M = decoder_input_ids.shape[1]

        position_ids_en = torch.arange(N, device=source_token_ids.device)
        position_ids_de = torch.arange(M, device=decoder_input_ids.device)

        # Encoder Embedding
        input_embedded_en = self.embedding_token_en(
            source_token_ids
        ) + self.embedding_position_en(position_ids_en).unsqueeze(0)

        # Decoder Embedding
        input_embedded_de = self.embedding_token_de(
            decoder_input_ids
        ) + self.embedding_position_de(position_ids_de).unsqueeze(0)

        # Encoder
        encoder_output = self.encoder(input_embedded_en)

        # Decoder
        decoder_output = self.decoder(input_embedded_de, encoder_output)

        # Output Projection
        logits = self.linear_output(decoder_output)

        return logits


# 实例化
model = Transformer(
    V=V, H=8, d_model=64, d_ff=256, max_len=1024, num_layers_en=6, num_layers_de=6
).to(device)


# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 100

model.train()

for epoch in range(num_epochs):

    # 清空梯度
    optimizer.zero_grad()

    # 前向传播
    logits = model(source_token_ids, decoder_input_ids)

    # Loss
    loss = loss_fn(logits.reshape(-1, V), target_token_ids.reshape(-1))

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印 Loss
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d} | " f"Loss: {loss.item():.4f}")
