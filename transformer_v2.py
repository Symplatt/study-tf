"""
Transformer 全流程完整手搓【第二轮】
使用PyTorch实现

为了易于理解：
- 使用 Post-LN
- dropout 暂时省略
- 暂时只考虑 <BOS>、<EOS> 两个特殊 Token，即使 <PAD>、<UNK> 也不考虑
- Encoder 和 Decoder 共用一张词表，注意力头数相同，max_len 相同
"""
