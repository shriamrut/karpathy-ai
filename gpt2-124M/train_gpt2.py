from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, 2)
        ''' 
         How dimensions turn out here in the above steps?
         x => B,T,C
            qkv = (B, T, C) @ (C, 3 * C) = (B, T, C) @ (C, 3 * C) = (B, T, C) @ (B, C, 3 * C)
                = (B, T, 3 * C)
            q, k, v = qkv.split(C, 2)
            q => (B, T, C)
            k => (B, T, C)
            v => (B, T, C)
        '''
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        ''' k => (B, T, C) 
            k = k.view(B, T, self.n_head, C // self.n_head)
            (B, T, C) = (B, T, n_h, hs)
            k = k.transpose(1,2)
            (B, T, n_h, hs) = (B, n_h, T, hs)
        '''
        # attention score 
        att = (q @ k.tranpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        '''
            q = (B, n_h, T, hs)
            k = (B, n_h, T, hs)

            k.T(-2, -1) = (B, n_h, hs, T)

            q @ k.T(-2, -1) = (B, n_h, T, hs) @ (B, n_h, hs, T)
                            = (B, n_h, T, T)
            # last dimension
            k.size(-1) = hs
        '''
         # eliminate the forward attentions scores
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        '''
        (B, n_h, T, T) @ (B, n_h, T, hs)
        => (B, n_h, T, hs)
        '''
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50k BPE merge + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension or channel dimension 

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2': dict(n_layer=12, n_head = 12, n_embd = 768), # 124 M params
            'gpt2-medium': dict(n_layer=12, n_head = 16, n_embd = 1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head = 20, n_embd = 1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head = 25, n_embd = 1600) #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPU model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys {len(sd_keys_hf)} != {len(sd_keys)}"
        print(f'loading weights from pretrained gpt: {model_type}')
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos_emb) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = pos_emb + tok_emb # (T, n_embd) + (B, T, n_embd) = (B, T, n_embd) + (B, T, n_embd) = (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) For each token, what is the next token thats coming in?
        return logits        
# --------------------------------------------------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30
device = 'cuda' if torch.cuda.is_available else 'cpu'
model = GPT.from_pretrained('gpt2')
model.eval()
model = model.to(device)