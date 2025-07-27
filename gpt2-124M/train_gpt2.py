from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length / context
    vocab_size: int = 50257 # number of tokens : 50k BPE + 256 byte tokens + 1 EOF
    n_layer: int = 12 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # check divisibility between embedding dim 
        assert config.n_embd % config.n_head == 0
        # keys, queries, and values hence 3
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 

        # output projections
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1) # (B, nh, T, T)
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # wrap it back together
        y = self.c_proj(y)
        return y  # (B, T, C)



class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing scheme
        # tie the input and output embeddings
        self.transformer.wte.weight = self.lm_head.weight
        # initialize weights
        self.apply(self.__init_weights)

    def __init_weights(self, module) :
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer ) ** -0.5 # scale init
            # initialize weights with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize embeddings with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward sequence of length %d, block size is only %d" % (T, self.config.block_size)

        # forward the GPT model
        pos = torch.arange(0, T, dtype = torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # (B*T, vocab_size) vs (B*T)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium'
                              , 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt : %s" % model_type)
        config_args = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embd=768), #124M params
            'gpt2-medium': dict(n_layer = 24, n_head = 16, n_embd = 1024) #350M params
          , 'gpt2-xl': dict(n_layer = 48, n_head = 20, n_embd = 1600), #1.5B params
            'gpt2-large': dict(n_layer = 36, n_head = 20, n_embd = 1280) #774M params
            }
        config = config_args[model_type].copy()
        config = GPTConfig(**config)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask buffer

        # init hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy weights from hugging face model to our model
        sd_keys_hf = sd_hf.keys()

        # ignore the mask buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups: any parameters that is >= 2D will be weight decaryed, otherwise no
        # ie. all weight tensors in matmuls + embeddings decay, , all biases and layernorms do not decay

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameters tensors: {len(decay_params)} with total {num_decay_params:,} params")
        print(f"num no decayed parameters tensors: {len(no_decay_params)} with total {num_no_decay_params:,} params")

        # create adamw optimzer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f"using fused adamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# -------------------------------------------------------

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long) # (N, )
        print("loaded %d tokens from input.txt" % len(self.tokens))
        print("1 epoch = %d batches" % (len(self.tokens) // (B * T)))

        # state
        self.current_position = 0
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0
        return x, y


# Implementing a learning rate decay
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    #1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    #2) if it > warmup_steps, return min learning rate
    if it > max_steps:
        return min_lr
    #3 in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff start at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#model.eval()
#device = 'cpu'
import time
device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
    torch.mps.manual_seed(1337)
torch.manual_seed(1337)

total_batch_size = 524288 # 2 ^ 19 , ~0.5M
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "total batch size must be divisible by B * T"
grad_accumulation_steps = total_batch_size // (B * T) 
print("total desired batch_size: ", total_batch_size)
print("==> calculated grad_accumulation_steps: ", grad_accumulation_steps)

train_loader = DataLoaderLite(B=B, T=T) 

# enable tf-32, highest is available
torch.set_float32_matmul_precision('high')
print("Using device: ", device)
model = GPT(GPTConfig(vocab_size=50304)) # Override to have nice numbers from 50257 to 50304.
#Running pretrained models -  GPT.from_pretrained('gpt2')

if device != 'mps':
    # enable torch compile for cuda and cpu
    # this will use triton for faster training
    print("Compiling model with torch.compile()")
    model = torch.compile(model) # triton failures update torch - pip install --upgrade torch
model = model.to(device) #TODO move it up and check?
           
model.train()

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = max_lr, device = device)
print("number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
print("number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accumulation_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device) # (B, T)
        #Use bfloat16 incase CUDA support it otherwise use float32 (trying float 16, even though its said to use gradient scalers)
        if device == 'cuda':
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        # scale the loss for gradient accumulation (V.IMPORTANT while using gradient accumulation)
        # because the gradient just add on each successive backward
        # addition of gradient corresponds to SUM in the objective, but
        # instead of SUM we mean mean. Scale the loss here so it comes out right
        loss = loss / grad_accumulation_steps 
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients - inplace
    lr = get_lr(step)
    # override the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.time()
    dt = (t1 - t0)
    token_processed = train_loader.B * train_loader.T * grad_accumulation_steps 
    token_per_sec = token_processed / dt
    print(f"Step {step} | loss: {loss_accum.item(): .6f} | lr: {lr:.4e}  | norm: {norm: .4f} | dt: {dt * 1000:.2f}ms | tokens per second: {token_per_sec:.2f} tokens/sec")

import sys; sys.exit(0)
#model = GPT.from_pretrained('gpt2').to(device)
model.eval()
num_return_sequences = 5
max_length = 64
enc = tiktoken.get_encoding("gpt2")
x = enc.encode("Hello, i am a language model ")
x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
x = torch.repeat_interleave(x, repeats=num_return_sequences, dim=0) # (B, T)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x) # (B, T, vocab_size)

        # take the last token logits
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # get the probs
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        # print probs
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (B, 50, vocab_size)
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)

        # gather the tokens
        xcol = torch.gather(topk_indices, dim=-1, index=ix) # (B, 1)
        
        # append the next tokens to the input
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("> ", decoded)