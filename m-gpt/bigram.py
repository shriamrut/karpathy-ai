import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size  = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#---------------------------------------

print(f"Device {device}")
torch.manual_seed(13337)

#Post downloading the dataset
with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[c] for c in s])

# split into train and test
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data)-block_size, 
                       (batch_size,1))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B,T,C  = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, 1) # B,C
            idx_next = torch.multinomial(probs, num_samples=1) # B,1
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample batch
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
