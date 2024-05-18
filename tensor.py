from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer
from einops import rearrange

'''Look out to the future but it tells you nothing so take another breath.  -Bastille, Icarus'''

class FFN(torch.nn.Module):
    def __init__(self, indims, hiddims):
        super().__init__()
        self.fc1 = torch.nn.Linear(indims, hiddims, bias=False)
        self.fc2 = torch.nn.Linear(hiddims//2, indims, bias=False)
        self.layer_norm = torch.nn.LayerNorm(indims)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x):
        res = x
        x = self.layer_norm(x)
        x1, x2 = torch.chunk(self.fc1(x), 2, dim=-1) 
        x = torch.nn.functional.gelu(x1) * x2 #GeGLU
        x = self.fc2(x)
        x = self.dropout(x)
        return x + res
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.norm(x)


class LSTM(torch.nn.Module):
    def __init__(self, indims, nheads, batch_first=True):
        super().__init__()
        self.nheads = nheads
        self.indims = indims
        self.lstm = torch.nn.LSTM(indims//nheads, indims//nheads, batch_first=batch_first, bias=False)
        self.r_proj = torch.nn.Linear(indims, indims, bias=False)
        self.q_proj = torch.nn.Linear(indims, indims, bias=False)
        self.out_proj = torch.nn.Linear(indims, indims, bias=False)
        self.layer_norm = torch.nn.LayerNorm(indims)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x, hidden=None):
        res = x
        x = self.layer_norm(x)
        q = self.q_proj(x)
        r = self.r_proj(x)
        if x.ndim == 3:
            q = rearrange(q, 'b s (h n) -> (b h) s n', h=self.nheads)
        else:
            q = rearrange(q, 'b (h n) -> (b h) n', h=self.nheads)
        lstm_out, hidden = self.lstm(q, hidden)
        
        if lstm_out.ndim == 3:
            lstm_out = rearrange(lstm_out, '(b h) s n -> b s (h n)', h=self.nheads)
        else:
            lstm_out = rearrange(lstm_out, '(b h) n -> b (h n)', h=self.nheads)
        
        lstm_out = lstm_out * torch.sigmoid(r)
        x = self.out_proj(lstm_out)
        x = self.dropout(x)
        return x + res, hidden


class Tensor(torch.nn.Module):
    def __init__(self, vocab_size, indims, nheads):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, indims)
        self.fc = torch.nn.Linear(indims, vocab_size)
        self.tensor_block = LSTM(indims, nheads)
        self.ffn = FFN(indims, indims*4)
        self.nheads = nheads
        self.pe = PositionalEncoding(indims)
    def forward(self, x, hiddens=None):
        x = self.embedding(x)
        x = self.pe(x)
        x, hiddens = self.tensor_block(x, hiddens)

        x = self.ffn(x)
        x = self.fc(x)
        return x, hiddens

class Trainer:
    def __init__(self, from_scratch=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = Tensor(self.tokenizer.vocab_size, 512, 8).to('mps')
        # self.model.load_state_dict(torch.load("model.pth"), strict=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True)['train']
        self.ntokens = 0

    def train(self):
        acc = 0
        self.dataset = self.dataset.shuffle()
        for f in self.dataset:
            loss = self.train_step(f['text'])
            if loss is None:
                continue # Skip this sample
            acc += 1
            print(f'Processed {acc} samples with loss {loss}, Total tokens: {self.ntokens}')
            if acc % 100 == 0:
                torch.save(self.model.state_dict(), "model.pth")
                print("Model saved")
            
    def train_step(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt")['input_ids']
        
        if input_ids.size(-1) < 2:
            return None
        
        if input_ids.size(-1) > 256:
            input_ids = input_ids[:, :256]

        present, future = input_ids[:, :-1], input_ids[:, 1:]
        self.optimizer.zero_grad()
        present = present.to('mps')
        future = future.to('mps')
        logits = self.model(present)[0]
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), future.view(-1))
        if loss > 1000:
            print("Loss too high, skipping")
            return None
        loss.backward()
        self.optimizer.step()
        self.ntokens = self.ntokens + present.size(-1)
        return loss.item()
    
    def train_dry(self):
        acc = 0
        for f in self.dataset:
            loss = self.train_step(f['text'])
            if loss is None:
                continue

            acc += 1
            print(f'Processed {acc} samples with loss {loss}, Total tokens: {self.ntokens}')
            if acc % 100 == 0:
                torch.save(self.model.state_dict(), "model.pth")
                print("Model saved")
            if acc == 1000:
                break

class Tester:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = Tensor(self.tokenizer.vocab_size, 512, 8).to('mps')
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()

    def generate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt")['input_ids']
        present = input_ids.to('mps')

        hidden = None

        for i in range(len(present[0])-1):
            # print(present[i])
            logits, hidden = self.model(present[:, i], hidden)
        
        for i in range(100):
            logits, hidden = self.model(present[:, -1], hidden)
            logits = logits / 0.4
            multinomial = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            present = torch.cat([present, multinomial], dim=1)
            if present[0, -1] == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(present[0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument('--test', action="store_true", default=False, help="Test the model")
    parser.add_argument('--train-dry-run', action="store_true", default=False, help="Train the model with a small portion of the dataset for testing.")

    args = parser.parse_args()
    if args.test:
        tester = Tester()
        print(tester.generate("Hello, my name is"))
        quit()
    trainer = Trainer(args.from_scratch)
    
    if args.train_dry_run:
        trainer.train_dry()
    else:
        trainer.train()