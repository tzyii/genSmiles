import torch
import utils
import time
from os import path


class RNN(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, hidden_dim, num_hidden, maxLength, state_fname, device):
        super().__init__()
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_embeddings = num_embeddings
        self.device = device
        # self.loccFn = torch.nn.CrossEntropyLoss()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx, device=self.device)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim,
                                num_hidden, batch_first=True, device=self.device)
        self.linear = torch.nn.Linear(
            hidden_dim, num_embeddings, device=self.device)

    def loadState(self):
        if path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)

    def forward(self, X):
        y = self.embedding(X.to(self.device))
        y, _ = self.rnn(y)
        y = self.linear(y)
        return y

    def sample(self, batch_size, temperature=1.0, eval=True):
        if eval:
            self.eval()
            nlls = None
        else:
            self.train()
            nlls = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=self.device)
        samples = torch.zeros((batch_size, self.maxLength), dtype=torch.long, device=self.device)
        X = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        hidden = None
        for i in range(self.maxLength):
            x = self.embedding(X.clone() if self.training else X)
            x, hidden = self.rnn(x, hidden)
            y = self.linear(x)
            if temperature != 1.0:
                y *= temperature
            y = torch.nn.functional.softmax(y, dim=2).squeeze(1)
            torch.multinomial(y, 1, out=X)
            samples[:, i] = X.data[:, 0]
            if self.training:
                nlls = nlls + torch.nn.functional.nll_loss(
                    torch.log(y), samples[:, i].clone(), reduction='none').view((-1, 1))
        return samples.cpu(), nlls

    def loss_per_sample(self, pred_y, y):
        return torch.nn.functional.cross_entropy(pred_y.to(self.device).transpose(1, 2), y.to(self.device), reduction='none').sum(dim=1).view((-1, 1))

    def trainModel(self, dataloader, optimizer, scheduler, nepoch, maxLength, tokenizer, printInterval):
        self.loadState()
        minloss = None
        numSample = 100
        for epoch in range(1, nepoch + 1):
            lossList, accumulatedLoss, numValid_list = [], 0, []
            for nbatch, (X, y) in enumerate(dataloader, 1):
                self.train()
                # pred_y = self(X)
                # loss = self.loccFn(pred_y.transpose(1, 2), y)
                pred_y = self(X)
                loss = self.loss_per_sample(pred_y, y).sum() / X.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossList.append(loss.item())
                accumulatedLoss += loss.item()
                if (nbatch == 1 or nbatch % printInterval == 0):
                    samples, _ = self.sample(100)
                    smilesStrs = tokenizer.getSmiles(samples)
                    numValid = sum([utils.isValidSmiles(sm)
                                   for sm in smilesStrs])
                    numValid_list.append(numValid)
                    print("[%s] Epoch %3d & Batch %4d: Loss= %.5e Valid= %3d/%3d" % (
                        time.ctime(), epoch, nbatch, sum(lossList) / len(lossList), numValid, numSample))
                    lossList.clear()
                    if minloss is None:
                        minloss = loss.item()
                    elif loss.item() < minloss:
                        self.saveState()
                        minloss = loss.item()
            scheduler.step()
            print("[%s] Epoch %3d: Loss= %.5e Valid= %3d/%3d" % (time.ctime(), epoch,
                  accumulatedLoss / nbatch, sum(numValid_list) / len(numValid_list), numSample))
