from tokens import Tokenizer
import os
import utils
import rnn
import torch
import argparse


class RL(object):
    def __init__(self, num_epoch, batch_size, tokenizer, model_init_param, state_fname, device):
        self.num_epoch, self.batch_size, self.tokenizer, self.model_init_param, self.state_fname, self.device = num_epoch, batch_size, tokenizer, model_init_param, state_fname, device

    def create_model(self, freeze=False):
        model = rnn.RNN(**self.model_init_param,
                        state_fname=self.state_fname, device=self.device)
        model.loadState()
        if freeze:
            for i, (n, p) in enumerate(model.named_parameters()):
                if i < 1:
                    p.requires_grad_(False)
        return model

    def reinforce(self, score_funcion, state_fname):
        agent = self.create_model(freeze=True)
        agent.state_fname = state_fname
        agent.train()
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=1.0e-5, weight_decay=1.0e-4)
        for i in range(1, self.num_epoch + 1):
            num_wrong = 0
            samples, nlls = agent.sample(self.batch_size, eval=False)
            smiles = self.tokenizer.getSmiles(samples)
            scores = [score_funcion(sm) for sm in smiles]
            for j in range(len(scores)):
                if scores[j] is None:
                    scores[j] = 0.0
                    num_wrong += 1
            scores = torch.tensor(scores, device=self.device)
            loss = torch.sum(nlls * scores.view(nlls.shape)) / samples.shape[0]
            optimizer.zero_grad()
            loss.backward()
            print('Batch %5d: Loss= % .5e Mean_Score= % .3f Median_Score= % .3f Max_Score= % .3f Min_Score= % .3f Num_Wrong= %3d/%3d' %
                (i, loss.item(), scores.mean(), scores.median(), scores.max(), scores.min(), num_wrong, self.batch_size))
            optimizer.step()
        return agent

    def reinvent(self, score_funcion, state_fname, sigma=50):
        prior = self.create_model()
        prior.eval()
        agent = self.create_model(freeze=True)
        agent.state_fname = state_fname
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=1.0e-5, weight_decay=1.0e-4)
        for i in range(1, self.num_epoch + 1):
            num_wrong = 0
            samples, agent_nlls = agent.sample(self.batch_size, eval=False)
            smiles = self.tokenizer.getSmiles(samples)
            scores = [score_funcion(sm) for sm in smiles]
            rightScores = []
            rightSmilesIndex = []
            for s in scores:
                if s is None:
                    rightSmilesIndex.append(False)
                    num_wrong += 1
                else:
                    rightSmilesIndex.append(True)
                    rightScores.append(s)
            rightSmilesIndex = torch.tensor(rightSmilesIndex)
            scores = torch.tensor(rightScores, device=self.device)
            if (rightSmilesIndex.sum() < 1):
                continue
            rightSmiles = [smiles[i] for i in range(
                rightSmilesIndex.shape[0]) if rightSmilesIndex[i]]
            right_agent_nlls = agent_nlls[rightSmilesIndex, :]
            tokenVectors = self.tokenizer.tokenize(
                rightSmiles, useTokenDict=True)
            tokenVectors[0] = tokenVectors[0] + ['<pad>', ] * \
                (agent.maxLength - 1 - len(tokenVectors[0]))
            samplesInput = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(
                tokenVectors, addStart=True)], padding_value=self.tokenizer.getTokensNum('<pad>'), batch_first=True)
            prior_nlls = prior.loss_per_sample(
                prior(samplesInput), samples[rightSmilesIndex, :])
            loss = torch.pow(-prior_nlls.detach() + right_agent_nlls + sigma *
                             scores.view((-1, 1)), 2).sum() / samplesInput.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Batch %5d: Loss= % .5e Mean_Score= % .3f Median_Score= % .3f Max_Score= % .3f Min_Score= % .3f Num_Wrong= %3d/%3d' %
                  (i, loss.item(), scores.mean(), scores.median(), scores.max(), scores.min(), num_wrong, self.batch_size))
        return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str)
    parser.add_argument('-p', '--parameter', type=str)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()
    method = args.method
    parameter = args.parameter
    device_type = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
    device = torch.device(device_type)
    utils.logger.info(f'Device={device_type}')
    fname_param = os.path.join(os.path.dirname(
        utils.config['fname_rnn_parameters']), parameter)
    tokenizer = utils.get_tokenizer()
    similEvaluator = utils.SimilEvaluator(
        utils.config['fname_dataset'], utils.config['fname_fps'], utils.config['maxLength'])
    rl = RL(num_epoch=300, batch_size=512, tokenizer=tokenizer,
            model_init_param=utils.config['rnn_param'], state_fname=utils.config['fname_rnn_parameters'], device=device)
    agent = None
    if method == 'reinforce':
        agent = rl.reinforce(similEvaluator.getSimil, fname_param)
    elif method == 'reinvent':
        agent = rl.reinvent(similEvaluator.getSimil, fname_param)

    if agent is not None:
        utils.mkdir_multi(os.path.dirname(fname_param))
        agent.saveState()

if __name__ == '__main__':
    main()