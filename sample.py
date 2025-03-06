import os
import argparse
import tqdm
import utils
import rnn
import vae
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-p', '--parameter', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()
    model_type = args.model
    fname_state = args.parameter
    fname_output = args.output
    device_type = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
    device = torch.device(device_type)
    utils.logger.info(f'Device={device_type}')

    utils.mkdir_multi(utils.config['sampled_dir'])
    fname_output = os.path.join(utils.config['sampled_dir'], fname_output)

    smiles = []
    model = None
    tokenizer = utils.get_tokenizer()
    if model_type == 'rnn':
        fname_state = os.path.join(os.path.dirname(
            utils.config['fname_rnn_parameters']), fname_state)
        model = rnn.RNN(**utils.config['rnn_param'],
                        state_fname=fname_state, device=device)
        model.loadState()
    elif model_type == 'vae':
        model = vae.VAE(**utils.config['vae_param'], encoder_state_fname=utils.config['fname_vae_encoder_parameters'],
                        decoder_state_fname=utils.config['fname_vae_decoder_parameters'], device=device)
        model.encoder.loadState()
        model.decoder.loadState()

    if model is not None:
        for i in tqdm.tqdm(range(100)):
            samples, _ = model.sample(100)
            smiles.extend(tokenizer.getSmiles(samples))
        utils.mkdir_multi(os.path.dirname(fname_output))
        with open(fname_output, 'w') as f:
            smiles.append('')
            f.write('\n'.join(smiles))

if __name__ == '__main__':
    main()
