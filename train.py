
from dataset import SmilesDataset
import utils
import torch
import rnn
import vae
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-i', '--printInterval', type=int)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()
    model_type = args.model
    printInterval = args.printInterval
    device_type = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
    device = torch.device(device_type)
    tokenizer = utils.get_tokenizer()
    utils.logger.info(f'Device={device_type}')
    utils.logger.info(f'Tokens: {list(tokenizer.tokensDict.keys())}')
    smilesDataset = SmilesDataset(
        utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
    if model_type == 'rnn':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.collate_fn)
        rnn_model = rnn.RNN(
            **utils.config['rnn_param'], state_fname=utils.config['fname_rnn_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(rnn_model.named_parameters()):
            utils.logger.info(f'Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        optimizer = torch.optim.Adam(
            rnn_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95)
        rnn_model.trainModel(smilesDataloader, optimizer, scheduler,
                             utils.config['num_epoch'], utils.config['maxLength'], tokenizer, printInterval)
    elif model_type == 'vae':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.one_hot_collate_fn)
        vae_model = vae.VAE(**utils.config['vae_param'], encoder_state_fname=utils.config['fname_vae_encoder_parameters'],
                            decoder_state_fname=utils.config['fname_vae_decoder_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(vae_model.encoder.named_parameters()):
            utils.logger.info(f'Encoder Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        for layer_index, (name, layer) in enumerate(vae_model.decoder.named_parameters()):
            utils.logger.info(f'Decoder Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        encoderOptimizer = torch.optim.Adam(
            vae_model.encoder.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        decoderOptimizer = torch.optim.Adam(
            vae_model.decoder.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        encoderScheduler = torch.optim.lr_scheduler.StepLR(
            encoderOptimizer, step_size=1, gamma=0.95)
        decoderScheduler = torch.optim.lr_scheduler.StepLR(
            decoderOptimizer, step_size=1, gamma=0.95)
        vae_model.trainModel(smilesDataloader, encoderOptimizer, decoderOptimizer, encoderScheduler,
                             decoderScheduler, 1.0, utils.config['num_epoch'], tokenizer, printInterval)

if __name__ == '__main__':
    main()