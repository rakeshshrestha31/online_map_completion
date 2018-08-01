"""
@package docstring
VAE using resnet (without average pooling and FC) as encoder and corresponding transposed resnet as decoder
"""

# custom imports
from models import resnet
from models import resnet_transpose

import utils.constants

# pytorch imports
import torch
from torch import nn
from torch.autograd import Variable

class ResnetVAE(nn.Module):
    def __init__(self, resnet_version: str):
        super(ResnetVAE, self).__init__()

        self.resnet_versions = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if resnet_version not in self.resnet_versions:
            print('Unknown resnet version ', resnet_version)
            exit(1)
        self.resnet_version = resnet_version

        self.encoder = getattr(resnet, self.resnet_version)
        self.decoder = getattr(resnet_transpose, self.resnet_version)

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_activations = self.encoder.layer_outputs

        encoder_activations['input'] = encoder_output

        # TODO: latent encoding
        decoder_output = self.decoder(encoder_activations)

        return decoder_output

if __name__ == '__main__':
    batch_size = 64
    input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
    input = Variable(torch.FloatTensor(batch_size, 4, *input_size))