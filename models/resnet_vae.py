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

# standard library imports
from collections import OrderedDict

class ResnetVAE(nn.Module):
    def __init__(self, resnet_version: str, latent_encoding_channels: int, skip_connection_type: str = 'concat'):
        super(ResnetVAE, self).__init__()

        self.resnet_versions = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if resnet_version not in self.resnet_versions:
            print('Unknown resnet version ', resnet_version)
            exit(1)
        self.resnet_version = resnet_version

        self.encoder = getattr(resnet, self.resnet_version)()
        self.decoder = getattr(resnet_transpose, self.resnet_version)(latent_encoding_channels, skip_connection_type)
        
        # ----------------------- Latent Mean ----------------------- #
        self.latent_mean_encoder = OrderedDict([
            ('latent_mean_conv', nn.Conv2d(
                in_channels=512 * self.decoder.block.expansion,
                out_channels=latent_encoding_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            )),

            ('batch_norm_latent_mean_conv', nn.BatchNorm2d(num_features=latent_encoding_channels)),

            ('activation_latent_mean_conv', nn.ReLU(inplace=True))
        ])

        # ----------------------- Latent logvariance ----------------------- #
        self.latent_logvariance_encoder = OrderedDict([
            ('latent_logvariance_conv', nn.Conv2d(
                in_channels=512 * self.decoder.block.expansion,
                out_channels=latent_encoding_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            )),

            ('batch_norm_latent_logvariance_conv', nn.BatchNorm2d(num_features=latent_encoding_channels)),

            ('activation_latent_logvariance_conv', nn.ReLU(inplace=True))
        ])
        
        # Set the layers as attributes so that cuda stuffs can be applied
        for layer_group in [self.latent_mean_encoder, self.latent_logvariance_encoder]:
            for layer_name, layer in layer_group.items():
                setattr(self, layer_name, layer)


    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_activations = self.encoder.layer_outputs
        
        # latent mean encoding
        x = encoder_output
        for layer_name, layer in self.latent_mean_encoder.items():
            x = layer(x)
        latent_mean_encoding = x

        # latent logvariance encoding
        x = encoder_output
        for layer_name, layer in self.latent_logvariance_encoder.items():
            x = layer(x)
        latent_logvariance_encoding = x
        
        encoder_activations['input'] = self.reparameterize(latent_mean_encoding, latent_logvariance_encoding) 

        decoder_output = self.decoder(encoder_activations)

        return decoder_output, latent_mean_encoding, latent_logvariance_encoding

    def reparameterize(self, mu: Variable, logvariance: Variable) -> Variable:
        """
        Randomly sample the latent vector given the mean and variance
        """
        std = logvariance.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import utils.constants

    import sys

    from models import resnet
    from utils.model_visualize import make_dot

    batch_size = 4
    latent_channels = 512

    input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)

    models = ['resnet18', 'resnet34', 'resnet50'] # , 'resnet101', 'resnet152']

    for model_name in models:
        print('model', model_name)
        for skip_connection_type in ['none', 'concat', 'add']:
            print('skip connection type', skip_connection_type)
            model = ResnetVAE(model_name, latent_channels, skip_connection_type)
            input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
            input = Variable(torch.empty(batch_size, 4, *input_size).uniform_(0, 1))

            for is_cuda in [False, True]:
                print('cuda', is_cuda)
                if is_cuda:
                    input = input.cuda()
                    model = model.cuda()
                else:
                    input = input.cpu()
                    model = model.cpu()

                output, mu, logvariance = model(input)
                print('output: ', output.size())

                dot = make_dot(output)
                dot.render("/tmp/model", '.')
                print('\n\n')
