"""@package docstring
Fully convolutional autoencoder architecture with skip connections concatenating encoder/decoder features"""

# custom library imports
from utils import system_utils
import utils.constants

# pytorch imports
import torch
from torch import nn
from torch.autograd import Variable

# standard libray imports
from collections import OrderedDict
import typing

class ResidualFullyConvVAE(nn.Module):
    def __init__(self, input_size: typing.Tuple[int, int], latent_encoding_channels: int = 64, skip_connection_type: str = 'concat'):
        super(ResidualFullyConvVAE, self).__init__()
        self.input_size = input_size
        self.skip_connection_type = skip_connection_type
        if self.skip_connection_type not in ['concat', 'add']:
            self.skip_connection_type = 'concat'

        # if skip connection is concat, the channels will expand
        skip_connection_channel_expansion = 2 if self.skip_connection_type == 'concat' else 1

        self.layers = OrderedDict([
            ####################### Encoder ###########################
            ('conv1',       nn.Conv2d(in_channels=4,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)),

            ('batch_norm_conv1', nn.BatchNorm2d(num_features=32)),

            ('activation_conv1', nn.LeakyReLU()),

            ##################################################

            ('conv2',       nn.Conv2d(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)),

            ('batch_norm_conv2', nn.BatchNorm2d(num_features=64)),

            ('activation_conv2', nn.LeakyReLU()),

            ##################################################

            ('conv3',       nn.Conv2d(in_channels=64,
                                      out_channels=128,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)),

            ('batch_norm_conv3', nn.BatchNorm2d(num_features=128)),

            ('activation_conv3', nn.LeakyReLU()),

            ####################### Latent encoding ###########################

            ('latent_mean_conv',   nn.Conv2d(in_channels=128,
                                             out_channels=latent_encoding_channels,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1)),

            ('batch_norm_latent_mean_conv', nn.BatchNorm2d(num_features=latent_encoding_channels)),

            ('activation_latent_mean_conv', nn.LeakyReLU()),

            ##################################################

            ('latent_logvariance_conv', nn.Conv2d(in_channels=128,
                                                  out_channels=latent_encoding_channels,
                                                  kernel_size=(3, 3),
                                                  stride=(2, 2),
                                                  padding=1)),

            ('batch_norm_latent_logvariance_conv', nn.BatchNorm2d(num_features=latent_encoding_channels)),

            ('activation_latent_logvariance_conv', nn.LeakyReLU()),

            ####################### Decoder ###########################

            ('deconv4',             nn.ConvTranspose2d(in_channels=latent_encoding_channels,
                                                       out_channels=128,
                                                       kernel_size=(3, 3),
                                                       stride=(2, 2),
                                                       padding=1,
                                                       output_padding=1)),

            ('batch_norm_deconv4',   nn.BatchNorm2d(num_features=128)),

            ('activation_deconv4',  nn.LeakyReLU()),

            ##################################################

            ('deconv3',             nn.ConvTranspose2d(in_channels=128*skip_connection_channel_expansion,
                                                       out_channels=64,
                                                       kernel_size=(3, 3),
                                                       stride=(2, 2),
                                                       padding=1,
                                                       output_padding=1)),

            ('batch_norm_deconv3',   nn.BatchNorm2d(num_features=64)),

            ('activation_deconv3',  nn.LeakyReLU()),

            ##################################################

            ('deconv2', nn.ConvTranspose2d(in_channels=64*skip_connection_channel_expansion,
                                           out_channels=32,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1,
                                           output_padding=1)),

            ('batch_norm_deconv2',  nn.BatchNorm2d(num_features=32)),

            ('activation_deconv2', nn.LeakyReLU()),

            ##################################################

            ('deconv1', nn.ConvTranspose2d(in_channels=32*skip_connection_channel_expansion,
                                           out_channels=4,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1,
                                           output_padding=1)),

            ('batch_norm_deconv1',  nn.BatchNorm2d(num_features=4)),

            ('activation_deconv1', nn.LeakyReLU()),

            ##################################################

            ('deconv0', nn.ConvTranspose2d(in_channels=4*skip_connection_channel_expansion,
                                           out_channels=1,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=1,
                                           output_padding=0)),

            ('batch_norm_deconv0',  nn.BatchNorm2d(num_features=1)),

            ('activation_deconv0', nn.Sigmoid()),
        ])

        # input to the latent encoding
        self.latent_encoding_input_layer = 'activation_conv3'

        # Dictionary to tell which previous layer's output a layer takes as skip connection (i.e. mapping from input of current layer: ouput of previous layer)
        self.skip_connections = {
            'activation_deconv4': 'batch_norm_conv3',
            'activation_deconv3': 'batch_norm_conv2',
            'activation_deconv2': 'batch_norm_conv1',
            'activation_deconv1': 'input'
        }

        # Set the layers as attributes so that cuda stuffs can be applied
        for layer_name, layer in self.layers.items():
            setattr(self, layer_name, layer)

    def forward(self, x: Variable) -> typing.Tuple[Variable, Variable, Variable]:
        layer_outputs = {'input': x}
        layers_iterator = iter(self.layers.items())

        # encoder
        for layer_name, layer in layers_iterator:
            x = layer(x)
            layer_outputs[layer_name] = x

            if layer_name == self.latent_encoding_input_layer:
                break

        # latent mean encodings
        x = layer_outputs[self.latent_encoding_input_layer]
        for layer_name, layer in layers_iterator:
            x = layer(x)
            layer_outputs[layer_name] = x

            if layer_name == 'activation_latent_mean_conv':
                break

        # latent logvariance encoding
        x = layer_outputs[self.latent_encoding_input_layer]
        for layer_name, layer in layers_iterator:
            x = layer(x)
            layer_outputs[layer_name] = x

            if layer_name == 'activation_latent_logvariance_conv':
                break

        # sample from the latent space distribution
        x = self.reparameterize(
            layer_outputs['activation_latent_mean_conv'],
            layer_outputs['activation_latent_logvariance_conv']
        )

        # decoder
        for layer_name, layer in layers_iterator:
            if layer_name in self.skip_connections:
                # print(
                #         self.skip_connections[layer_name],
                #         layer_outputs[self.skip_connections[layer_name]].size(),
                #         x.size()
                # )
                if self.skip_connection_type == 'concat':
                    x = torch.cat([x, layer_outputs[self.skip_connections[layer_name]]], 1)
                elif self.skip_connection_type == 'add':
                    x = x + layer_outputs[self.skip_connections[layer_name]]

            # print(layer_name, x.size())
            x = layer(x)
            layer_outputs[layer_name] = x

        return x, layer_outputs['activation_latent_mean_conv'], layer_outputs['activation_latent_logvariance_conv']

    def reparameterize(self, mu: Variable, logvariance: Variable) -> Variable:
        """
        Randomly sample the latent vector given the mean and variance
        """
        std = logvariance.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


if __name__ == '__main__':
    batch_size = 64
    input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
    input = Variable(torch.FloatTensor(batch_size, 4, *input_size))

    for i in range(2):
        model = ResidualFullyConvVAE(input_size, latent_encoding_channels=64, skip_connection_type='concat')
        if i:
            input = input.cuda()
            model = model.cuda()
        output, latent_mu, latent_logvar = model(input)
        print('concat output:', output.size())

        model = ResidualFullyConvVAE(input_size, skip_connection_type='add')
        if i:
            model = model.cuda()
        output, latent_mu, latent_logvar = model(input)
        print('add output:', output.size())
