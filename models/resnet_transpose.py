import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=stride-1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(Bottleneck, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            inplanes * self.expansion, planes * self.expansion, kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes * self.expansion)
        self.deconv2 = nn.ConvTranspose2d(
            planes * self.expansion, planes * self.expansion,
            kernel_size=3, stride=stride,
            padding=1, output_padding=stride-1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.deconv3 = nn.ConvTranspose2d(
            planes * self.expansion, planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels, skip_connection_type: str = 'concat'):
        self.inplanes = 512
        self.block = block
        super(ResNet, self).__init__()

        self.skip_connection_type = skip_connection_type
        if self.skip_connection_type not in ['concat', 'add', 'none']:
            self.skip_connection_type = 'none'

        skip_connection_channel_expansion = 2 if self.skip_connection_type == 'concat' else 1

        self.input_upsample = None

        self.layers = OrderedDict([
            # upsamples the latent encoding to be same as encoder output (before encoding)
            ('input_upsample', nn.ConvTranspose2d(
                input_channels, self.inplanes * block.expansion, kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False
            )),

            # these are in reverse order (during forward)
            ('layer4', self._make_layer(block, 256, layers[3], stride=2)),
            ('layer3', self._make_layer(block, 128, layers[2], stride=2)),
            ('layer2', self._make_layer(block, 64, layers[1], stride=2)),
            ('layer1', self._make_layer(block, 64, layers[1], stride=1)),

            # upsamples the output to be same as conv1 of encoder (for max unpooling)
            ('output_upsample1', nn.ConvTranspose2d(
                64 * block.expansion, 64, kernel_size=1,
                bias=False
            )),

            ('output_upsample2', nn.ConvTranspose2d(
                64 * skip_connection_channel_expansion, 64, kernel_size=1,
                bias=False
            )),

            ('maxunpool', nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)),

            ('deconv1', nn.ConvTranspose2d(64, 4, kernel_size=7, stride=2, padding=3, output_padding=1,
                                           bias=False)),

            ('bn1', nn.BatchNorm2d(4)),
            ('relu1', nn.ReLU(inplace=True)),

            # final output deconv

            ('deconv2', nn.ConvTranspose2d(4, 1, kernel_size=1,
                                  bias=False)),
            ('bn2', nn.BatchNorm2d(1)),
            ('relu2', nn.ReLU(inplace=True))
        ])

        self.layer_outputs = {}

        # Dictionary to tell which previous layer's output a layer takes as skip connection (i.e. mapping from input of current layer: ouput of previous layer)
        self.skip_connections = {
            'layer4': 'layer4',
            'layer3': 'layer3',
            'layer2': 'layer2',
            'layer1': 'layer1',
            'output_upsample2': 'maxpool',
            # 'maxunpool': 'conv1'
        }

        # Set the layers as attributes so that cuda stuffs can be applied
        for layer_name, layer in self.layers.items():
            setattr(self, layer_name, layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        skip_connection_channel_expansion = 2 if self.skip_connection_type == 'concat' else 1

        upsample = None
        if stride != 1 or self.inplanes * skip_connection_channel_expansion != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes * block.expansion * skip_connection_channel_expansion,
                    planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, output_padding=stride-1,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes * skip_connection_channel_expansion, planes, stride, upsample))

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict['input']

        self.layer_outputs = {}

        for layer_name, layer in self.layers.items():
            if layer_name in self.skip_connections:
                # print(
                #         self.skip_connections[layer_name],
                #         layer_outputs[self.skip_connections[layer_name]].size(),
                #         x.size()
                # )
                if self.skip_connection_type == 'concat':
                    x = torch.cat([x, input_dict[self.skip_connections[layer_name]]], 1)
                elif self.skip_connection_type == 'add':
                    x = x + input_dict[self.skip_connections[layer_name]]

            kwargs = {}
            if layer_name == 'maxunpool':
                kwargs['indices'] = input_dict['maxpool_indices']
                kwargs['output_size'] = (x.size(-2) * 2, x.size(-1) * 2)

            x = layer(x, **kwargs)
            self.layer_outputs[layer_name] = x

        # if self.input_upsample:
        #     x = self.input_upsample(x)
        #     self.layer_outputs['input_upsample'] = x

        # x = self.layer4(x)
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['layer3']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['layer3']
        # self.layer_outputs['layer4'] = x
        #
        # x = self.layer3(x)
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['layer2']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['layer2']
        # self.layer_outputs['layer3'] = x
        #
        # x = self.layer2(x)
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['layer1']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['layer1']
        # self.layer_outputs['layer2'] = x
        #
        # x = self.layer1(x)
        # self.layer_outputs['layer1'] = x
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['layer1']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['layer1']
        #
        # # upsamples the output to be same as conv1 of encoder (for max unpooling)
        # x = self.output_upsample(x)
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['maxpool']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['maxpool']
        # self.layer_outputs['output_upsample'] = x
        #
        # x = self.maxunpool(x, input_dict['maxpool_indices'], output_size=(x.size(-2) * 2, x.size(-1) * 2))
        # if self.skip_connection_type == 'concat':
        #     x = torch.cat([x, input_dict['conv1']], dim=1)
        # elif self.skip_connection_type == 'add':
        #     x += input_dict['conv1']
        # self.layer_outputs['maxunpool'] = x
        #
        # x = self.deconv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # self.layer_outputs['deconv1'] = x

        return x


def resnet18(input_channels, skip_connection_type='none'):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels, skip_connection_type)

    return model


def resnet34(input_channels, skip_connection_type='none'):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_channels, skip_connection_type)

    return model


def resnet50(input_channels, skip_connection_type='none'):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_channels, skip_connection_type)

    return model


def resnet101(input_channels, skip_connection_type='none'):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_channels, skip_connection_type)

    return model


def resnet152(input_channels, skip_connection_type='none'):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], input_channels, skip_connection_type)

    return model


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

    for model in models:
        print('model', model)
        encoder = getattr(resnet, model)()
        for skip_connection_type in ['none', 'concat', 'add']:
            print('skip connection type', skip_connection_type)
            decoder = getattr(sys.modules[__name__], model)(latent_channels, skip_connection_type)

            input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
            input = Variable(torch.empty(batch_size, 4, *input_size).uniform_(0, 1))

            for is_cuda in [False]: # , True]:
                print('cuda', is_cuda)
                if is_cuda:
                    input = input.cuda()
                    encoder = encoder.cuda()
                    decoder = decoder.cuda()
                else:
                    input = input.cpu()
                    encoder = encoder.cpu()
                    decoder = decoder.cpu()

                encoder_output = encoder(input)
                encoder_activations = encoder.layer_outputs

                latent_encoding_layer = nn.Conv2d(
                    in_channels=512 * decoder.block.expansion,
                    out_channels=latent_channels,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1
                )

                latent_encoding_output = latent_encoding_layer(encoder_output)

                print('latent', latent_encoding_output.size())

                encoder_activations['input'] = latent_encoding_output

                decoder_output = decoder(encoder_activations)
                print('output: ', decoder_output.size())

                dot = make_dot(decoder_output)
                dot.render("/tmp/model", '.')
                print('\n\n')
