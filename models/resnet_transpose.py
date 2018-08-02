import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


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

    def __init__(self, block, layers, input_channels):
        self.inplanes = 512
        self.block = block
        super(ResNet, self).__init__()

        self.input_upsample = None

        # upsamples the latent encoding to be same as encoder output (before encoding)
        self.input_upsample = nn.ConvTranspose2d(
            input_channels, self.inplanes * block.expansion, kernel_size=3, stride=2, padding=1, output_padding=1,
            bias=False
        )

        # these are in reverse order (during forward)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[1], stride=1)

        # upsamples the output to be same as conv1 of encoder (for max pooling)
        self.output_upsample = nn.ConvTranspose2d(
            64 * block.expansion, 64, kernel_size=1,
            bias=False
        )

        self.deconv1 = nn.ConvTranspose2d(64, 4, kernel_size=7, stride=2, padding=3, output_padding=1,
                                          bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)

        self.layer_outputs = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes * block.expansion, planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, output_padding=stride-1,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict['input']

        self.layer_outputs = {}

        if self.input_upsample:
            x = self.input_upsample(x)
            self.layer_outputs['input_upsample'] = x

        # TODO: skip connections from encoder
        x = self.layer4(x)
        self.layer_outputs['layer4'] = x

        x = self.layer3(x)
        self.layer_outputs['layer3'] = x

        x = self.layer2(x)
        self.layer_outputs['layer2'] = x

        x = self.layer1(x)
        self.layer_outputs['layer1'] = x

        x = self.output_upsample(x)
        self.layer_outputs['output_upsample'] = x

        x = self.maxunpool(x, input_dict['maxpool_indices'], output_size=(x.size(-2) * 2, x.size(-1) * 2))
        self.layer_outputs['maxunpool'] = x

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.layer_outputs['deconv1'] = x

        return x


def resnet18(input_channels):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels)

    return model


def resnet34(input_channels):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_channels)

    return model


def resnet50(input_channels):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_channels)

    return model


def resnet101(input_channels):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_channels)

    return model


def resnet152(input_channels):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], input_channels)

    return model


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import utils.constants

    import sys

    from models import resnet

    batch_size = 4
    latent_channels = 512

    input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)

    models = ['resnet18', 'resnet34', 'resnet50'] # , 'resnet101', 'resnet152']

    for model in models:
        encoder = getattr(resnet, model)()
        decoder = getattr(sys.modules[__name__], model)(latent_channels)

        input_size = (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
        input = Variable(torch.empty(batch_size, 4, *input_size).uniform_(0, 1))

        for is_cuda in [False]: # , True]:
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
