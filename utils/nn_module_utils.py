"""@package docstring
Utilities for pytorch nn
"""
import torch
from torch import nn
from torch.autograd import Variable

import typing
import math
import sys
import os
import shutil

class Flatten(nn.Module):
    """
    Flattens the input to be a vector
    """
    def forward(self, input: Variable) -> Variable:
        """

        :param input: autograd.Variable/Tensor of size batch_size x <any number of dimensions>
        :return: autograd.Variable or Tensor of size batch_size x <appropriate size of only 1 dimension>
        """
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    """
    Unflattens the input to be desired shape
    """
    def __init__(self, target_size: typing.Tuple[int, ...]):
        """

        :param target_size: target size to reshape the input
        """
        super(Unflatten, self).__init__()
        self.target_size = target_size

    def forward(self, input: Variable) -> Variable:
        """

        :param input: autograd.Variable/Tensor of size batch_size x <any number of dimensions>
        :return: autograd.Variable or Tensor of size batch_size x <target dimensions>
        """
        return input.view(input.size(0), *(self.target_size))

class Threshold(nn.Module):
    """
    Threshold an input differentiably
    """

    def __init__(self, threshold: float):
        super(Threshold, self).__init__()
        self.threshold = threshold

        # self.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0], grad_i[1] * 0.01))
        # self.ReLU = nn.ReLU(True)

    def forward(self, x: typing.Type[Variable]):
        return (x >= self.threshold).type_as(x)

        # return self.ReLU(x + self.threshold) - self.threshold
        # return self.ReLU(x) + self.threshold

def get_output_feature_size(input_size: typing.Tuple[int, int], layers: typing.List[nn.Module]) \
        -> typing.Tuple[int, int]:
    """

    :param input_size: size (height, width) of the input image to the network
    :param layers: iterable over the layers (conv, pool or batch norm) to be applied
    :return: the size of the output feature after going through the layers
    """
    output_feature = input_size
    for layer in layers:
        if hasattr(layer, "kernel_size") and hasattr(layer, "stride"):
            kernel_size = layer.kernel_size if type(layer.kernel_size) == tuple \
                else (layer.kernel_size, layer.kernel_size)
            padding = layer.padding if type(layer.padding) == tuple \
                else (layer.padding, layer.padding)
            stride = layer.stride if type(layer.stride) == tuple \
                else (layer.stride, layer.stride)

            output_feature = tuple([
                int(math.floor(
                    (output_feature[i] - kernel_size[i] + 2 * padding[i]) / stride[i] + 1
                )) for i in range(2)
            ])

    return output_feature
    
class CheckPointSaver:
    def __init__(self, metrics: typing.List[str], checkpoint_dir: str='checkpoints', best_metrics: typing.Union[None, float]=None):
        """
        :param metrics: metric names (list)
        :param best_metrics: values of the best metric
        """
        self.best_metric_values = best_metrics
        self.metric_names = metrics
        self.best_model_checkpoint_dir = checkpoint_dir
        os.system('mkdir -p {}'.format(self.best_model_checkpoint_dir))

        if best_metrics is not None:
            if len(metrics) != len(best_metrics):
                sys.exit('Mismatch len of metric names and values')
        else:
            self.best_metric_values = [None for _ in range(len(self.metric_names))]

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        checkpoint_file = os.path.join(self.best_model_checkpoint_dir, filename)
        torch.save(state, checkpoint_file)

        for i in range(len(self.metric_names)):
            metric_name = self.metric_names[i]
            best_metric_value = self.best_metric_values[i]
            
            if metric_name in state:
                is_best = False
                if best_metric_value is None:
                    is_best = True
                elif 'loss' in metric_name:
                    is_best = state[metric_name] < best_metric_value
                elif 'acc' in metric_name:
                    is_best = state[metric_name] > best_metric_value

                if is_best:
                    shutil.copyfile(checkpoint_file, os.path.join(self.best_model_checkpoint_dir, 'model_best_' + metric_name + '.pth.tar'))
                    self.best_metric_values[i] = state[metric_name]

def get_deconv_output_size(input_size: typing.Tuple[int, int], layers: typing.List[nn.Module]) -> typing.Tuple[int, int]:
    """

    :param input_size: size (height, width) of the input to the network
    :param layers: iterable over the deconv (convTranspose2d) layers to be applied
    :return: the size after applying the deconvs
    """

    output_size = input_size

    for layer in layers:
        if isinstance(layer, nn.ConvTranspose2d):
            kernel_size = layer.kernel_size if type(layer.kernel_size) == tuple \
                else (layer.kernel_size, layer.kernel_size)
            padding = layer.padding if type(layer.padding) == tuple \
                else (layer.padding, layer.padding)
            stride = layer.stride if type(layer.stride) == tuple \
                else (layer.stride, layer.stride)
            output_padding = layer.output_padding if type(layer.output_padding) == tuple \
                else (layer.output_padding, layer.output_padding)

            output_size = tuple([
                int(math.ceil(
                    (output_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i],

                )) for i in range(2)
            ])

    return output_size

def get_deconv_stride(input_size: typing.Tuple[int, int],
                      output_size: typing.Tuple[int, int],
                      kernel_size: typing.Union[typing.Tuple[int, int], int],
                      padding: typing.Union[typing.Tuple[int, int], int],
                      output_padding: typing.Union[typing.Tuple[int, int], int]) -> typing.Tuple[int, int]:
    """

    :param input_size: size (height, width) of the input of the deconv layer
    :param output_size: size (height, width) of the output of the deconv layer
    :param kernel_size: 2D (de)convolution kernel size
    :param padding: 2D padding
    :return: the required number of strides
    """

    kernel_size = kernel_size if type(kernel_size) == tuple \
        else (kernel_size, kernel_size)
    padding = padding if type(padding) == tuple \
        else (padding, padding)
    output_padding = output_padding if type(output_padding) == tuple \
        else (output_padding, output_padding)

    stride = tuple([
        int(math.floor(
            (output_size[i] + 2 * padding[i] - kernel_size[i] - output_padding[i]) / (input_size[i] - 1)
        )) for i in range(2)
    ])

    return stride
