"""@package docstring
custom loss functions"""

# pytorch imports
import torch
from torch.autograd import Variable

# standard library imports
import typing

def kl_divergence_loss(mu: typing.Type[Variable], logvariance: typing.Type[Variable]) -> typing.Type[Variable]:
    """
    Kullback–Leibler divergence loss (see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014)
    :param mu: mean of the latent vector
    :param logvariance: log of variance of the latent vector
    :return: the KL divergence loss
    """
    kld_loss = -0.5 * torch.sum(1 + logvariance - mu.pow(2) - logvariance.exp())
    return kld_loss