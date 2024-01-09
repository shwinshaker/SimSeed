#!./env python

import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['Hooker', 'spec_norm', 'spec_norm_linear', 'OutputHooker']

def spec_norm(weight, input_dim):
    # exact solution by svd and fft
    assert len(input_dim) == 2
    assert len(weight.shape) == 4
    fft_coeff = np.fft.fft2(weight, input_dim, axes=[2, 3])
    D = np.linalg.svd(fft_coeff.T, compute_uv=False, full_matrices=False)
    return np.max(D)

def spec_norm_linear(weight):
    assert len(weight.shape) == 2
    D = np.linalg.svd(weight, compute_uv=False, full_matrices=False)
    return np.max(D)


class OutputHooker:
    """
        Simple hooker that gets the output of a module
    """
    def __init__(self, name, module, device=None):
        self.name = name
        self.module = module
        self.device = device

        # extraction protocol
        self.hooker = module.register_forward_hook(self.hook)

        # ease pycharm complain
        self.input = None
        self.output = None

    def hook(self, module, input, output):
        self.output = output

    def unhook(self):
        self.hooker.remove()


class Hooker:
    """
        Calculate the Lipschitz constant of each node
        hook on single module, e.g. conv, linear, bn
            maximum singular value (conv, fc)
            alpha / running_var (bn)
    """

    def __init__(self, name, module, device=None, n_power_iterations=10):

        # name it
        class_name = module._get_name()
        self.name = name
        self.module = module
        self.device = device
        self.n_power_iterations = n_power_iterations

        # lip calculation function
        if class_name.startswith('Conv'):
            self.lip = self.__conv_lip
        elif class_name.startswith('Linear'):
            self.lip = self.__fc_lip
        elif class_name.startswith('BatchNorm'):
            self.lip = self.__bn_lip
        else:
            raise KeyError('')

        # extraction protocol
        self.hooker = module.register_forward_hook(self.hook)

        # ease pycharm complain
        self.input = None
        self.output = None

    def hook(self, module, input, output):
        self.input = input
        self.output = output

    def unhook(self):
        self.hooker.remove()
        self.__remove_buffers()

    def __fc_lip(self):
        buffers = dict(self.module.named_buffers())
        if 'u' not in buffers:
            assert 'v' not in buffers
            self.__init_buffers(self.input[0].size(), self.output.size())

        v_ = self.__get_buffer('v')
        u_ = self.__get_buffer('u')
        weight = self.__get_parameter('weight')

        v, u = v_.clone().to(self.device), u_.clone().to(self.device)
        for _ in range(self.n_power_iterations):
            u = F.linear(v, weight, bias=None)
            u = self.__normalize(u)
            v = F.linear(u, weight.transpose(0, 1), bias=None)
            v = self.__normalize(v)
            
        sigma = torch.norm(F.linear(v, weight, bias=None).view(-1))
        # comparison with exact solution
        # print('%s - specnorm_iter: %.4f - specnorm_svd: %.4f' % (self.name, sigma.item(), spec_norm_linear(weight.cpu().detach().numpy())))

        # modify buffer - because tensor are copied for every operation, needs to modify the memory
        v_.copy_(v)
        u_.copy_(u)

        return sigma

    def __conv_lip(self):
        # only when needed, i.e. after the entire validation batch, do power iteration and compute spectral norm, to gain efficiency

        buffers = dict(self.module.named_buffers())
        if 'u' not in buffers:
            assert 'v' not in buffers
            # assert 'sigma' not in buffers
            self.__init_buffers(self.input[0].size(), self.output.size())

        # get buffer
        v_ = self.__get_buffer('v')
        u_ = self.__get_buffer('u')
        # sigma_ = self.__get_buffer('sigma')

        # get weight
        weight = self.__get_parameter('weight')
        stride = self.module.stride
        padding = self.module.padding

        # power iteration
        # v, u, sigma = v_.clone().to(self.device), \
        #               u_.clone().to(self.device), \
        #               sigma_.clone().to(self.device)
        v, u = v_.clone().to(self.device), \
               u_.clone().to(self.device)

        """
            The output of deconvolution may not be exactly same as its convolution counterpart
            dimension lost when using stride > 1
            that's why need additional output padding
            See:
                https://towardsdatascience.com/is-the-transposed-convolution-layer-and-convolution-layer-the-same-thing-8655b751c3a1
                http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        """
        transpose_dim = stride[-1] * (u.size()[-1]-1) + weight.size()[-1] - 2 * padding[-1]
        output_padding = v.size()[-1] - transpose_dim
        for _ in range(self.n_power_iterations):
            u = F.conv2d(v, weight, stride=stride, padding=padding, bias=None)
            u = self.__normalize(u)
            v = F.conv_transpose2d(u, weight, stride=stride, padding=padding, output_padding=output_padding)
            v = self.__normalize(v)
            
        sigma = torch.norm(F.conv2d(v, weight, stride=stride, padding=padding, bias=None).view(-1))
        # comparison with exact solution
        # print('%s - specnorm_iter: %.4f - specnorm_svd: %.4f' % (self.name, sigma.item(), spec_norm(weight.cpu().detach().numpy(), u.size()[2:])))

        # modify buffer - because tensor are copied for every operation, needs to modify the memory
        v_.copy_(v)
        u_.copy_(u)
        # sigma_.copy_(sigma)

        return sigma

    def __init_buffers(self, input_dim, output_dim):
        # replace the first batch dim by 1
        v_dim = (1, *input_dim[1:]) 
        u_dim = (1, *output_dim[1:])
        # print(self.name, v_dim, u_dim) # should be (1, 16, 32, 32) and (1, 16, 32, 32) for the first one

        v = self.__normalize(torch.randn(v_dim))
        u = self.__normalize(torch.randn(u_dim))

        self.module.register_buffer('v', v)
        self.module.register_buffer('u', u)
        # self.module.register_buffer('sigma', torch.ones(1))

    def __remove_buffers(self):
        pass
        # delattr(self.module, 'v')
        # delattr(self.module, 'u')
        # delattr(self.module, 'sigma')

    def __bn_lip(self):
        # buffers = dict(self.module.named_buffers())
        # if 'sigma' not in buffers:
        #     self.module.register_buffer('sigma', torch.ones(1))

        # sigma_ = self.__get_buffer('sigma')
        weight = self.__get_parameter('weight')
        var = self.__get_buffer('running_var')
        sigma = torch.max(torch.abs(weight) / torch.sqrt(var + self.module.eps)) # .to(device)
        # sigma_.copy_(sigma)

        return sigma

    def __get_parameter(self, name):
        return dict(self.module.named_parameters())[name].detach()

    def __get_buffer(self, name):
        return dict(self.module.named_buffers())[name].detach()

    def __normalize(self, tensor):
        dim = tensor.size()
        return F.normalize(tensor.view(-1), dim=0).view(dim)
        
