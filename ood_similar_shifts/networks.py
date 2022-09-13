import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod
import pdb
import numpy as np

Tensor = TypeVar('torch.tensor')


# Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)
        


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
    

class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules.append(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dims[0]),
                nn.LeakyReLU())
        )
        # Build Encoder
        for j in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[j], hidden_dims[j + 1]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], in_dim),
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
#         result = result.view(-1, TODO)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def sample_adv(self, mean, var, num_samples):
        z = torch.randn(num_samples, self.latent_dim)*var + mean
        z = z.float().to(device)
        samples = vae.decode(z)
        return samples, z
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, fourier):
        super().__init__()
        input_dim = obs_dim*40 if fourier else obs_dim
        self.trunk = mlp(input_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(weight_init)
        self.fourier = fourier        
        self.obs_f = LFF(obs_dim, obs_dim*40)

    def forward(self, obs):
        if self.fourier:
            obs = self.obs_f(obs)
        next_pred = self.trunk(obs)
        return next_pred


# Define the forward model for nonlinear hypernet
class TransformPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
    
    # Going forward with passed in parameters
    def forward_parameters(self, in_val, parameters=None):
        if parameters is None: 
            parameters = list(self.parameters())
        
        output = in_val    
        for params_idx in range(0, len(parameters) - 2, 2):
            w = parameters[params_idx]
            b = parameters[params_idx + 1]
            output = F.linear(output, w, b)
            output = F.relu(output)
        w = parameters[-2]
        b = parameters[-1]
        output = F.linear(output, w, b)
        return output        



class LFF(nn.Linear):
    def __init__(self, inp, out, bscale=0.5):
        #out = 40*inp
        super().__init__(inp, out)
        nn.init.normal(self.weight, std=bscale/inp)
        nn.init.uniform(self.bias, -1.0, 1.0)
    
    def forward(self, x):
        x = np.pi * super().forward(x)
        return torch.sin(x)


class BilinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, hidden_depth, fourier):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.fourier = fourier
        self.obs_f = LFF(obs_dim, obs_dim*40)
        self.delta_f = LFF(obs_dim, obs_dim*40)
        input_dim = obs_dim*40 if fourier else obs_dim
        self.obs_trunk = mlp(input_dim, hidden_dim, hidden_dim*act_dim, hidden_depth)
        self.delta_trunk = mlp(input_dim, hidden_dim, hidden_dim*act_dim, hidden_depth)        

    def forward(self, obs, deltas):
        if self.fourier:
            obs = self.obs_f(obs)
            deltas = self.delta_f(deltas)
        ob_embedding = self.obs_trunk(obs)
        ob_embedding = torch.reshape(ob_embedding, (-1, self.act_dim, self.hidden_dim)) #act_dim x hidden_dim
        delta_embedding = self.delta_trunk(deltas)
        delta_embedding = torch.reshape(delta_embedding, (-1, self.hidden_dim, self.act_dim)) #hidden_dim x act_dim
        pred = torch.diagonal(torch.matmul(ob_embedding, delta_embedding), dim1=1, dim2=2)
        return pred


class DeepSets(nn.Module):
    def __init__(self, state_dim, goal_dim, act_dim, hidden_dim, hidden_depth, fourier):
        super().__init__()
        self.fourier = fourier
        self.obs_f = LFF(state_dim, state_dim*40)
        self.goal_f = LFF(goal_dim, goal_dim*40) 
        input_dim_state = state_dim*40 if fourier else state_dim    
        input_dim_goal = goal_dim*40 if fourier else goal_dim
        self.obs_trunk = mlp(input_dim_state, hidden_dim, hidden_dim, hidden_depth)
        self.goal_trunk = mlp(input_dim_goal, hidden_dim, hidden_dim, hidden_depth)    
        self.comb_layer = mlp(hidden_dim, hidden_dim, act_dim, 1)    

    def forward(self, obs, deltas):
        if self.fourier:
            obs = self.obs_f(obs)
            deltas = self.goal_f(deltas)        
        ob_embedding = self.obs_trunk(obs)
        delta_embedding = self.goal_trunk(deltas)
        combined = ob_embedding + delta_embedding
        pred = self.comb_layer(combined)
        return pred