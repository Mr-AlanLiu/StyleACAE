import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scae import cv_ops
from torch_scae.nn_ext import Conv2dStack, multiple_attention_pooling_2d
from torch_scae.nn_utils import measure_shape

from torch_scae import math_ops
from torch_scae.distributions import GaussianMixture
from torch_scae.general_utils import prod
from torch_scae.nn_ext import relu1, MLP
from torch_scae.nn_utils import choose_activation

from torch_scae import cv_ops, math_ops
from torch_scae import nn_ext
from torch_scae.general_utils import prod
from torch_scae.math_ops import l2_loss
from training.set_transformer import SetTransformer

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch.utils.checkpoint import checkpoint

import torchvision

from torch.distributions import Bernoulli, LogisticNormal, Normal

#from training.gol import gol
from training import gol as gol

#import torch.nn as nn
#import torch.nn.functional as F

from typing import Tuple
from monty.collections import AttrDict

#rom torch_scae import cv_ops
#from torch_scae.nn_ext import Conv2dStack, multiple_attention_pooling_2d
#from torch_scae.nn_utils import measure_shape


import numpy as np
#global x3

#BEGIN-----------------------------------SCAE--------------------------------

#------------------------------PCAE-ENCODER----------------------------------
class CNNEncoder(nn.Module):
    def __init__(self,
                 #input_shape: Tuple[3,32,32],
                 #out_channels: Tuple[128, 128, 128, 128],
                 #kernel_sizes:Tuple[3,3,3,3],
                 #strides: Tuple[1,1,2,2],
                 #input_shape=[3,32,32],
                 #out_channels=[128,128,128,128],
                 #kernel_sizes=[3,3,3,3],
                 #strides=[1,1,2,2],
                 input_shape,
                 out_channels,
                 kernel_sizes,
                 strides,
                 activation=nn.ReLU,
                 activate_final=True,):
        super().__init__()
        self.network = Conv2dStack(in_channels=input_shape[0],
                                   out_channels=out_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   activation=activation,
                                   activate_final=activate_final)
        #self.output_shape=output_shape
        self.output_shape = measure_shape(self.network, input_shape=input_shape)
        
        #print('typle:',type(self.output_shape))

    def forward(self, image):
        return self.network(image)


class CapsuleImageEncoder(nn.Module):
    def __init__(self,
                 input_shape=[3, 64, 64],
                 #encoder:CNNEncoder,
                 #n_caps: 32,
                #n_poses: 6,
                 #n_special_features: int = 64,
                 #input_shape: Tuple[int, int, int],
                 
                 n_caps=64,
                 n_poses=6,
                 n_special_features= 0,
                 #noise_scale: float = 4.,
                 noise_scale = 4.,
                 similarity_transform=False,
                 ):

        super().__init__()
        self.input_shape = input_shape
        #self.encoder = encoder,
        self.n_caps = n_caps  # M
        self.n_poses = n_poses  # P
        self.n_special_features = n_special_features  # S
        self.noise_scale = noise_scale
        self.similarity_transform = similarity_transform
        self.encoder=CNNEncoder(input_shape=[3,64,64], out_channels=[128,128,128,128], kernel_sizes=[3,3,3,3], strides=[2,2,1,1], activation=nn.ReLU, activate_final=True)
        self._build()

        self.output_shapes = AttrDict(
            pose=(n_caps, n_poses),
            presence=(n_caps,),
            feature=(n_caps, n_special_features),
        )

    def _build(self):
        self.img_embedding_bias = nn.Parameter(
            data=torch.zeros(self.encoder.output_shape, dtype=torch.float32),
            requires_grad=True
        )
        in_channels = self.encoder.output_shape[0]
        self.caps_dim_splits = [self.n_poses, 1, self.n_special_features]  # 1 for presence
        self.n_total_caps_dims = sum(self.caps_dim_splits)
        out_channels = self.n_caps * (self.n_total_caps_dims + 1)  # 1 for attention
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, image):  # (B, C, H, H)
        #image=image.float
        #image =image.to(device)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #image = image.to(device)
        batch_size = image.shape[0]  # B
        #print('image_shapeA:',image.shape)
        torch.cuda.empty_cache()
        img_embedding = self.encoder(image) # (B, D, G, G)
        #torchvision.utils.save_image(img_embedding, '/home/zhao/Alan/GAN-NEW/Styleformer-main-2021-7-5/out', nrow=8, padding=2,normalize=False, range=None, scale_each=False, pad_value=0)
        h = img_embedding + self.img_embedding_bias.unsqueeze(0)  # (B, D, G, G)
        h = self.att_conv(h)  # (B, M * (P + 1 + S + 1), G, G)
        #part_capsules = torchvision.utils.make_grid( h, nrow=8, pad_value=0, padding=1)#可视化
 
       
        h = multiple_attention_pooling_2d(h, self.n_caps)  # (B, M * (P + 1 + S), 1, 1)
        h = h.view(batch_size, self.n_caps, self.n_total_caps_dims)  # (B, M, (P + 1 + S))
        del img_embedding
       
       
       

        # (B, M, P), (B, M, 1), (B, M, S)
        pose, presence_logit, special_feature = torch.split(h, self.caps_dim_splits, -1)
        
        del h

        if self.n_special_features == 0:
            special_feature = None

        presence_logit = presence_logit.squeeze(-1)  # (B, M)
        if self.training and self.noise_scale > 0.:
            noise = (torch.rand_like(presence_logit) - .5) * self.noise_scale
            presence_logit = presence_logit + noise  # (B, M)

        presence = torch.sigmoid(presence_logit)  # (B, M)
        presence = presence.detach()
        pose = cv_ops.geometric_transform(pose, self.similarity_transform)  # (B, M, P)
        pose = pose.detach()
        #print('feature.shape:', special_feature.shape)
        return AttrDict(pose=pose,
                        presence=presence,
                        feature=special_feature)
'''
#-------------------------------PCAE-DECODER---------------------------------
class TemplateGenerator(nn.Module):
    """Template-based primary capsule decoder for images."""

    def __init__(self,
                 n_templates=32,
                 n_channels=3,
                 template_size=Tuple[11,11],
                 template_nonlin='sigmoid',
                 dim_feature=None,
                 colorize_templates=True,
                 color_nonlin='sigmoid'):

        super().__init__()
        self.n_templates = n_templates  # M
        self.template_size = template_size  # (H, W)
        self.n_channels = n_channels  # C
        self.template_nonlin = choose_activation(template_nonlin)
        self.dim_feature = dim_feature  # F
        self.colorize_templates = colorize_templates
        self.color_nonlin = choose_activation(color_nonlin)

        self._build()

    def _build(self):
        # create templates
        template_shape = (
            1, self.n_templates, self.n_channels, *self.template_size
        )

        # make each templates orthogonal to each other at init
        n_elems = prod(template_shape[2:])  # channel, height, width
        n = max(self.n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:self.n_templates, :n_elems].reshape(template_shape)
        q = q.astype(np.float32)
        q = (q - q.min()) / (q.max() - q.min())
        self.template_logits = nn.Parameter(torch.from_numpy(q),
                                            requires_grad=True)

        if self.colorize_templates:
            self.templates_color_mlp = MLP(
                sizes=[self.dim_feature, 32, self.n_channels])

    def forward(self, feature=None, batch_size=None):
        """
        Args:
          feature: [B, n_templates, dim_feature] tensor; these features
          are used to change templates based on the input, if present.
          batch_size (int): batch_size in case feature is None

        Returns:
          (B, n_templates, n_channels, *template_size) tensor.
        """
        # (B, M, F)
        if feature is not None:
            batch_size = feature.shape[0]

        # (1, M, C, H, W)
        raw_templates = self.template_nonlin(self.template_logits)

        if self.colorize_templates and feature is not None:
            n_templates = feature.shape[1]
            template_color = self.templates_color_mlp(
                feature.view(batch_size * n_templates, -1)
            )  # (BxM, C)
            if self.color_nonlin == relu1:
                template_color += .99
            template_color = self.color_nonlin(template_color)
            template_color = template_color.view(
                batch_size, n_templates, template_color.shape[1]
            )  # (B, M, C)
            templates = raw_templates * template_color[:, :, :, None, None]
        else:
            templates = raw_templates.repeat(batch_size, 1, 1, 1, 1)

        return AttrDict(
            raw_templates=raw_templates,
            templates=templates,
        )


class TemplateBasedImageDecoder(nn.Module):
    """Template-based primary capsule decoder for images."""

    def __init__(self,
                 n_templates:32,
                 template_size: Tuple[11, 11],
                 output_size: Tuple[32, 32],
                 learn_output_scale=[32,32],
                 use_alpha_channel=True,
                 background_value=True):

        super().__init__()
        self.n_templates = n_templates
        self.template_size = template_size
        self.output_size = output_size
        self.learn_output_scale = learn_output_scale
        self.use_alpha_channel = use_alpha_channel
        self.background_value = background_value

        self._build()

    def _build(self):
        if self.use_alpha_channel:
            shape = (1, self.n_templates, 1, *self.template_size)
            self.templates_alpha = nn.Parameter(torch.zeros(*shape),
                                                requires_grad=True)#(1, 部件胶囊的个数或模板的个数，1，H, W)
        else:
            self.temperature_logit = nn.Parameter(torch.rand(1),
                                                  requires_grad=True)

        if self.learn_output_scale:
            self.scale = nn.Parameter(torch.rand(1), requires_grad=True)

        self.bg_mixing_logit = nn.Parameter(torch.tensor([0.0]),
                                            requires_grad=True)
        if self.background_value:
            self.bg_value = nn.Parameter(torch.tensor([0.0]),
                                         requires_grad=True)

    def forward(self,
                templates,
                pose,
                presence=None,
                bg_image=None):
        """Builds the module.

        Args:
          templates: (B, n_templates, n_channels, *template_size) tensor
          pose: [B, n_templates, 6] tensor.
          presence: [B, n_templates] tensor.
          bg_image: [B, n_channels, *output_size] tensor representing the background.

        Returns:
          (B, n_templates, n_channels, *output_size) tensor.
        """
        device = templates.device

        # B, M, C, H, W
        batch_size, n_templates, n_channels, height, width = templates.shape

        # transform templates
        templates = templates.view(batch_size * n_templates,
                                   *templates.shape[2:])  # (B*M, C, H, W)
        affine_matrices = pose.view(batch_size * n_templates, 2, 3)  # (B*M, 2, 3)
        target_size = [
            batch_size * n_templates, n_channels, *self.output_size]#（batch_size*胶囊的数量或是模板的数量, n_channels, H, W）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            affine_grids = F.affine_grid(affine_matrices, target_size)# affine_grids=( batch_size*胶囊的数量或是模板的数量, n_channels, H, W)
            transformed_templates = F.grid_sample(
                templates, affine_grids, align_corners=False)#(batch_size*胶囊的数量或是模板的数量, n_channels, H, W
        transformed_templates = transformed_templates.view(
            batch_size, n_templates, *target_size[1:])#(batch_size,胶囊的数量或是模板的数量, n_channels, H, W)
        del templates, target_size, affine_matrices

        # background image
        if bg_image is not None:
            bg_image = bg_image.unsqueeze(1)
        else:
            bg_image = torch.sigmoid(self.bg_value).repeat(
                *transformed_templates[:, :1].shape)

        transformed_templates = torch.cat([transformed_templates, bg_image], 1)#batch_size,胶囊的数量或是模板的数量+1, n_channels, H, W
        del bg_image

        if self.use_alpha_channel:
            template_mixing_logits = self.templates_alpha.repeat(
                batch_size, 1, 1, 1, 1)#(batch_size,部件胶囊个数或是模板的个数，1，H,W)
            template_mixing_logits = template_mixing_logits.view(
                batch_size * n_templates, *template_mixing_logits.shape[2:])#（batchz_size*部件胶囊个数或是模板的个数，1，H,W）
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                template_mixing_logits = F.grid_sample(
                    template_mixing_logits, affine_grids, align_corners=False)#(batch_size,胶囊的数量或是模板的数量, n_channels, H, W)
            template_mixing_logits = template_mixing_logits.view(
                batch_size, n_templates, *template_mixing_logits.shape[1:])#batch_size,胶囊的数量或是模板的数量, n_channels, H, W

            bg_mixing_logit = F.softplus(self.bg_mixing_logit).repeat(
                *template_mixing_logits[:, :1].shape)#(batch-size,1,n_channels,H,W)
            template_mixing_logits = torch.cat(
                [template_mixing_logits, bg_mixing_logit], dim=1)#batch_size,胶囊的数量或是模板的数量+1, n_channels, H, W
            del bg_mixing_logit
        else:
            temperature = F.softplus(self.temperature_logit + .5) + 1e-4
            template_mixing_logits = transformed_templates / temperature
            del temperature

        if self.learn_output_scale:
            scale = F.softplus(self.scale) + 1e-4
        else:
            scale = torch.tensor([1.0], device=device)

        if presence is not None:
            bg_presence = presence.new_ones([batch_size, 1])#batch_size,1
            presence = torch.cat([presence, bg_presence], dim=1)#(B,M+1)
            presence = presence.view(
                *presence.shape, *([1] * len(template_mixing_logits.shape[2:])))#这句话太绕了，batch_size, M+1,1
            template_mixing_logits += math_ops.log_safe(presence)
            del bg_presence, presence

        rec_pdf = GaussianMixture.make_from_stats(
            loc=transformed_templates,
            scale=scale,
            mixing_logits=template_mixing_logits
        )

        return AttrDict(
            transformed_templates=transformed_templates,
            mixing_logits=template_mixing_logits,
            pdf=rec_pdf,
        )
#------------------------------------OCAE-DECODER---------------------------
class CapsuleLayer(nn.Module):
    """Implementation of a capsule layer."""

    # number of parameters needed to parametrize linear transformations.
    n_transform_params = 6  # P

    def __init__(self,
                 n_caps= 64,
                 dim_feature= 32,
                 n_votes= 32,
                 dim_caps= 32,
                 hidden_sizes=(128,),
                 caps_dropout_rate=0.0,
                 learn_vote_scale=True,
                 allow_deformations=True,
                 noise_type=None,
                 noise_scale=4,
                 similarity_transform=False,
                 ):
        """Builds the module.

        Args:
          n_caps: int, number of capsules.
          dim_caps: int, number of capsule parameters
          hidden_sizes: int or sequence of ints, number of hidden units for an MLP
            which predicts capsule params from the input encoding.
          n_caps_dims: int, number of capsule coordinates.
          caps_dropout_rate: float in [0, 1].
          n_votes: int, number of votes generated by each capsule.
          learn_vote_scale: bool, learns input-dependent scale for each
            capsules' votes.
          allow_deformations: bool, allows input-dependent deformations of capsule-part
            relationships.
          noise_type: 'normal', 'logistic' or None; noise type injected into
            presence logits.
          noise_scale: float >= 0. scale parameters for the noise.
          similarity_transform: boolean; uses similarity transforms if True.
        """
        super().__init__()

        self.n_caps = n_caps  # O
        self.dim_feature = dim_feature  # F
        self.hidden_sizes = list(hidden_sizes)  # [H_i, ...]
        self.dim_caps = dim_caps  # D
        self.caps_dropout_rate = caps_dropout_rate
        self.n_votes = n_votes
        self.learn_vote_scale = learn_vote_scale
        self.allow_deformations = allow_deformations
        self.noise_type = noise_type
        self.noise_scale = noise_scale

        self.similarity_transform = similarity_transform

        self._build()

    def _build(self):
        # Use separate parameters to do predictions for different capsules.
        sizes = [self.dim_feature] + self.hidden_sizes + [self.dim_caps]
        self.mlps = nn.ModuleList([
            nn_ext.MLP(sizes=sizes)
            for _ in range(self.n_caps)
        ])

        self.output_shapes = (
            [self.n_votes, self.n_transform_params],  # OPR-dynamic
            [1, self.n_transform_params],  # OVR
            [1],  # per-object presence
            [self.n_votes],  # per-vote-presence
            [self.n_votes],  # per-vote scale
        )
        self.splits = [prod(i) for i in self.output_shapes]
        self.n_outputs = sum(self.splits)  # A

        # we don't use bias in the output layer in order to separate the static
        # and dynamic parts of the OP
        sizes = [self.dim_caps + 1] + self.hidden_sizes + [self.n_outputs]
        self.caps_mlps = nn.ModuleList([
            nn_ext.MLP(sizes=sizes, bias=False)
            for _ in range(self.n_caps)
        ])

        self.caps_bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_caps, *shape), requires_grad=True)
            for shape in self.output_shapes[1:]
        ])

        # constant object-part relationship matrices, OPR-static
        self.cpr_static = nn.Parameter(
            torch.zeros([1, self.n_caps, self.n_votes, self.n_transform_params]),
            requires_grad=True
        )

    def forward(self, feature, parent_transform=None, parent_presence=None):
        """
        Args:
          feature: Tensor of encodings of shape [B, O, F].
          parent_transform: Tuple of (matrix, vector).
          parent_presence: pass

        Returns:
          A bunch of stuff.
        """
        device = feature.device

        batch_size = feature.shape[0]  # B

        # Predict capsule and additional params from the input encoding.
        # [B, O, D]

        caps_feature_list = feature.unbind(1)  # [(B, F)] * O
        caps_param_list = [self.mlps[i](caps_feature_list[i])
                           for i in range(self.n_caps)]  # [(B, D)] * O
        del feature, caps_feature_list
        raw_caps_param = torch.stack(caps_param_list, 1)  # (B, O, D)
        del caps_param_list

        if self.caps_dropout_rate == 0.0:
            caps_exist = torch.ones(batch_size, self.n_caps, 1)  # (B, O, 1)
        else:
            pmf = Bernoulli(1. - self.caps_dropout_rate)
            caps_exist = pmf.sample((batch_size, self.n_caps, 1))  # (B, O, 1)
        caps_exist = caps_exist.to(device)

        caps_param = torch.cat([raw_caps_param, caps_exist], -1)  # (B, O, D+1)
        del raw_caps_param, caps_exist

        caps_eparam_list = caps_param.unbind(1)  # [(B, D+1)] * O
        all_param_list = [self.caps_mlps[i](caps_eparam_list[i])
                          for i in range(self.n_caps)]  # [(B, A)] * O
        del caps_eparam_list
        all_param = torch.stack(all_param_list, 1)  # (B, O, A)
        del all_param_list
        all_param_split_list = torch.split(all_param, self.splits, -1)
        result = [t.view(batch_size, self.n_caps, *s)
                  for (t, s) in zip(all_param_split_list, self.output_shapes)]
        del all_param
        del all_param_split_list

        # add up static and dynamic object part relationship
        cpr_dynamic = result[0]  # (B, O, V, P)
        if not self.allow_deformations:
            cpr_dynamic = torch.zeros_like(cpr_dynamic)
        cpr_dynamic_reg_loss = l2_loss(cpr_dynamic) / batch_size
        cpr = self._make_transform(cpr_dynamic + self.cpr_static)  # (B, O, V, 3, 3)
        del cpr_dynamic

        # add bias to all remaining outputs
        # (B, O, 1, P), (B, O, 1), (B, O, V), (B, O, V)
        cvr, presence_logit_per_caps, presence_logit_per_vote, scale_per_vote = [
            t + bias
            for (t, bias) in zip(result[1:], self.caps_bias_list)
        ]
        del result

        # this is for hierarchical
        # (B, O, 1, 3, 3)
        if parent_transform is None:
            cvr = self._make_transform(cvr)
        else:
            cvr = parent_transform

        cvr_per_vote = cvr.repeat(1, 1, self.n_votes, 1, 1)  # (B, O, V, 3, 3)
        # PVR = OVR x OPR
        vote = torch.matmul(cvr_per_vote, cpr)  # (B, O, V, 3, 3)
        del cvr_per_vote, cpr

        if self.caps_dropout_rate > 0.0:
            presence_logit_per_caps = presence_logit_per_caps \
                                      + math_ops.log_safe(caps_exist)

        def add_noise(tensor):
            """Adds noise to tensors."""
            if self.noise_type == 'uniform':
                noise = (torch.rand_like(tensor) - 0.5) * self.noise_scale
            elif self.noise_type == 'logistic':
                pdf = LogisticNormal(0., self.noise_scale)
                noise = pdf.sample(tensor.shape)
            elif not self.noise_type:
                noise = torch.tensor([0.0])
            else:
                raise ValueError(f'Invalid noise type: {self.noise_type}')
            return tensor + noise.to(device)

        presence_logit_per_caps = add_noise(presence_logit_per_caps)  # (B, O, 1)
        presence_logit_per_vote = add_noise(presence_logit_per_vote)  # (B, O, V)

        if parent_presence is not None:
            presence_per_caps = parent_presence
        else:
            presence_per_caps = torch.sigmoid(presence_logit_per_caps)

        vote_presence = presence_per_caps * torch.sigmoid(presence_logit_per_vote)  # (B, O, V)
        del presence_per_caps

        # (B, O, V)
        if self.learn_vote_scale:
            # for numerical stability
            scale_per_vote = F.softplus(scale_per_vote + .5) + 1e-2
        else:
            scale_per_vote = torch.ones_like(scale_per_vote, device=device)

        return AttrDict(
            vote=vote,  # (B, O, V, 3, 3)
            scale=scale_per_vote,  # (B, O, V)
            vote_presence=vote_presence,  # (B, O, V)
            presence_logit_per_caps=presence_logit_per_caps,  # (B, O, 1)
            presence_logit_per_vote=presence_logit_per_vote,  # (B, O, V)
            cpr_dynamic_reg_loss=cpr_dynamic_reg_loss,
        )

    def _make_transform(self, params):
        return cv_ops.geometric_transform(params, self.similarity_transform,
                                          nonlinear=True, as_matrix=True)


class CapsuleLikelihood:
    """Capsule voting mechanism."""

    def __init__(self, vote, scale, vote_presence, dummy_vote):
        super().__init__()
        self.n_caps = vote.shape[1]  # O
        self.vote = vote  # (B, O, M, P)
        self.scale = scale  # (B, O, M)
        self.vote_presence = vote_presence  # (B, O, M)
        self.dummy_vote = dummy_vote  # (1, 1, M, P)

    def _get_pdf(self, votes, scales):
        return Normal(votes, scales)

    def __call__(self, x, presence=None):  # (B, M, P), (B, M)
        device = x.device

        batch_size, n_input_points, dim_in = x.shape  # B, M, P

        # since scale is a per-caps scalar and we have one vote per capsule
        vote_component_pdf = self._get_pdf(self.vote,
                                           self.scale.unsqueeze(-1))

        # expand input along caps dimensions
        expanded_x = x.unsqueeze(1)  # (B, 1, M, P)
        vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)  # (B, O, M, P)
        vote_log_prob = vote_log_prob_per_dim.sum(-1)  # (B, O, M)
        del x, expanded_x, vote_log_prob_per_dim

        # (B, 1, M)
        dummy_vote_log_prob = torch.zeros(
            batch_size, 1, n_input_points, device=device) + np.log(0.01)

        # p(x_m | k, m)
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], 1)  # (B, O+1, M)
        del dummy_vote_log_prob

        #
        dummy_logit = torch.full((batch_size, 1, n_input_points),
                                 fill_value=np.log(0.01), device=device)

        mixing_logit = math_ops.log_safe(self.vote_presence)  # (B, O, M)
        mixing_logit = torch.cat([mixing_logit, dummy_logit], 1)  # (B, O+1, M)
        mixing_log_prob = mixing_logit - mixing_logit.logsumexp(1, keepdim=True)  # (B, O+1, M)

        # mask for votes which are better than dummy vote
        vote_presence_binary = (mixing_logit[:, :-1] > mixing_logit[:, -1:]).float()  # (B, O, M)

        # (B, O + 1, M)
        posterior_mixing_logits_per_point = mixing_logit + vote_log_prob
        del vote_log_prob

        # (B, M)
        mixture_log_prob_per_point = posterior_mixing_logits_per_point.logsumexp(1)

        if presence is not None:
            mixture_log_prob_per_point = mixture_log_prob_per_point * presence.float()

        # (B,)
        mixture_log_prob_per_example = mixture_log_prob_per_point.sum(1)
        del mixture_log_prob_per_point

        # scalar
        mixture_log_prob_per_batch = mixture_log_prob_per_example.mean()
        del mixture_log_prob_per_example

        # winner object index per part
        winning_vote_idx = torch.argmax(
            posterior_mixing_logits_per_point[:, :-1], 1)  # (B, M)

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)
        batch_idx = batch_idx.repeat(1, n_input_points)  # (B, M)

        point_idx = torch.arange(n_input_points, device=device).unsqueeze(0)  # (1, M)
        point_idx = point_idx.repeat(batch_size, 1)  # (B, M)

        idx = torch.stack([batch_idx, winning_vote_idx, point_idx], -1)
        del batch_idx
        del point_idx

        # (B, M, P)
        winning_vote = self.vote[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
        assert winning_vote.shape == (batch_size, n_input_points, dim_in)

        # (B, M)
        winning_presence = \
            self.vote_presence[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
        assert winning_presence.shape == (batch_size, n_input_points)
        del idx

        # is winner capsule or dummy
        is_from_capsule = winning_vote_idx // n_input_points

        # Soft winner. START
        # (B, O+1, M)
        posterior_mixing_prob = F.softmax(posterior_mixing_logits_per_point, 1)
        del posterior_mixing_logits_per_point

        dummy_vote = self.dummy_vote.repeat(batch_size, 1, 1, 1)  # (B, 1, M, P)
        dummy_presence = torch.zeros([batch_size, 1, n_input_points], device=device)

        votes = torch.cat((self.vote, dummy_vote), 1)  # (B, O+1, M, P)
        vote_presence = torch.cat([self.vote_presence, dummy_presence], 1)  # (B, O+1, M)
        del dummy_vote
        del dummy_presence

        # (B, M, P)
        soft_winner_vote = torch.sum(posterior_mixing_prob.unsqueeze(-1) * votes, 1)
        assert soft_winner_vote.shape == (batch_size, n_input_points, dim_in)

        # (B, M)
        soft_winner_presence = torch.sum(posterior_mixing_prob * vote_presence, 1)
        assert soft_winner_presence.shape == (batch_size, n_input_points)
        # Soft winner. END

        # (B, O, M)
        posterior_mixing_prob = posterior_mixing_prob[:, :-1]

        return AttrDict(
            log_prob=mixture_log_prob_per_batch,
            vote_presence_binary=vote_presence_binary,
            winner=winning_vote,
            winner_presence=winning_presence,
            soft_winner=soft_winner_vote,
            soft_winner_presence=soft_winner_presence,
            posterior_mixing_prob=posterior_mixing_prob,
            mixing_log_prob=mixing_log_prob,
            mixing_logit=mixing_logit,
            is_from_capsule=is_from_capsule,
        )


class CapsuleObjectDecoder(nn.Module):
    def __init__(self, capsule_layer: CapsuleLayer):
        """
        Args:
          capsule_layer: a capsule layer to predict object parameters
        """
        super().__init__()
        self.capsule_layer = capsule_layer

        self.dummy_vote = nn.Parameter(
            torch.zeros(1, 1, capsule_layer.n_votes, capsule_layer.n_transform_params),
            requires_grad=True
        )

    @property
    def n_obj_capsules(self):
        return self.capsule_layer.n_caps

    def forward(self,
                obj_encoding: torch.Tensor,
                part_pose: torch.Tensor,
                part_presence: torch.Tensor = None):
        """
        Args:
          obj_encoding: Tensor of shape [B, O, D].
          part_pose: Tensor of shape [B, M, P]
          part_presence: Tensor of shape [B, M] or None; if it exists, it
            indicates which input parts exist.

        Returns:
          A bunch of stuff.
        """
        batch_size, n_caps = obj_encoding.shape[:2]
        n_votes = part_pose.shape[1]

        res = self.capsule_layer(obj_encoding)
        # remove homogeneous coord row from transformation matrices
        # and flatten last two dimensions
        res.vote = res.vote[..., :-1, :].view(batch_size, n_caps, n_votes, -1)
        # compute capsule presence by maximum part vote
        res.caps_presence = res.vote_presence.max(-1)[0]

        # compute likelihood of object decoding
        likelihood = CapsuleLikelihood(
            vote=res.vote,
            scale=res.scale,
            vote_presence=res.vote_presence,
            dummy_vote=self.dummy_vote
        )
        ll_res = likelihood(part_pose, presence=part_presence)
        res.update(ll_res)
        del likelihood

        return res


# prior sparsity loss
# l2(aggregated_prob - constant)
def capsule_l2_loss(caps_presence,
                    n_classes: int,
                    within_example_constant=None,
                    **unused_kwargs):
    """Computes l2 penalty on capsule activations."""

    del unused_kwargs

    batch_size, num_caps = caps_presence.shape  # B, O

    if within_example_constant is None:
        within_example_constant = float(num_caps) / n_classes  # K / C
    within_example = torch.mean(
        (caps_presence.sum(1) - within_example_constant) ** 2)

    between_example_constant = float(batch_size) / n_classes  # B / C
    between_example = torch.mean(
        (caps_presence.sum(0) - between_example_constant) ** 2)

    return within_example, between_example


# posterior sparsity loss
def capsule_entropy_loss(caps_presence, k=1, **unused_kwargs):
    """Computes entropy in capsule activations."""
    del unused_kwargs

    # caps_presence (B, O)

    within_prob = math_ops.normalize(caps_presence, 1)  # (B, O)
    within_example = math_ops.cross_entropy_safe(within_prob,
                                                 within_prob * k)  # scalar

    total_caps_prob = torch.sum(caps_presence, 0)  # (O, )
    between_prob = math_ops.normalize(total_caps_prob, 0)  # (O, )
    between_example = math_ops.cross_entropy_safe(between_prob,
                                                  between_prob * k)  # scalar
    # negate since we want to increase between example entropy
    return within_example, -between_example


# kl(aggregated_prob||uniform)
def neg_capsule_kl(caps_presence, **unused_kwargs):
    del unused_kwargs

    n_caps = int(caps_presence.shape[-1])
    return capsule_entropy_loss(caps_presence, k=n_caps)


def sparsity_loss(loss_type, *args, **kwargs):
    """Computes capsule sparsity loss according to the specified type."""
    if loss_type == 'l2':
        sparsity_func = capsule_l2_loss
    elif loss_type == 'entropy':
        sparsity_func = capsule_entropy_loss
    elif loss_type == 'kl':
        sparsity_func = neg_capsule_kl
    else:
        raise ValueError(f"Invalid sparsity loss: {loss_type}")

    return sparsity_func(*args, **kwargs)



#END-------------------------------------SCAE--------------------------------
'''
        
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_mlp(x, fc1_weight, fc2_weight, u_weight, activation, mlp_ratio, mlp_drop, styles):
    batch_size = x.shape[0]
    seq_length = x.shape[1]
    hidden_dimension = x.shape[2]
    act_func = get_act(activation)
    layernorm = nn.InstanceNorm1d(seq_length, affine=False)
    skip = x
    
    
    styles1 = styles[:, :hidden_dimension]
    styles2 = styles[:, hidden_dimension:]
    
    x = x * styles1.to(x.dtype).reshape(batch_size, 1, -1)
    x = layernorm(x)
    
    
    fc1 = None
    fc2 = None
    fc1_dcoefs = None
    fc2_dcoefs = None
    
    fc1 = fc1_weight.unsqueeze(0)
    fc2 = fc2_weight.unsqueeze(0)
    fc1 = fc1 * styles1.reshape(batch_size, 1, -1)
    fc2 = fc2 * styles2.reshape(batch_size, 1, -1)
    
    
    fc1_dcoefs = (fc1.square().sum(dim=[2]) + 1e-8).rsqrt()
    fc2_dcoefs = (fc2.square().sum(dim=[2]) + 1e-8).rsqrt()
   
    x = torch.matmul(x, fc1_weight.t().to(x.dtype))
    x = x * fc1_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    
    x = x * styles2.to(x.dtype).reshape(batch_size, 1, -1)
    x = act_func(x)
    #x = F.dropout(x, p=mlp_drop)
    x = torch.matmul(x, fc2_weight.t().to(x.dtype))
    x = x * fc2_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    if x.shape[2] != skip.shape[2]:
        #print("bad")
        u = None
        u_dcoefs = None

        u = u_weight
        u_dcoefs = (u.square().sum(dim=[1]) + 1e-8).rsqrt()

        skip = torch.matmul(skip, u_weight.t().to(x.dtype))
        skip = skip * u_dcoefs.to(x.dtype).reshape(1, 1, -1)
    #x = F.dropout(x, p=mlp_drop)
    
    return x

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_style_mlp(x, weight, styles):
    batch_size = x.shape[0]
    channel = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]

    w = None
    dcoefs = None
    
    w = weight.unsqueeze(0)
    w = w * styles.reshape(batch_size, 1, -1)
    dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    x = x.reshape(batch_size, channel, width*height).permute(0, 2, 1)
    x = x * styles.to(x.dtype).reshape(batch_size, 1, -1)
    x = torch.matmul(x, weight.t().to(x.dtype))
    x = x * dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    x = x.permute(0, 2, 1).reshape(batch_size, -1, width, height)
    
    return x
  
#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_channel_attention(x, q_weight, k_weight, v_weight, w_weight, u_weight, proj_weight, styles, num_heads):
    
    batch_size = x.shape[0]
    seq_length = x.shape[1]
    hidden_dimension = x.shape[2]
    
    assert hidden_dimension % num_heads == 0
    
    depth = hidden_dimension // num_heads
    
    attention_scale = torch.tensor(depth ** -0.5).to(x.dtype)

    layernorm = nn.InstanceNorm1d(seq_length, affine=False) 
    
    styles1 = styles[:, :hidden_dimension]
    styles2 = styles[:, hidden_dimension:]


    x = x * styles1.to(x.dtype).reshape(batch_size, 1, -1)
    x = layernorm(x)
    
    q = q_weight.unsqueeze(0)
    q = q * styles1.reshape(batch_size, 1, -1)
    q_dcoefs = (q.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    k = k_weight.unsqueeze(0)
    k = k * styles1.reshape(batch_size, 1, -1)
    k_dcoefs = (k.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    v = v_weight.unsqueeze(0)
    v = v * styles1.reshape(batch_size, 1, -1)
    v_dcoefs = (v.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    w = w_weight.unsqueeze(0)
    w = w * styles2.reshape(batch_size, 1, -1)
    w_dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    
    q_value = torch.matmul(x, q_weight.t().to(x.dtype)) * q_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    q_value = q_value.reshape(batch_size, seq_length, num_heads, depth).permute(0,2,1,3)
    k_value = torch.matmul(x, k_weight.t().to(x.dtype)) * k_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    k_value = k_value.reshape(batch_size, seq_length, num_heads, depth).permute(0,2,1,3)
    if proj_weight is not None:
        k_value = torch.matmul(k_value.permute(0,1,3,2), proj_weight.t().to(x.dtype)).permute(0,1,3,2)
    v_value = torch.matmul(x, v_weight.t().to(x.dtype))

    v_value = v_value * v_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    
    v_value = v_value * styles2.to(x.dtype).reshape(batch_size, 1, -1)
    skip = v_value
    if proj_weight is not None:
        v_value = torch.matmul(v_value.permute(0,2,1), proj_weight.t().to(x.dtype)).permute(0,2,1)
        v_value = v_value.reshape(batch_size, 256, num_heads, depth).permute(0,2,1,3)
    
    else:
         v_value = v_value.reshape(batch_size, seq_length, num_heads, depth).permute(0,2,1,3)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
    attn = torch.matmul(q_value, k_value.permute(0,1,3,2)) * attention_scale 
    revised_attn = attn 

    attn_score = revised_attn.softmax(dim=-1)

    x = torch.matmul(attn_score , v_value).permute(0, 2, 1, 3).reshape(batch_size, seq_length, hidden_dimension) 

    x = torch.matmul(x, w_weight.t().to(x.dtype))

    x = x * w_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    
    u = u_weight.unsqueeze(0)
    u = u * styles2.reshape(batch_size, 1, -1)
    u_dcoefs = (u.square().sum(dim=[2]) + 1e-8).rsqrt()
    
    skip = torch.matmul(skip, u_weight.t().to(x.dtype))
    skip = skip * u_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    
    x = x + skip

    return x        

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x
        
#----------------------------------------------------------------------------
'''
@persistence.persistent_class
class MappingNetwork(nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features =0 # w_dim
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))  

        # Main layers
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x
'''

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Encoderlayer(nn.Module):
    def __init__(self, h_dim, w_dim, out_dim, seq_length, depth, minimum_head, use_noise=True, conv_clamp=None, proj_weight=None, channels_last=False):
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = max(minimum_head, h_dim // depth)
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.use_noise = use_noise
        self.conv_clamp = conv_clamp
        self.affine1 = FullyConnectedLayer(w_dim, h_dim*2, bias_init=1)
        
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        
        self.q_weight = torch.nn.Parameter(torch.FloatTensor(h_dim, h_dim).uniform_(-1./math.sqrt(h_dim), 1./math.sqrt(h_dim)).to(memory_format=memory_format))
        self.k_weight = torch.nn.Parameter(torch.FloatTensor(h_dim, h_dim).uniform_(-1./math.sqrt(h_dim), 1./math.sqrt(h_dim)).to(memory_format=memory_format))        
        self.v_weight = torch.nn.Parameter(torch.FloatTensor(h_dim, h_dim).uniform_(-1./math.sqrt(h_dim), 1./math.sqrt(h_dim)).to(memory_format=memory_format))
        self.w_weight = torch.nn.Parameter(torch.FloatTensor(out_dim, h_dim).uniform_(-1./math.sqrt(h_dim), 1./math.sqrt(h_dim)).to(memory_format=memory_format))
        
        self.proj_weight = proj_weight
        
        self.u_weight = torch.nn.Parameter(torch.FloatTensor(out_dim, h_dim).uniform_(-1./math.sqrt(h_dim), 1./math.sqrt(h_dim)).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([self.seq_length, 1]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_dim]))
        

        
    def forward(self, x, w, noise_mode='random', gain=1):
        assert noise_mode in ['random', 'const', 'none']
        misc.assert_shape(x, [None, self.seq_length, self.h_dim])
        styles1 = self.affine1(w)
        
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], self.seq_length, 1], device = x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
        #with torch.no_grad():
        with torch.no_grad():
           torch.cuda.empty_cache()
           x = modulated_channel_attention(x=x, q_weight=self.q_weight, k_weight=self.k_weight, v_weight=self.v_weight, w_weight=self.w_weight, u_weight=self.u_weight, proj_weight=self.proj_weight, styles=styles1, num_heads=self.num_heads)   
        
        if noise is not None:
            x = x.add_(noise)
       
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = x + self.bias.to(x.dtype)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = torch.clamp(x, max=act_clamp, min=-act_clamp)
        return x
            
    
#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = None
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels).uniform_(-1./math.sqrt(in_channels), 1./math.sqrt(in_channels)).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) 
        x = modulated_style_mlp(x=x, weight=self.weight, styles=styles)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x
    
#----------------------------------------------------------------------------    

@persistence.persistent_class
class EncoderBlock(nn.Module):
    def __init__(self, h_dim, w_dim, out_dim, depth, minimum_head, img_resolution, resolution, img_channels, is_first, is_last, architecture='skip', linformer=False, conv_clamp=None, use_fp16=False, fp16_channels_last=False, resample_filter =[1,3,3,1], scale_ratio=2, **layer_kwargs):
        super().__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.depth = depth
        self.minimum_head = minimum_head
        self.img_resolution = img_resolution
        self.resolution = resolution
        self.img_channels = img_channels
        self.seq_length = resolution * resolution
        self.is_first = is_first
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.num_attention = 0
        self.num_torgb = 0
        self.scale_ratio = scale_ratio
        self.conv_clamp = conv_clamp
        self.proj_weight = None
        #self.n_caps = n_caps
        #self.n_poses = n_poses
        #self.image=image
        #self.part_encoder=part_encoder

        self.part_encoder= CapsuleImageEncoder(input_shape=[1,32,32], n_caps=32, n_poses=6, n_special_features=16, noise_scale=4.0,  similarity_transform=False)
        
        memory_format = torch.contiguous_format
        
        if self.resolution>=32 and linformer:
            self.proj_weight = torch.nn.Parameter(torch.FloatTensor(256, self.seq_length                ).uniform_(-1./math.sqrt(self.seq_length), 1./math.sqrt(self.seq_length)).to(memory_format=memory_format))
        
        
        
        #if self.is_first and self.resolution == 8: #cifar10:8    ; stl10: 12
            #self.const = torch.nn.Parameter(torch.randn([self.seq_length, self.h_dim]))
        
        if self.is_first:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.seq_length, self.h_dim))
            
        if not self.is_last or out_dim is None:
            self.out_dim = h_dim
        
        self.enc = Encoderlayer(h_dim=self.h_dim, w_dim=self.w_dim, out_dim=self.out_dim, seq_length=self.seq_length, depth=self.depth, minimum_head=self.minimum_head, conv_clamp=self.conv_clamp, proj_weight=self.proj_weight)
        self.num_attention += 1
        
        if self.is_last and self.architecture == 'skip':
            self.torgb = ToRGBLayer(self.out_dim, self.img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1
      
        
    def forward(self, x, img, image, ws, force_fp32=True, fused_modconv=None):
        part_enc_res = self.part_encoder(image)
        misc.assert_shape(ws, [None, self.num_attention + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Input
        if self.is_first and self.resolution ==8: #cifar:8   stl10:12
            #print('part_enc_res.pose:',part_enc_res.pose.shape)
            #print('part_enc_res.presence',part_enc_res.presence.shape)
            p_1=part_enc_res.pose.view(-1,6)
            p_2=part_enc_res.presence.view(-1,1)
            x_1=nn.Parameter(torch.randn([6, self.h_dim]).to(device))
            x_2=nn.Parameter(torch.randn([1, self.seq_length]).to(device))
            x_11= torch.matmul(p_1, x_1)#(bm,p)*(p,h_dim)=(bm,h_dim)
            x_12= torch.matmul(p_2, x_2)#(bm,1)*(1,seq_length)=(bm,seq_length)
            x_13= x_12.view(self.seq_length,-1)#(seq_length.bm)
            x= torch.matmul(x_13,x_11)#(seq_length,bm)*(bm,h_dim)
            '''
            #x = self.const.to(dtype=dtype, memory_format=memory_format)
            x_1 = nn.Parameter(torch.randn([32, self.h_dim]).to(device))#self.n_capsglobal
            #device= x_1.device
            x_2 = nn.Parameter(torch.randn([7, self.seq_length])).to(device)#self.n_poses + 1
            #device= x_2.device
            #x_2 = torch.nn.Parameter(torch.randn([n_poses + 1, self.seq]))
            x_11= torch.matmul(part_enc_res.pose, x_1)#(b,m,p)=(b,n_caps,n_poses)*(n_caps,h_dim)
            #x_11=x_11.device
            x_12= torch.matmul(part_enc_res.presence,x_2)#(b,m,1)=(b,n_caps,1)*(n_poses+1, seq_length)
            #x_12=x_12.device
            x_112= torch.cat((x_11,x_12),1)
            x_new= x_112.view(-1, self.seq_length, self.h_dim)
            
            x = x_new.unsqueeze(0).repeat([ws.shape[0], 1, 1])'''
            x = x.to(dtype=dtype, memory_format=memory_format)#self.seq_length, self.h_dim
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1])
        else:
            '''
            x_1 = nn.Parameter(torch.randn([32, self.h_dim])).to(device)
            #x_1=x_1.device
            x_2 = nn.Parameter(torch.randn([7, self.seq_length])).to(device)
            #x_2=x_2.device
            #x_2 = torch.nn.Parameter(torch.randn([n_poses + 1, self.seq]))
            x_11= torch.matmul(part_enc_res.pose, x_1)
            #x_11 = x_11.device
            x_12= torch.matmul(part_enc_res.presence,x_2)
           #x_12= x_12.device
            x_112= torch.cat((x_11,x_12),1)
            x_new= x_112.view(-1, self.seq_length, self.h_dim)
            
            x = x_new.unsqueeze(0).repeat([ws.shape[0], 1, 1])
            x = x.to(dtype=dtype, memory_format=memory_format)'''
            '''
            p_1=part_enc_res.pose.view(-1,6)
            p_2=part_enc_res.presence.view(-1,1)
            x_1=nn.Parameter(torch.randn([6, self.h_dim]).to(device))
            x_2=nn.Parameter(torch.randn([1, self.seq_length]).to(device))
            x_11= torch.matmul(p_1, x_1)#(bm,p)*(p,h_dim)=(bm,h_dim)
            x_12= torch.matmul(p_2, x_2)#(bm,1)*(1,seq_length)=(bm,seq_length)
            x_13= x_12.view(self.seq_length,-1)#(seq_length.bm)
            x= torch.matmul(x_13,x_11)#(seq_length,bm)*(bm,h_dim)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1])
           # misc.assert_shape(x, [None, self.seq_length, self.h_dim])
            x = x.to(dtype=dtype, memory_format=memory_format)'''
            misc.assert_shape(x, [None, self.seq_length, self.h_dim])
            x = x.to(dtype=dtype, memory_format=memory_format)
        #print('x1:',x.shape)
        #Main layers
        if self.is_first:
            x = x + self.pos_embedding

        
        if self.architecture == 'resnet':
            y = self.skip(x.permute(0,2,1).reshape(ws.shape[0], self.h_dim, self.resolution, self.resolution))
            x = self.enc(x, next(w_iter))
            y = y.reshape(ws.shape[0], self.h_dim, self.seq_length)
            x = y.add_(x)
        else:
            x = self.enc(x, next(w_iter)).to(dtype=dtype, memory_format=memory_format)
        x2 = x
        #print('x2:',x2.shape)
        #ToRGB

        #print("x.shape:",x.shape)
        if self.is_last:
            if img is not None:
                misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution //2])
                img = upfirdn2d.upsample2d(img, self.resample_filter)
                         
            if self.architecture == 'skip':
                y = self.torgb(x.permute(0,2,1).reshape(ws.shape[0], self.out_dim, self.resolution, self.resolution), next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
            #upsample
            if self.resolution!=self.img_resolution:
                x = upfirdn2d.upsample2d(x.permute(0,2,1).reshape(ws.shape[0], self.out_dim, self.resolution, self.resolution), self.resample_filter)
                x = x.reshape(ws.shape[0], self.out_dim, self.seq_length * self.scale_ratio **2).permute(0,2,1)
                
            
            
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        assert x2 is None or x2.dtype == torch.float32
        #assert part_enc_res.pose is None or x2.dtype == torch.float32
        #assert part_enc_res.presence is None or x2.dtype == torch.float32

        return x, img,x2, part_enc_res.pose, part_enc_res.presence

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution,img_channels, depth, minimum_head, num_layers, G_dict, conv_clamp, channel_base = 8192, channel_max = 256, num_fp16_res = 0, linformer=False):
        #assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        gol._init()
        #先必须在主模块初始化（只在Main模块需要一次即可）
        #self.h_dim = h_dim    
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_block = num_layers
        self.block_resolutions = [2 ** i for i in range(3, self.img_resolution_log2 + 1)]#self.block_resolutions = [3*2 ** (i-1) for i in range(3, self.img_resolution_log2 + 1)]#
        #self.part_encoder=part_encoder
        #self.image=image
        #assert len(self.block_resolutions) == len(self.num_block)
        channels_dict = dict(zip(*[self.block_resolutions, G_dict]))
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res),8) #cifar:8  ; stl10:12
        #self.part_encoder= part_encoder
        self.num_ws = 0

        #self.*/=CapsuleImageEncoder()
        for i, res in enumerate(self.block_resolutions):
            h_dim = channels_dict[res]
            out_dim = None
            if res!=self.img_resolution:
                out_dim =channels_dict[res*2]
            use_fp16 = (res >= fp16_resolution)
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                is_first = (j == 0)
                is_last = (j == num_block_res - 1)
                block = EncoderBlock(
                                   h_dim=h_dim, w_dim=w_dim, out_dim=out_dim, depth=depth, minimum_head=minimum_head,                                              img_resolution=img_resolution, resolution=res, img_channels=img_channels, 
                                   is_first=is_first, is_last=is_last, use_fp16=use_fp16, conv_clamp=conv_clamp,                                                  linformer=linformer
                                    )
                self.num_ws += block.num_attention
                if is_last:
                    self.num_ws += block.num_torgb
                setattr(self, f'b{res}_{j}', block)
    

    def forward(self, ws=None, image=None):
        #self.image=image
        block_ws = []
        #self.image=image
    
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for i, res in enumerate(self.block_resolutions):
                num_block_res = self.num_block[i]
                res_ws = []
                for j in range(num_block_res):
                    block = getattr(self, f'b{res}_{j}')
                    res_ws.append(ws.narrow(1, w_idx, block.num_attention + block.num_torgb))
                    w_idx += block.num_attention
                block_ws.append(res_ws)
        #global x3 ,x6,x7        
        x = img=None
        #image = None
        gol._init()
        for i, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            #print('res',res)
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                block = getattr(self, f'b{res}_{j}')
                x, img,a1,a2,a3 = block(x, img,image, cur_ws[j])
                if  x.shape[0]==32 and j==0 and res==8 and a1.shape[0]==32:
                    
                    gol.set_value('x3',a1)
                    gol.set_value('x6',a2)
                    gol.set_value('x7',a3)

                    #x3 = a1
                    #x6 = a2
                    #x7 = a3
        #x3 =  gol.get_value('x3')      
        #print('x3.shape:',a1.size())
        #print('x6.shape',a2.size())
        #print('x7.shape',a3.size())
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels,  mapping_kwargs ={}, synthesis_kwargs= {}):
                    # ocae_decoder_capsule_params={}, ocae_encoder_set_transformer_params={}, pcae_cnn_encoder_params={},
                    # pcae_decoder_params={}, pcae_encoder_params={}, pcae_template_generator_params={}):
        super().__init__()
        gol._init()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        #self.part_encoder=part_encoder
        #self,image=image
        #self.input_shape=input_shape
        #self.image_shape = image_shape
        #self.strides=strides
        #part_encoder =  CapsuleImageEncoder() 
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        #-----------------------SCAE-----------------------
        #self.CNNEncoder = CNNEncoder(input_shape=image_shape, strides=strides, activation=nn.ReLU, **pcae_cnn_encoder_params) 
        #self.CapsuleImageEncoder=CapsuleImageEncoder(input_shape=image_shape, encoder=encoder, **pcae_encoder_params)
        #self.SetTransformer=SetTransformer(**ocae_encoder__set_transformer_params)
        #self.CapsuleLayer=CapsuleLayer(**ocae_decoder_capsule_params)
        #-----------------------SCAE-----------------------       
        
    def forward(self, z, c, image, truncation_psi=1, truncation_cutoff=None, epoch=None, **synthesis_kwargs):
        #cnn_encoder = CNNEncoder(n_templates= n_templates, template_size= template_size, output_size= output_size, **config.pcae_cnn_encoder)
        #part_encoder = CapsuleImageEncoder(encoder=cnn_encoder, **config.pcae_encoder)
        #print('image.shape_g:',image.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        image = image.float()
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        output = self.synthesis(ws, image)  
         
        return output
        #print('image.shape_g:',image.shape)




