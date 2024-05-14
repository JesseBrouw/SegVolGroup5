from __future__ import annotations

from typing import Sequence, Union

import escnn
from escnn import nn as enn

import torch
from torch import nn

import numpy as np

from monai.networks.blocks.transformerblock import TransformerBlock

from monai.utils import ensure_tuple_rep

# Modified from https://github.com/Project-MONAI/MONAI
class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = SO3SteerablePatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out

class SO3SteerablePatchEmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super(SO3SteerablePatchEmbeddingBlock).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        # The model will be equivariant under planer rotations
        self.r3_act = enn.gspaces.rot3dOnR3()

        # The group SO(3)
        self.Group = self.r3_act.fibergroup

        # The input field is a scalar field, because we are dealing with 3D-gray images (CT scans)
        in_type = enn.FieldType(self.r3_act, [self.r3_act.trivial_repr])
        self.input_type = in_type
        # The output is still a scalar, but we use #hidden_size channels to create the embeddings for the ViT
        out_type = enn.FieldType(self.r3_act, hidden_size*[self.r3_act.trivial_repr])
        
        # The 3D group equivariant convolution, that performs one pass on each block (similar to normal ViT)
        self.patch_embeddings = enn.R3Conv(in_type, out_type, kernel_size=patch_size, stride=patch_size)

        # Create the learnable position embeddings
        self.position_embeddings = torch.nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        torch.nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.patch_embeddings(x).tensor
        
        # Flatten all spatial dimensions into one, and add the learnable encoding
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return embeddings

    def check_equivariance(self):
        with torch.no_grad():
            x = torch.randn(3, 1, 32, 32 , 32)
            
            # the volumes rotated by 90 degrees in the ZY plane (i.e. around the X axis)
            x_x90 = x.rot90(1, (2, 3))
            x_x90 = enn.GeometricTensor(x_x90, self.input_type)
            yx = self.patch_embeddings(x_x90).tensor

            # the volumes rotated by 90 degrees in the XZ plane (i.e. around the Y axis)
            x_y90 = x.rot90(1, (2, 4))
            x_y90 = enn.GeometricTensor(x_y90, self.input_type)
            yy = self.patch_embeddings(x_y90).tensor

            # the volumes rotated by 90 degrees in the YX plane (i.e. around the Z axis)
            x_z90 = x.rot90(1, (3, 4))
            x_z90 = enn.GeometricTensor(x_z90, self.input_type)
            yz = self.patch_embeddings(x_z90).tensor

            # the volumes rotated by 180 degrees in the XZ plane (i.e. around the Y axis)
            x_y180 = x.rot90(2, (2, 4))
            x_y180 = enn.GeometricTensor(x_y180, self.input_type)
            yy180 = self.patch_embeddings(x_y180).tensor

            x = enn.GeometricTensor(x, self.input_type)
            y = self.patch_embeddings(x).tensor

            # Rotate the outputs back to the original orientation, and check whether output matches with y.
            print('TESTING INVARIANCE:                     ')
            print('90 degrees ROTATIONS around X axis:  ' + ('Equiv' if torch.allclose(y, yx.rot90(3, (2,3)), atol=1e-3) else 'False'))
            print('90 degrees ROTATIONS around Y axis:  ' + ('Equiv' if torch.allclose(y, yy.rot90(3, (2,4)), atol=1e-3) else 'False'))
            print('90 degrees ROTATIONS around Z axis:  ' + ('Equiv' if torch.allclose(y, yz.rot90(3, (3,4)), atol=1e-3) else 'False'))
            print('180 degrees ROTATIONS around Y axis: ' + ('Equiv' if torch.allclose(y, yy180.rot90(2, (2,4)), atol=1e-4) else 'False'))

    
    

