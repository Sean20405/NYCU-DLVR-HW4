import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from kornia.color import rgb_to_lab


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[2, 7, 12, 21], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.ModuleList()
        
        # Extract features from specified layers
        self.feature_layers = feature_layers
        for i in range(max(feature_layers) + 1):
            layer = vgg[i]
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            self.feature_extractor.append(layer)
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        if use_gpu:
            self.feature_extractor = self.feature_extractor.cuda()
    
    def forward(self, x, y):
        # Normalize inputs to [0, 1] range for VGG
        if x.max() > 1.0:
            x = x / 255.0
        if y.max() > 1.0:
            y = y / 255.0
            
        # VGG expects inputs in range [0, 1] but normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        x_features = x
        y_features = y
        
        for i, layer in enumerate(self.feature_extractor):
            x_features = layer(x_features)
            y_features = layer(y_features)
            
            if i in self.feature_layers:
                loss += nn.functional.mse_loss(x_features, y_features)
        
        return loss
    

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Expand to 3 channels (RGB)
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # Register as buffer so they move to the correct device automatically
        self.register_buffer('sobel_x_kernel', self.sobel_x)
        self.register_buffer('sobel_y_kernel', self.sobel_y)
        
    def get_edges(self, x):
        # Apply Sobel filters
        edge_x = nn.functional.conv2d(x, self.sobel_x_kernel, padding=1, groups=3)
        edge_y = nn.functional.conv2d(x, self.sobel_y_kernel, padding=1, groups=3)
        
        # Calculate edge magnitude
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges
    
    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        # Calculate L1 loss between edge maps
        edge_loss = nn.functional.l1_loss(pred_edges, target_edges)
        return edge_loss
    

class CombinedLoss(nn.Module):
    """
    Combine L2 loss and perceptual loss for training.
    """
    def __init__(self, l2_weight=5.0, perceptual_weight=0.001, edge_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        l2_loss = self.l2_loss(pred, target) * self.l2_weight
        perceptual_loss = self.perceptual_loss(pred, target) * self.perceptual_weight
        edge_loss = self.edge_loss(pred, target) * self.edge_weight
        
        total_loss = l2_loss + perceptual_loss + edge_loss
        return total_loss, {
            'l2_loss': l2_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'edge_loss': edge_loss.item()
        }
    

class CombinedLossWoVgg(nn.Module):
    """
    Combine L2 loss and perceptual loss for training.
    """
    def __init__(self, l2_weight=5.0, edge_weight=0.1):
        super(CombinedLossWoVgg, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.edge_loss = EdgeLoss()
        self.l2_weight = l2_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        l2_loss = self.l2_loss(pred, target) * self.l2_weight
        edge_loss = self.edge_loss(pred, target) * self.edge_weight
        
        total_loss = l2_loss + edge_loss
        return total_loss, {
            'l2_loss': l2_loss.item(),
            'edge_loss': edge_loss.item()
        }


class CombinedLossWoVggColor(nn.Module):
    """
    Combine L2 loss + edge loss + color loss for training.
    """
    def __init__(self, l2_weight=5.0, edge_weight=0.1, color_weight=0.01):
        super(CombinedLossWoVggColor, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.edge_loss = EdgeLoss()
        self.l1_loss = nn.L1Loss()  # Used for color loss
        self.l2_weight = l2_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight
    
    def forward(self, pred, target):
        l2_loss = self.l2_loss(pred, target) * self.l2_weight
        edge_loss = self.edge_loss(pred, target) * self.edge_weight

        # Color loss
        lab_pred = rgb_to_lab(pred / 255.0)  # Normalize to [0, 1] before conversion
        lab_gt = rgb_to_lab(target)
        color_loss = self.l1_loss(lab_pred[:, 1:, :, :], lab_gt[:, 1:, :, :]) * self.color_weight
        
        total_loss = l2_loss + edge_loss + color_loss
        return total_loss, {
            'l2_loss': l2_loss.item(),
            'edge_loss': edge_loss.item(),
            'color_loss': color_loss.item()
        }