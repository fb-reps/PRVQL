import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from bert_model.bert_module import BertLayer_Cross
from updatequery.utils import get_top_k_bbox, get_topK_bbox_feature_tenor
from updateclipfeature.update_atten_mask import update_attention_mask
from utils.model_utils import positionalencoding3d
from utils.model_utils import BasicBlock_Conv2D
from utils.anchor_utils import generate_anchor_boxes_on_regions
from dataset.dataset_utils import bbox_xyhwToxyxy
from einops import rearrange
import math
import torchvision
from dataset import dataset_utils
from model.mae import vit_base_patch16
from easydict import EasyDict as EDict

from updatequery import transformer as customize_transformer

base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    #4 types of size
aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32)                                #3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]

import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch

flag_vis=False

def extract_sub_modules_state_dict(checkpoint, prefix="clipMatcher_base.backbone."):
    """
    Extracts and renames keys from a flattened checkpoint state dict.

    Args:
        checkpoint (dict): Loaded checkpoint dictionary.
        prefix (str): The prefix to filter backbone-related keys.

    Returns:
        dict: A dictionary containing only the backbone weights with renamed keys.
    """
    return {k.replace(prefix, ""): v for k, v in checkpoint.items() if k.startswith(prefix)}


def visualize_feature_maps(feature_maps, channels_to_visualize=[0, 127, 255], image_name="sub_folder"):
    """
    Visualize selected channels of a feature map tensor as grayscale images.

    Parameters:
    - feature_maps: PyTorch tensor of shape (channels, height, width).
    - channels_to_visualize: List of channel indices to visualize.
    - image_name: Prefix of the output image file names.
    """
    # Check if the 'debug_vis' directory exists, if not, create it
    if flag_vis:
        save_dir = f'debug_vis/{image_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Ensure the feature_maps is a PyTorch tensor
        if not isinstance(feature_maps, torch.Tensor):
            raise TypeError("feature_maps must be a PyTorch tensor")
        
        # Convert to a NumPy array
        feature_maps_np = feature_maps.detach().cpu().numpy()
        
        
        # Visualize each channel
        for channel_idx in channels_to_visualize:
            if channel_idx >= feature_maps.size(0):
                raise IndexError(f"Channel index {channel_idx} is out of bounds for dimension 0 with size {feature_maps.size(0)}")
            
            # Create a figure for each channel
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(feature_maps_np[channel_idx, :, :], cmap='gray')
            ax.axis('off')  # Hide the axis
            ax.set_title(f'Channel {channel_idx}')
            
            # Save the figure for each channel with an incremented image name
            plt.savefig(os.path.join(save_dir, f'{channel_idx}.png'))
            plt.close(fig)  # Close the figure to free up memory

def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dino':
        assert type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_{}'.format(type))
        down_rate = int(type.replace('vitb', '').replace('vits', ''))
        backbone_dim = 768
        if type == 'vitb16' and config.model.bakcbone_use_mae_weight:
            mae_weight = torch.load('/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')['model']
            backbone.load_state_dict(mae_weight)
    elif name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))
        down_rate = 14
        if type == 'vitb14':
            backbone_dim = 768
        elif type == 'vits14':
            backbone_dim = 384
    elif name == 'mae':
        backbone = vit_base_patch16()
        cpt = torch.load('/vision/hwjiang/download/model_weight/mae_pretrain_vit_base.pth')['model']
        backbone.load_state_dict(cpt, strict=False)
        down_rate = 16
        backbone_dim = 768
    return backbone, down_rate, backbone_dim

class ClipEncoder(nn.Module):
    def __init__(self, config, down_rate, backbone_dim) -> None:
        super().__init__()
        self.config = config

        self.down_rate, self.backbone_dim = down_rate,backbone_dim

        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer
        self.resolution_anchor_feat = config.model.resolution_anchor_feat

        self.anchors_xyhw = generate_anchor_boxes_on_regions(image_size=[self.clip_size_coarse, self.clip_size_coarse],
                                                        num_regions=[self.resolution_anchor_feat, self.resolution_anchor_feat])
        self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   #[R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)

        #query down heads
        self.query_down_heads = []
        for _ in range(int(math.log2(self.query_feat_size))):
            self.query_down_heads.append(
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.query_down_heads = nn.ModuleList(self.query_down_heads)

        
        
        #clip-query correspondence
        self.CQ_corr_transformer = []
        for _ in range(1):
            self.CQ_corr_transformer.append(
                customize_transformer.TransformerDecoderLayer(
                    d_model=256,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        #feature downsample layers
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(self.num_head_layers-1):
            self.in_channel = 256 if i != 0 else self.backbone_dim
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        #spatial-temporal PE
        self.pe_3d = positionalencoding3d(d_model=256, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        #spatial-temporal transformer layer
        self.feat_corr_transformer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            self.feat_corr_transformer.append(
                    customize_transformer.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None

        #output head
        self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.resolution_anchor_feat)

    def replicate_for_hnm(self, query_feat, clip_feat):
        '''
        query_feat in shape [b,c,h,w]
        clip_feat in shape [b*t,c,h,w]
        '''
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b
        
        clip_feat = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)

        new_clip_feat, new_query_feat = [], []
        for i in range(b):
            for j in range(b):
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[j])

        new_clip_feat = torch.stack(new_clip_feat)      #[b^2,t,c,h,w]
        new_query_feat = torch.stack(new_query_feat)    #[b^2,c,h,w]

        new_clip_feat = rearrange(new_clip_feat, 'b t c h w -> (b t) c h w')
        return new_clip_feat, new_query_feat

    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask
    
    def forward(self, clip_feat, query_feat, training=False,b=4,t=30,h=32,w=32):
        
        
        #find spatial correspondence between query-frame
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')  #[b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                         #[b*t,n,c]
        qk_S=[]
        qkv_S=[]
        for layer in self.CQ_corr_transformer:
            clip_feat, qk_S,qkv_S = layer(clip_feat, query_feat)                                                     #[b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)                               #[b*t,c,h,w]
        #down-size features and find spatial-temporal correspondence
        qk_ST=[]
        qkv_ST=[]
        for head in self.down_heads:
            clip_feat = head(clip_feat)
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                
                for layer in self.feat_corr_transformer:
                    clip_feat,qk_ST,qkv_ST = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break
        
        #refine anchors
        anchors_xyhw = self.anchors_xyhw.to(clip_feat.device)                   #[n,4]
        anchors_xyxy = self.anchors_xyxy.to(clip_feat.device)                   #[n,4]
        anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           #[1,1,n,4]
        anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           #[1,1,n,4]
        
        bbox_refine, prob = self.head(clip_feat)                                #[b*t,n=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  #[b,t,N,4], in xyhw frormulation
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                #[b,t,n,1]
        bbox_refine += anchors_xyhw                                             #[b,t,n,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           #represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           #anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    #[b,t,n,4]

        result = {
            'center': center,           #[b,t,n,2]
            'hw': hw,                   #[b,t,n,2]
            'bbox': bbox,               #[b,t,n,4]
            'prob': prob.squeeze(-1),   #[b,t,n]
            'anchor': anchors_xyxy,      #[1,1,n,4]
            'atten_weight_S': qk_S, #[120,1024,1024]
            'atten_weight_ST':qk_ST,
            'atten_S': qkv_S, #[120,1024,256]
            'atten_ST': qkv_ST #[4,1920,256]
        }
        return result

class convolutionBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3, stride=1, padding=1, dilation=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpdateQuery(nn.Module):
    def __init__(self, inchannel,outchannel) -> None:
        super().__init__()
        self.conv1 = convolutionBlock(inchannel, 64)
        self.conv2 = convolutionBlock(64, 128)
        self.conv3 = convolutionBlock(128, 256)
        self.conv4 = convolutionBlock(256, outchannel=outchannel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        bc, h, w = x.shape
        x = x.view(bc, -1)  # Flatten spatial dimensions
        x = self.fc2(self.relu(self.fc1(x)))
        x = x.view(bc, h, w)  # Restore shape
        return x



class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        
        self.config = config
        self.topK_nums = config.iterations.topK_nums
        self.topK_threshold = config.iterations.topK_threshold
        self.query_iter= config.iterations.query_iter

        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.clipencoderlayers = nn.ModuleList([ClipEncoder(config,self.down_rate,self.backbone_dim) for _ in range(self.query_iter)])
        self.last_clipencoderlayer=ClipEncoder(config,self.down_rate,self.backbone_dim)

        bert_config = EDict(
            num_attention_heads=8,
            hidden_size=256,
            attention_head_size=256,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=256
        )
        self.ca=BertLayer_Cross(bert_config)
        self.topk_conv=convolutionBlock(inchannel=256*3,outchannel=256,stride=1,padding=1)

        self.s_mlp=MLP(32*32, 1024, 32*32)
        self.t_mlp=MLP(32*32, 1024, 32*32)
        self.ks_mlp=MLP(32*32, 1024, 32*32)

        #feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )


    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  #we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'mae':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.forward_features(x) #[b,1+h*w,c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out[:,1:].reshape(b, h, w, dim).permute(0,3,1,2)  #[b,c,h,w]
            out = F.interpolate(out, size=(16,16), mode='bilinear')
            if return_h_w:
                return out, h, w
            return out
        
    def process_crossatten_features(self, query_feat, topk_feat):
        ''''
        topk_feat: b k c 5 5
        query_feat: b c 32 32
        '''

        topk_feat = rearrange(topk_feat, 'b k c h w -> b  (k c) h w')

        topk_feat = self.topk_conv(topk_feat)

        #Rearrange shapes for Channel Attention
        query_feat = rearrange(query_feat, 'b c h w -> b  (h w) c')
        topk_feat = rearrange(topk_feat, 'b c h w -> b  (h w) c')

        #Perform Channel Attention operation
        query_feat = self.ca(q=query_feat, kv=topk_feat)

        #Rearrange query features to the original shape
        query_feat = rearrange(query_feat, 'b  (h w) c -> b c h w', h=32)


        return query_feat
        
    def forward(self, clip, query, query_frame_bbox=None, training=False, fix_backbone=True):
        '''
        clip: in shape [b,t,c,h,w]
        query: in shape [b,c,h2,w2]
        '''
        b, t = clip.shape[:2]
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')
        #get backbone features
        if fix_backbone:
            with torch.no_grad():
                from torchkeras import summary
                #print(summary(self.extract_feature, input_data_args=query))
                query_feat = self.extract_feature(query)
                clip_feat = self.extract_feature(clip)
        else:
            query_feat = self.extract_feature(query)        #[b c h w]
            clip_feat = self.extract_feature(clip)          #(b t) c h w
        c, h, w = clip_feat.shape[-3:]

        # visualize_feature_maps(query_feat[0],channels_to_visualize=[1],image_name="query_feat_after_extract_feature")
        query_feat_after_extract_feature_vis=query_feat.clone().detach()

        if torch.is_tensor(query_frame_bbox) and self.config.train.use_query_roi:
            idx_tensor = torch.arange(b, device=clip.device).float().view(-1, 1)
            query_frame_bbox = dataset_utils.recover_bbox(query_frame_bbox, h, w)
            roi_bbox = torch.cat([idx_tensor, query_frame_bbox], dim=1)
            query_feat = torchvision.ops.roi_align(query_feat, roi_bbox, (h,w))
        
        #reduce channel size
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        # visualize_feature_maps(query_feat[0],channels_to_visualize=[1],image_name="query_feat_after_reduce_channel_size")
        query_feat_after_reduce_channel_size_vis=query_feat.clone().detach()
        clip_feat_after_reduce_channel_size_vis = clip_feat.clone().detach()

        if self.config.train.use_hnm and training:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   #b -> b^2
            b = b**2

        results = []
        original_top_k_indices=[]
        original_top_k_scores=[]
        original_top_k_bbox=[]
        update_atten_mask=[]
        clip_feat_copy = clip_feat.clone()
        for layer_id, layer in enumerate(self.clipencoderlayers):
            
            result = layer(clip_feat, query_feat,training=training,b=b,t=t,h=h,w=w)
            results.append(result)
            pred_bbox = result['bbox']
            pred_prob = result['prob']
            atten_S=result['atten_weight_S'] #bt,hw,d
            atten_ST=result['atten_weight_ST'] #b,th1w1,d


            original_top_k_indices, original_top_k_scores, original_top_k_bbox = get_top_k_bbox(pred_prob, pred_bbox, k=self.topK_nums, t=30, threshold=self.topK_threshold)
            # visualize_feature_maps(query_feat[0],channels_to_visualize=[127],image_name="query_feat_after_get_top_k_bbox")
            query_feat_after_get_top_k_bbox_vis=query_feat.clone().detach()
            
            feat_frames_query_tensor = get_topK_bbox_feature_tenor(query_feat=query_feat,
                                                                   clip_feat=clip_feat, 
                                                                   original_top_k_indices=original_top_k_indices, 
                                                                   original_top_k_bbox=original_top_k_bbox, 
                                                                   k=self.topK_nums,
                                                                   t=30, 
                                                                   threshold=self.topK_threshold, 
                                                                   original_top_k_scores=original_top_k_scores)
            #feat_frames_query_tensor = rearrange(feat_frames_query_tensor, 'b k c h w -> b (k c) h w')
            topk_feat_vis = torch.unbind(feat_frames_query_tensor, dim=1)

            #visual query features in process
            query_feat_after_roi_vis=query_feat.clone().detach()
            topk1_feat_after_roi_vis=topk_feat_vis[0].clone().detach()
            topk2_feat_after_roi_vis=topk_feat_vis[1].clone().detach()
            topk3_feat_after_roi_vis=topk_feat_vis[2].clone().detach()

            query_feat=self.process_crossatten_features(query_feat, feat_frames_query_tensor)

            clip_feat_before_update_mask_vis=clip_feat.clone().detach()

            ##update clip_feature
            update_atten_mask,mask_S,mask_ST = update_attention_mask(self.s_mlp, self.t_mlp, self.ks_mlp, atten_S,atten_ST,w_s=0.5,w_st=0.5) # bt hw d

            clip_feat = 0.1*(update_atten_mask*clip_feat)+(1-0.1)*clip_feat

            # visualize_feature_maps(query_feat[0],channels_to_visualize=[127],image_name="query_feat_after_merge_model")
            query_feat_after_merge_model_vis=query_feat.clone().detach()


        clip_feat_copy = clip_feat.clone()
        result = self.last_clipencoderlayer(clip_feat, query_feat,training=training,b=b,t=t,h=h,w=w)
        

        topk_dict = {'original_top_k_indices':original_top_k_indices, 
                     'original_top_k_scores':original_top_k_scores, 
                     'original_top_k_bbox':original_top_k_bbox}
        featrue_vis={
            # 'query_vis':query_vis,
            'query_feat_after_extract_feature_vis':query_feat_after_extract_feature_vis,
            'query_feat_after_reduce_channel_size_vis':query_feat_after_reduce_channel_size_vis,
            'query_feat_after_get_top_k_bbox_vis':query_feat_after_get_top_k_bbox_vis,
            'query_feat_after_roi_vis':query_feat_after_roi_vis,
            'topk1_feat_after_roi_vis':topk1_feat_after_roi_vis,
            'topk2_feat_after_roi_vis':topk2_feat_after_roi_vis,
            'topk3_feat_after_roi_vis':topk3_feat_after_roi_vis,
            'query_feat_after_merge_model_vis':query_feat_after_merge_model_vis

        }
        atten_map = {
            'clip_feat_after_reduce_channel_size_vis':clip_feat_after_reduce_channel_size_vis,
            'clip_feat_before_update_mask_vis':clip_feat_before_update_mask_vis,
            'atten_S':mask_S,
            'atten_ST':mask_ST,
            'update_weight': update_atten_mask,
            'update_clip': clip_feat_copy,
        }
        result['topk_dict'] = topk_dict
        result['featrue_vis'] = featrue_vis
        result['atten_map'] = atten_map
        
        results.append(result)


        return result, results
    

    
    


class Head(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3
        
        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        if self.num_up_layers > 0:
            x = self.up_convs(x.clone())  #Create a copy of the upsampling operation

        B, c, h, w = x.shape

        #Use the clone() method to create a copy of the input to avoid in-place modifications
        feat_reg, feat_cls = self.in_conv(x.clone()).split([c, c], dim=1)

        #Use dropout operation here
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)
        feat_cls = self.classification_conv(feat_cls)

        out_reg = self.regression_head(feat_reg)
        out_cls = self.classification_head(feat_cls)

        out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

        return out_reg, out_cls
