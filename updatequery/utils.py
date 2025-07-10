from einops import rearrange
import torch
import torchvision


def get_top_k_bbox(pred_prob, pred_bbox, k, t, threshold=0.6):
    b, t, N = pred_prob.shape

    pred_prob = rearrange(pred_prob, 'b t N -> (b t) N') #bt N

    pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c', N=N) #bt N c




    #get max one
    pred_prob_top, top_idx = torch.max(pred_prob, dim=-1)
    pred_bbox_top = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4)).squeeze()
    #get top k pro
    pred_prob_top = rearrange(pred_prob_top, '(b t) -> b t', t=t)
    top_k_values, top_k_indices = torch.topk(pred_prob_top, k=k, dim=-1)
  
    pred_bbox_top = rearrange(pred_bbox_top, '(b t) x -> b t x', t=t)
    top_k_bbox = torch.gather(pred_bbox_top, dim=1, index=top_k_indices.unsqueeze(-1).repeat(1, 1, 4))

    return top_k_indices, top_k_values, top_k_bbox


def get_topK_bbox_feature_tenor(query_feat,clip_feat, original_top_k_indices, original_top_k_bbox, k,original_top_k_scores, t, threshold):
    '''
    clip: in shape [bt,,c,h,w] eg:120, 256, 32, 32
    query: in shape [b,c,h2,w2], eg:4, 256, 32, 32
    original_top_k_bbox : in shape [b,k,4]
    original_top_k_indices : in shape [b,k,1]
    original_top_k_scores : in shape [b,k,1]
    return: dic with frame in bbox erea
    '''

    clip_feat = rearrange(clip_feat, '(b t)c h w -> b t c h w', t=t)    #b t c h w
    b,t, _, hc, wc = clip_feat.shape
    h_roi,w_roi=5,5
    _,c,_,_=query_feat.shape
    top_k_bboxes = original_top_k_bbox

    feat_frames = []
    
    #Create an all-zero tensor with the same shape as query_feat[b_idx]
    zero_tensor = torch.zeros(c, h_roi, w_roi)

    #If you need this tensor on a specific device, such as a GPU, you can use the to() method
    zero_tensor = zero_tensor.to(query_feat.device)

    for b_idx in range(b):
        batch_feat_frames = []
        for k_idx in range(k):
            if torch.sigmoid(original_top_k_scores[b_idx][k_idx]) < threshold:
                batch_feat_frames.append(zero_tensor)
            else:
                top_k_idx=original_top_k_indices[b_idx][k_idx] #top k idx in clip t dim
                frame_feat = clip_feat[b_idx][top_k_idx]  #Get the k-th frame of the b-th batch  c h w

                bbox = top_k_bboxes[b_idx][k_idx]  #Get the bounding boxes for the k-th frame of the b-th batch : x1 y1 x2 y2
                
                

                # Boundary box coordinates [x1, y1, x2, y2], the value is also between 0-1
                bbox = torch.clone(bbox)

                bbox = recover_bbox(bbox, hc, wc)

                #Lose it Roy
                idx_tensor = torch.arange(1, device=query_feat.device).float().view(-1, 1)
                #Print(idx tensor.shape,bbox.shape,idx tensor,'idx tensor.shape,bbox.shape,idx tensor')
                bbox=bbox.unsqueeze(0) #1 4
                frame_feat=frame_feat.unsqueeze(0) #1 c h w
                roi_bbox = torch.cat([idx_tensor, bbox], dim=1) #1 5 : 0 x1 y1 x2 y2
                #Define output size and spatial scale
                frame_box_feat = torchvision.ops.roi_align(frame_feat, 
                                                           roi_bbox,
                                                           (h_roi, w_roi))

                frame_box_feat = frame_box_feat.squeeze(0)
                #Print(frame box feat.shape,"frame box feat")
                batch_feat_frames.append(frame_box_feat)

        batch_feat_frames=torch.stack([batch_feat_frames[i] for i in range(len(batch_feat_frames))])   
        feat_frames.append(batch_feat_frames)
    feat_frames_tensor=torch.stack([feat_frames[i] for i in range(len(feat_frames))])
    #Print(feat frames tensor.shape,'feat frames tensor.shape')
    return feat_frames_tensor


def recover_bbox(bbox, h, w):
    '''
    Recovers the bounding box coordinates to the original size.
    
    Args:
    - bbox (torch.Tensor): Bounding box coordinates in shape [4] or [...,4].
    - h (int): Original height.
    - w (int): Original width.
    
    Returns:
    - bbox (torch.Tensor): Recovered bounding box coordinates.
    '''
    bbox_cp = bbox
    
    #Scale the bounding box coordinates to the original size
    if len(bbox.shape) > 1: #[n,4]
        bbox_cp[..., 0] *= h
        bbox_cp[..., 1] *= w
        bbox_cp[..., 2] *= h
        bbox_cp[..., 3] *= w
    else:
        bbox_cp[0] *= h
        bbox_cp[1] *= w
        bbox_cp[2] *= h
        bbox_cp[3] *= w
    
    return bbox_cp
