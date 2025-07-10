import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
from dataset import dataset_utils
import torch
from einops import rearrange
import numpy as np


def vis_pred_clip(sample, preds, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    clip = sample['clip_origin'].detach().cpu()        # [B,T,3,H,W]
    query = sample['query_origin'].detach().cpu()      # [B,3,H2,W2]
    query_aug = sample['query'].detach().cpu()         # [B,3,H2,W2]
    bbox = sample['clip_bbox'].detach().cpu()          # [B,T,4]
    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    B, T, _, H, W = clip.shape
    _, _, H2, W2 = query_aug.shape
    
    for pred_id, pred in enumerate(preds):

        bbox_pred = pred['bbox'].detach().cpu()            # [B,T,4]
        prob_pred = pred['prob'].detach().cpu()            # [B,T]

        for i in range(B):
            frames = []
            cur_clip, cur_query = clip[i], query[i]                                     # [T,3,H,W], [3,H2,W2]
            cur_bbox, cur_bbox_pred = bbox[i], bbox_pred[i].clamp(min=0.0, max=1.0)     # [T,4]
            cur_prob, cur_prob_pred = prob[i], prob_pred[i]                             # [T]

            cur_query = cur_query.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()        # [H2,W2,3]
            for j in range(T):
                # draw clips with bbox
                img = cur_clip[j].clamp(min=0.0, max=1.0)                               
                img = img.permute(1,2,0).numpy()                # [H,W,3]
                fig, ax = plt.subplots(1,2, dpi=100)
                fig.suptitle('Prob: gt {:.3f}, pred {:.3f}'.format(cur_prob[j].item(), torch.sigmoid(cur_prob_pred[j]).item()), fontsize=20)
                ax[0].imshow(img)
                ax[1].imshow(cur_query)
                if cur_prob[j].item() > 0.5:
                    draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[j], H, W)  # [4]
                    rect = patches.Rectangle((draw_bbox_gt[1], draw_bbox_gt[0]), 
                                            draw_bbox_gt[3]-draw_bbox_gt[1], draw_bbox_gt[2]-draw_bbox_gt[0], 
                                            linewidth=1, edgecolor='r', facecolor='none')
                    ax[0].add_patch(rect)
                if cur_prob[j].item() > 0.5:
                    draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                    rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                            draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                            linewidth=1, edgecolor='g', facecolor='none')
                    ax[0].add_patch(rect)
                if torch.sigmoid(cur_prob_pred[j]).item() > 0.5:
                    draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                    rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                            draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                            linewidth=1, edgecolor='b', facecolor='none')
                    ax[0].add_patch(rect)
                plt.savefig(os.path.join(output_dir, 'tmp.png'))
                plt.close()
                frames.append(cv2.imread(os.path.join(output_dir, 'tmp.png'))[...,::-1])
            save_name = os.path.join(output_dir, '{}_{}_{}.gif'.format(iter_num, i, pred_id))
            imageio.mimsave(save_name, frames, 'GIF', duration=0.2)


def vis_pred_scores(sample, preds, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    B, T = prob.shape

    for pred_id, pred in enumerate(preds):
        prob_pred = pred['prob'].detach().cpu()            # [B,T]
        if 'gt_iou' in pred.keys():
            prob_iou = pred['gt_iou'].detach().cpu()            # [B,T]
        if 'prob_refine' in pred.keys():
            prob_refine = pred['prob_refine'].detach().cpu()            # [B,T]

        for i in range(B):
            cur_prob, cur_prob_pred = prob[i].numpy(), torch.sigmoid(prob_pred[i]).numpy()     # [T]
            x = np.arange(T)
            plt.plot(x, cur_prob_pred, marker=None, color='b', label='pred')
            plt.plot(x, cur_prob, marker=None, color='r', label='gt')
            if 'prob_refine' in pred.keys():
                cur_prob_refine = torch.sigmoid(prob_refine[i]).numpy()
                plt.plot(x, cur_prob_refine, marker=None, color='g', label='pred')
            if 'gt_iou' in pred.keys():
                cur_prob_iou = prob_iou[i].numpy() * 0.9
                plt.plot(x, cur_prob_iou, marker=None, color='c', label='pred')
            plt.xlabel('number of frames')
            plt.ylabel('occurance score')
            plt.ylim((0.0, 1.05))
            plt.legend(loc='best')
            save_name = os.path.join(output_dir, '{}_{}_{}.jpg'.format(iter_num, i, pred_id))
            plt.savefig(save_name)
            plt.close()


def vis_pred_clip_inference(clips, queries, pred, save_path, iter_num):
    #clips = clips.detach().cpu()            # [b,t,c,h,w]
    queries = queries.detach().cpu()        # [c,h,w]
    # bbox = pred['bbox_raw']                 # [b*t,4]
    # prob = torch.sigmoid(pred['prob_raw'])  # [b*t]
    bbox = pred['bbox']                 # [b*t,4]
    prob = torch.sigmoid(pred['prob'])  # [b*t]
    save_name = save_path + f'_{iter_num}.mp4'
    writer = imageio.get_writer(save_name, fps=5)

    #clips = rearrange(clips, 'b t c h w -> (b t) c h w')

    T, _, H, W = clips.shape
    _, H2, W2 = queries.shape

    frames = []
    for i in range(T):
        cur_clip = clips[i].clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_query = queries.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_bbox = bbox[i]#.clamp(min=0.0, max=1.0)
        cur_prob = prob[i]

        fig, ax = plt.subplots(1,2)
        fig.suptitle('Prob {:.3f}'.format(cur_prob.item()), fontsize=20)
        ax[0].imshow(cur_clip)
        ax[1].imshow(cur_query)
        if cur_prob.item() > 0.5:
            draw_bbox_pred = cur_bbox #dataset_utils.recover_bbox(cur_bbox, H, W)  # [4]
            rect = patches.Rectangle((draw_bbox_pred[0], draw_bbox_pred[1]), 
                                      draw_bbox_pred[2]-draw_bbox_pred[0], draw_bbox_pred[3]-draw_bbox_pred[1], 
                                      linewidth=1, edgecolor='b', facecolor='none')
            ax[0].add_patch(rect)
        plt.savefig(save_path + '_tmp.jpg')
        plt.close()
        writer.append_data(cv2.imread(save_path + '_tmp.jpg')[...,::-1])
    writer.close()


def vis_pred_topk(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    topk_dic=pred['topk_dict']
    original_top_k_indices=topk_dic['original_top_k_indices']
    original_top_k_scores=topk_dic['original_top_k_scores']
    original_top_k_bbox=topk_dic['original_top_k_bbox']

    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    clip = sample['clip_origin'].detach().cpu()        # [B,T,3,H,W]
    query = sample['query_origin'].detach().cpu()      # [B,3,H2,W2]
    # query_aug = sample['query'].detach().cpu()         # [B,3,H2,W2]
    bbox = sample['clip_bbox'].detach().cpu()          # [B,T,4]
    # bbox_pred = pred['bbox'].detach().cpu()            # [B,T,4]
    top_k_indices_cpu = original_top_k_indices.detach().cpu()            # [B,k]
    top_k_scores_cpu = original_top_k_scores.detach().cpu()            # [B,k]
    top_k_bbox_cpu = original_top_k_bbox.detach().cpu()            # [B,k,4]

    


    B, T, _, H, W = clip.shape
    # _, _, H2, W2 = query_aug.shape
    _,K=top_k_indices_cpu.shape

    for i in range(B):
        frames = []
        cur_clip, cur_query = clip[i], query[i]                                     # [T,3,H,W], [3,H2,W2]
        cur_bbox = bbox[i]     # [T,4]
        cur_prob = prob[i]                             # [T]

        cur_topk_scores = top_k_scores_cpu[i] # [k]
        cur_topk_indices = top_k_indices_cpu[i] # [k]
        cur_topk_bbox = top_k_bbox_cpu[i] # [k,4]


        cur_query = cur_query.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()        # [H2,W2,3]
        for j in range(K):
            # draw clips with bbox
            ori_idx=cur_topk_indices[j]
            img = cur_clip[ori_idx].clamp(min=0.0, max=1.0)                               
            img = img.permute(1,2,0).numpy()                # [H,W,3]
            fig, ax = plt.subplots(1,2, dpi=100)
            draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)
            draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[ori_idx], H, W)

            # Use .item() to get a single numeric value
            cur_prob_value = cur_prob[ori_idx].item()
            cur_pred_value = torch.sigmoid(cur_topk_scores[j]).item()
            draw_bbox_values = [x.item() for x in draw_bbox_pred]
            draw_bbox_values_gt = [x.item() for x in draw_bbox_gt]
            

            # Then use these values ​​to format the string
            title_str = 'Prob: gt {:.3f}:[({:.0f},{:.0f}),({:.0f},{:.0f})]\n pred {:.3f}:[({:.0f},{:.0f}),({:.0f},{:.0f})]'.format(
                cur_prob_value,
                draw_bbox_values_gt[0],draw_bbox_values_gt[1],
                draw_bbox_values_gt[2],draw_bbox_values_gt[3],
                cur_pred_value,
                draw_bbox_values[0], draw_bbox_values[1],
                draw_bbox_values[2], draw_bbox_values[3],
            )

            # Set the title using the corrected string
            fig.suptitle(title_str, fontsize=10)

            ax[0].imshow(img)
            ax[1].imshow(cur_query)
            if cur_prob[ori_idx].item() > 0.5:
                draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[ori_idx], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_gt[1], draw_bbox_gt[0]), 
                                         draw_bbox_gt[3]-draw_bbox_gt[1], draw_bbox_gt[2]-draw_bbox_gt[0], 
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)
            if cur_prob[ori_idx].item() > 0.5:
                draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                         linewidth=1, edgecolor='g', facecolor='none')
                ax[0].add_patch(rect)
            if cur_topk_scores[j].item() > 0.6:
                draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                         linewidth=1, edgecolor='b', facecolor='none')
                ax[0].add_patch(rect)
            plt.savefig(os.path.join(output_dir, 'tmp.png'))
            plt.close()
            frames.append(cv2.imread(os.path.join(output_dir, 'tmp.png'))[...,::-1])
        save_name = os.path.join(output_dir, '{}_{}.gif'.format(iter_num, i))
        imageio.mimsave(save_name, frames, 'GIF', duration=1)



def vis_feature(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    featrue_vis=pred['featrue_vis']

    cpu_featrue_vis = {key: tensor.cpu() for key, tensor in featrue_vis.items()}

    query_vis=cpu_featrue_vis['query_feat_after_extract_feature_vis']
    


    B,_,H,W=query_vis.shape

    for i in range(B):
        frames = []
        fig, ax = plt.subplots(2,4, dpi=100)
        for index, (key, value) in enumerate(cpu_featrue_vis.items()):                                                                          
            
            cur_feat_np,cur_channel=random_channel_vis(value[i])
            
            idxi=index//4
            idxj=index%4
            #These values ​​are then used to format the string
            title_str = '%s:channel %s' %(key,cur_channel)

            #Set title with corrected string
            
            ax[idxi,idxj].set_title(title_str,fontsize=5)

            ax[idxi,idxj].imshow(cur_feat_np,cmap='gray')

            plt.tight_layout()
            
        plt.tight_layout()    
        plt.savefig(os.path.join(output_dir, '{}_{}.png'.format(iter_num, i)))
        plt.close()

def random_channel_vis(image_tensor):
    '''tensor: shape c h w
    reture  h w
    '''
    channel_num=image_tensor.shape[0]
    # vis_channel=random.randint(0, channel_num - 1)
    vis_channel=2
    cur_image=image_tensor[vis_channel]
    cur_iamge_np = cur_image.numpy()
    return cur_iamge_np,vis_channel