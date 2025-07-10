from einops import rearrange
import torch
import torch.nn.functional as F

def update_attention_mask(model_s_mlp, model_t_mlp, model_ks_mlp, mha_atten_weight, mha_atten_weight_ST,w_s,w_st):
    """
    xtemp_atten bt,hw,hw
    xtemp_atten_ST b,th2w2,th2w2
    output: bt hw d, bt h w, bt h w
    """
    b=4
    t=30
    h=32
    w=32
    h2=8
    w2=8

    #interpolate
    attention_weights_center = mha_atten_weight[:,:,512+16] # bt hw 

    # each h*w (frame_feature) sum to 1 in t atten
    attention_weights_center=F.softmax(attention_weights_center,dim=-1)

    attention_weights_center=rearrange(attention_weights_center,'(b t) hw->b (t hw)',t=t) # b thw
    # attention_weights_center=F.softmax(attention_weights_center,dim=1)
    attention_weights_center=rearrange(attention_weights_center,'b (t h w)->(b t) h w',t=t, h=h) #bt h w

    # Calculate the number of 8x8 tensors that can be extracted
    new_weight_ST=extract_diagonal_tensors_with_batch(mha_atten_weight_ST,64) #b t h2w2 h2w2
    new_weight_ST_center=new_weight_ST[:,:,:,32+4] #b t h2w2 
    new_weight_ST_center=rearrange(new_weight_ST_center,'b t (h w)->b t h w',h=8)

    

    
    new_weight_ST_center = F.interpolate(new_weight_ST_center, size=(h, w), mode='bilinear', align_corners=False) # b t h w

    # each h*w (frame feature) sum to 1 in st atten
    new_weight_ST_center=rearrange(new_weight_ST_center,'b t h w ->b t (h w)')
    new_weight_ST_center=F.softmax(new_weight_ST_center,dim=-1)
    new_weight_ST_center=rearrange(new_weight_ST_center,'b t (h w) -> (b t ) h w',t=t, h=h) #bt h w

    #use attention
    attention_weights_center=model_s_mlp(attention_weights_center)
    new_weight_ST_center=model_t_mlp(new_weight_ST_center)

    result = (w_s*attention_weights_center+new_weight_ST_center*w_st)*h*w

    #use attention
    result=model_ks_mlp(result)

    result=result.unsqueeze(-1).repeat(1,1,1,256)

    result =rearrange(result,'bt h w d ->bt d h w')
    

    return result,attention_weights_center,new_weight_ST_center

def extract_diagonal_tensors_with_batch(tensor, block_size):
    """
    Extract the sub-tensor diagonally from each batch of the given 3D tensor and return a new tensor, keeping the batch_size dimension.
    
    parameter:
    tensor (torch.Tensor): Enter a tensor with shape (batch_size, N, N).
    block_size (int): The size of the sub-tenster (block_size, block_size).
    
    return:
    torch.Tensor: Extracted sub-tensor, shape is (batch_size, t, block_size, block_size), where t is the number of extracted tensors.
    """
    batch_size = tensor.size(0)
    N = tensor.size(1)
    t = N //block_size # Assume that each two-dimensional tensor is a square matrix

    # Initialize a new tensor to store the extracted sub-tensors
    extracted_tensors = torch.zeros(batch_size, t, block_size, block_size, device=tensor.device)

    # Loop every batch
    for b in range(batch_size):
        for i in range(t):
            extracted_tensors[b, i] = tensor[b, i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size]

    return extracted_tensors

