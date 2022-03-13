import torch

def convert_state_dict(state_dict, inplace=True):
    if not inplace:
        state_dict = state_dict.copy()
    
    # Fix path embedding
    state_dict['to_patch.weight'] = state_dict.pop('to_patch_embedding.1.weight')#.reshape(64, 10, 14).permute(0, 2, 1)
    state_dict['to_patch.bias'] = state_dict.pop('to_patch_embedding.1.bias')

    # Fix to_qkv
    keys = [k for k in state_dict if 'to_qkv' in k]
    for key in keys:
        to_qkv_weight = state_dict.pop(key)
        inner_dim = to_qkv_weight.shape[0] // 3
        state_dict[key.replace('to_qkv', 'to_q')] = to_qkv_weight[0:inner_dim, :]
        state_dict[key.replace('to_qkv', 'to_k')] = to_qkv_weight[inner_dim:2*inner_dim, :]
        state_dict[key.replace('to_qkv', 'to_v')] = to_qkv_weight[2*inner_dim:, :]

    return state_dict

state_dict = torch.load(f"vit_h8patch20.pth", map_location=torch.device('cpu'))
state_dict = convert_state_dict(state_dict)
torch.save(state_dict, f"vit_h8patch20_linear_emb.pth")