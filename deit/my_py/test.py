import torch
H = W = 4
sr_ratio = 2
N_reduce = 4

# flat_patch_pos = torch.arange(0, H * W)
# x_pos = flat_patch_pos // (sr_ratio * H)

# origin_x_pos = flat_patch_pos // H
# origin_y_pos = flat_patch_pos % H

# flat_patch_pos_for_y = torch.zeros(H * W)
# for i, pos in enumerate(zip(origin_x_pos, origin_y_pos)):
#     flat_patch_pos_for_y[i] = pos[0] + pos[1] * H
# y_pos = flat_patch_pos_for_y // (sr_ratio * H)

# token_visible_num = x_pos * sr_ratio + y_pos + 1 
# causal_mask = torch.full((H * W, N_reduce), float('-inf'))
# for i in range(H * W):
#     causal_mask[i, :int(token_visible_num[i])] = 0.


flat_patch_pos = torch.arange(0, H * W)
# flat_patch_pos = torch.arange(0, H * W, device=x.device)
x_pos = flat_patch_pos // (sr_ratio * H)

origin_x_pos = flat_patch_pos // H
origin_y_pos = flat_patch_pos % H

flat_patch_pos_for_y = origin_x_pos + origin_y_pos * H
y_pos = flat_patch_pos_for_y // (sr_ratio * H)

token_visible_num = x_pos * sr_ratio + y_pos + 1
token_visible_num = token_visible_num.long()  # Ensure it's long type for indexing

# Initialize the causal mask with -inf and use broadcasting to fill in the visible tokens
# causal_mask = torch.full((H * W, N_reduce), float('-inf'), device=x.device)
# row_indices = torch.arange(H * W, device=x.device).unsqueeze(1)
# col_indices = torch.arange(N_reduce, device=x.device).unsqueeze(0)
causal_mask = torch.full((H * W, N_reduce), float('-inf'))
row_indices = torch.arange(H * W).unsqueeze(1)
col_indices = torch.arange(N_reduce).unsqueeze(0)
mask = col_indices < token_visible_num.unsqueeze(1)
causal_mask[mask] = 0.



print('')