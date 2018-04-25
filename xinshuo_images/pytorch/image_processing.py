# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com


# centroid
# def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
#   assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(num_pts, -1), 1)
#   index_w = (index % W).float()
#   index_h = (index / W).float()
  
#   def normalize(x, L):
#     return -1. + 2. * x.data / (L-1)
#   boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
#   boxes[0] = normalize(boxes[0], W)
#   boxes[1] = normalize(boxes[1], H)
#   boxes[2] = normalize(boxes[2], W)
#   boxes[3] = normalize(boxes[3], H)
#   affine_parameter = torch.zeros((num_pts, 2, 3))
#   affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
#   affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
#   affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
#   affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
  
#   # extract the sub-region heatmap
#   theta = MU.np2variable(affine_parameter,heatmap.is_cuda,False)
#   grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
#   grid = F.affine_grid(theta, grid_size)
#   sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
#   sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

#   # slow for speed improvement
#   X = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, 1, radius*2+1)
#   Y = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, radius*2+1, 1)
  
#   sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
#   x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
#   y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.stack([x, y],1), score

# def find_tensor_peak(heatmap, radius, downsample):
#   assert heatmap.dim() == 2, 'The dimension of the heatmap is wrong : {}'.format(heatmap.dim())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   H, W = heatmap.size(0), heatmap.size(1)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(-1), 0)
#   index = int(MU.variable2np(index))
#   index_h, index_w = np.unravel_index(index, (H,W))

#   sw, sh = int(index_w - radius),     int(index_h - radius)
#   ew, eh = int(index_w + radius + 1), int(index_h + radius + 1)
#   sw, sh = max(0, sw), max(0, sh)
#   ew, eh = min(W, ew), min(H, eh)
  
#   subregion = heatmap[sh:eh, sw:ew]
#   threshold = 0.000001
#   eps = np.finfo(float).eps

#   with torch.cuda.device_of(subregion):
#     X = MU.np2variable(torch.arange(sw, ew).unsqueeze(0))
#     Y = MU.np2variable(torch.arange(sh, eh).unsqueeze(1))

#   indicator = (subregion > threshold).type( type(subregion.data) )
#   eps = (subregion <= threshold).type( type(subregion.data) ) * eps
#   subregion = subregion * indicator + eps
  
#   x = torch.sum( subregion * X ) / torch.sum( subregion )
#   y = torch.sum( subregion * Y ) / torch.sum( subregion )
     
#   ## calculate the score
#   np_x, np_y = MU.variable2np(x), MU.variable2np(y)
#   x2, y2 = min(W-1, int(np.ceil(np_x))), min(H-1, int(np.ceil(np_y)))
#   x1, y1 = max(0, x2-1), max(0, y2-1)
#   ## Bilinear interpolation
#   if x1 == x2: 
#     R1, R2 = heatmap[y1, x1], heatmap[y1, x2]
#   else:
#     R1 = (x2-x)/(x2-x1)*heatmap[y1, x1] + (x-x1)/(x2-x1)*heatmap[y1, x2]
#     R2 = (x2-x)/(x2-x1)*heatmap[y2, x1] + (x-x1)/(x2-x1)*heatmap[y2, x2]
#   if y1 == y2:
#     score = R1
#   else:
#     score = (y2-y)/(y2-y1)*R1 + (y-y1)/(y2-y1)*R2
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.cat([x, y]), score
