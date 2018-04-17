

# # to test, supposed to be equivalent to gray2rgb
# def mat2im(mat, cmap, limits):
#   '''
# % PURPOSE
# % Uses vectorized code to convert matrix "mat" to an m-by-n-by-3
# % image matrix which can be handled by the Mathworks image-processing
# % functions. The the image is created using a specified color-map
# % and, optionally, a specified maximum value. Note that it discards
# % negative values!
# %
# % INPUTS
# % mat     - an m-by-n matrix  
# % cmap    - an m-by-3 color-map matrix. e.g. hot(100). If the colormap has 
# %           few rows (e.g. less than 20 or so) then the image will appear 
# %           contour-like.
# % limits  - by default the image is normalised to it's max and min values
# %           so as to use the full dynamic range of the
# %           colormap. Alternatively, it may be normalised to between
# %           limits(1) and limits(2). Nan values in limits are ignored. So
# %           to clip the max alone you would do, for example, [nan, 2]
# %          
# %
# % OUTPUTS
# % im - an m-by-n-by-3 image matrix  
#   '''
#   assert len(mat.shape) == 2
#   if len(limits) == 2:
#     minVal = limits[0]
#     tempss = np.zeros(mat.shape) + minVal
#     mat    = np.maximum(tempss, mat)
#     maxVal = limits[1]
#     tempss = np.zeros(mat.shape) + maxVal
#     mat    = np.minimum(tempss, mat)
#   else:
#     minVal = mat.min()
#     maxVal = mat.max()
#   L = len(cmap)
#   if maxVal <= minVal:
#     mat = mat-minVal
#   else:
#     mat = (mat-minVal) / (maxVal-minVal) * (L-1)
#   mat = mat.astype(np.int32)
  
#   image = np.reshape(cmap[ np.reshape(mat, (mat.size)), : ], mat.shape + (3,))
#   return image

# def jet(m):
#   cm_subsection = linspace(0, 1, m)
#   colors = [ cm.jet(x) for x in cm_subsection ]
#   J = np.array(colors)
#   J = J[:, :3]
#   return J

# def generate_color_from_heatmap(maps, num_of_color=100, index=None):
#   assert isinstance(maps, np.ndarray)
#   if len(maps.shape) == 3:
#     return generate_color_from_heatmaps(maps, num_of_color, index)
#   elif len(maps.shape) == 2:
#     return mat2im( maps, jet(num_of_color), [maps.min(), maps.max()] )
#   else:
#     assert False, 'generate_color_from_heatmap wrong shape : {}'.format(maps.shape)
