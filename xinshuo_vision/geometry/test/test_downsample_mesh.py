# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com


from xinshuo_vision import reduce_faces
# from xinshuo_io import load_list_from_folder

# infile1 = '/home/xinshuo/31736.ply'
# infile2 = '/home/xinshuo/translated.obj'
# outfile = '/home/xinshuo/merged.ply'
outfile = '/media/xinshuo/disk2/datasets/extra/white_small_downsample_v2.obj'
in_dir = '/media/xinshuo/disk2/datasets/extra/white_small.obj'
# obj_list, _ = load_list_from_folder(in_dir, ext_filter=['.obj'], depth=1, recursive=False, sort=True, save_path=None, debug=True)
# print obj_list
num_faces = 50

reduce_faces(in_dir, outfile, num_faces)

# translated_mesh = '/home/xinshuo/tmp.ply'
# translation = [89.7634, -20.1220, 956.5000]

# translate_mesh(outfile, translated_mesh, translation)
