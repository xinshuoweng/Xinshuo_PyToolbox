# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com


from xinshuo_vision import merge_mesh, translate_mesh

infile1 = '/home/xinshuo/31736.ply'
infile2 = '/home/xinshuo/translated.obj'
outfile = '/home/xinshuo/merged.ply'

merge_mesh(infile1, infile2, outfile)

# translated_mesh = '/home/xinshuo/tmp.ply'
# translation = [89.7634, -20.1220, 956.5000]

# translate_mesh(outfile, translated_mesh, translation)
