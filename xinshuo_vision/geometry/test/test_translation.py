# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com


from xinshuo_vision import translate_obj, merge_mesh

# infile1 = '/home/xinshuo/31736.ply'
infile = '/home/xinshuo/red.obj'
outfile = '/home/xinshuo/translated.obj'

translation = [-32.4844, -45.4029, 966.8720]
translate_obj(infile, outfile, translation)

# translated_mesh = '/home/xinshuo/tmp.ply'


# translate_mesh(outfile, translated_mesh, translation)
