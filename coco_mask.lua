--[[----------------------------------------------------------------------------
This script is for attaching color mask to the original image using coco maskapi.

Author: Xinshuo Weng
Email: xinshuow@andrew.cmu.edu
------------------------------------------------------------------------------]]
require 'image'
require 'paths'
local lfs = require 'lfs'
local matio = require 'matio'
local coco = require 'coco'
local maskApi = coco.MaskApi

-- source image folder path
dataset = 'real7'
local image_path_base = '/home/xinshuow/math_project/ours/' .. dataset .. '/images/'
local mask_path_base = '/home/xinshuow/math_project/ours/' .. dataset .. '/masks3/'
local save_path_base = '/home/xinshuow/math_project/ours/' .. dataset .. '/maskontop3_0.2_new/'
local imagelist = {}
local masklist = {}

--------------------------------------------------------------------------------
-- do it

print('| start')
-- read the image list from the path
for file in lfs.dir(image_path_base) do
	if string.find(file, 'input') ~= nil then
		index = string.find(file, '.png')
		id = string.sub(file, index - 5, index-1)	-- get the id of the image
		-- print(image_path .. file)
		imagelist[tonumber(id)] = file
	end
end

-- read the mask list from the path
for file in lfs.dir(mask_path_base) do
	if string.find(file, 'output') ~= nil then
		index = string.find(file, '.png')
		id = string.sub(file, index - 5, index-1)	-- get the id of the mask
		masklist[tonumber(id)] = file
	end
end
assert(#imagelist == #masklist, 'number of input is not equal to the output image')

-- attach the mask to the original image
if paths.dirp(save_path_base) == false then
	lfs.mkdir(save_path_base)
end

for i = 1, #imagelist do
	local img = image.load(image_path_base .. imagelist[i])
	local masks = image.load(mask_path_base .. masklist[i])

	-- save result
	-- test = masks[{{5}, {}, {}}]
	-- print(test:size())
	maskApi.drawMasks(img, masks, 1)
	-- matio.save('test.mat', masks)
	image.save(save_path_base .. imagelist[i], img)
	-- image.save(string.format('./test_maks.jpg', config.model),test)
end

print('| done')

