# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
from __future__ import print_function
from pprint import pprint

import os, sys, time
import glob, glob2
import numpy as np
from scipy.misc import imsave
from PIL import Image

from xinshuo_python import *
from xinshuo_miscellaneous import string2ext_filter, remove_empty_item_from_list, str2num

import httplib2
# from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient import discovery

def fileparts(pathname):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed
	'''
	pathname = safepath(pathname)
	if len(pathname) == 0:
		return ('', '', '')
	if pathname[-1] == '/':
		if len(pathname) > 1:
			return (pathname[:-1], '', '')	# ignore the final '/'
		else:
			return (pathname, '', '')	# ignore the final '/'
	directory = os.path.dirname(os.path.abspath(pathname))
	filename = os.path.splitext(os.path.basename(pathname))[0]
	ext = os.path.splitext(pathname)[1]
	return (directory, filename, ext)

def mkdir_if_missing(pathname, debug=True):
    pathname = safepath(pathname)
    if debug:
        assert is_path_exists_or_creatable(pathname), 'input path is not valid or creatable: %s' % pathname
    dirname, _, _ = fileparts(pathname)

    if not is_path_exists(dirname):
        mkdir_if_missing(dirname)

    if isfolder(pathname) and not is_path_exists(pathname):
        os.mkdir(pathname)


######################################################### dict related #########################################################

def save_struct(struct_save, save_path, debug_mode):
    with open(save_path, 'w') as f:
        
        for k, v in struct_save.__dict__.items():
            # print(k: v)
            # print(v)
            f.write('%s    %s\n' % (k, v))
    return

######################################################### txt related #########################################################

def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safepath(file_path)
    if debug:
        assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path

    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    num_lines = len(data)
    file.close()

    return data, num_lines

def save_txt_file(data_list, save_path, debug=True):
    '''
    save a list of string to a file
    '''
    save_path = safepath(save_path)
    if debug:
        assert is_path_exists_or_creatable(save_path), 'text file is not able to be created at path: %s!' % save_path

    first_line = True
    with open(save_path, 'w') as file:
        for item in data_list:
            if first_line:
                file.write('%s' % item)
                first_line = False
            else:
                file.write('\n%s' % item)
    file.close()

    return    

######################################################### list related #########################################################

def load_list_from_file(file_path, debug=True):
    '''
    this function reads list from a txt file
    '''
    file_path = safepath(file_path)
    _, _, extension = fileparts(file_path)

    if debug:
        assert extension == '.txt', 'File doesn''t have valid extension.'
    file = open(file_path, 'r')
    if debug:
        assert file != -1, 'datalist not found'

    fulllist = file.read().splitlines()
    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)
    file.close()

    return fulllist, num_elem

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    '''
    load a list of files or folders from a system path

    parameter:
        folder_path: root to search 
        ext_filter: a string to represent the extension of files interested
        depth: maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive: 
            False: only return current level
            True: return all levels till to the input depth
    '''
    folder_path = safepath(folder_path)
    if debug:
        assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path):
        return [], 0

    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter):    # convert to a list
        ext_filter = [ext_filter]

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist

        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1):
            wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safepath(save_path)
        if debug:
            assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist:
                file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

def load_list_from_folders(folder_path_list, ext_filter=None, depth=1, recursive=False, save_path=None, debug=True):
    '''
    load a list of files or folders from a list of system path
    '''
    if debug:
        assert islist(folder_path_list) or isstring(folder_path_list), 'input path list is not correct'
    if isstring(folder_path_list):
        folder_path_list = [folder_path_list]

    fulllist = list()
    num_elem = 0
    for folder_path_tmp in folder_path_list:
        fulllist_tmp, num_elem_tmp = load_list_from_folder(folder_path_tmp, ext_filter=ext_filter, depth=depth, recursive=recursive)
        fulllist += fulllist_tmp
        num_elem += num_elem_tmp

    # save list to a path
    if save_path is not None:
        save_path = safepath(save_path)
        if debug:
            assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist:
                file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

# def generate_list_from_folder(save_path, src_path, ext_filter='jpg'):
# 	save_path = safepath(save_path)
# 	src_path = safepath(src_path)
# 	assert isfolder(src_path) and is_path_exists(src_path), 'source folder not found or incorrect'
# 	if not isfile(save_path):
# 		assert isfolder(save_path), 'save path is not correct'
# 		save_path = os.path.join(save_path, 'datalist.txt')

# 	if ext_filter is not None:
# 		assert isstring(ext_filter), 'extension filter is not correct'

# 	filepath = os.path.dirname(os.path.abspath(__file__))
# 	cmd = 'th %s/generate_list.lua %s %s %s' % (filepath, src_path, save_path, ext_filter)
# 	os.system(cmd)    # generate data list


def generate_list_from_data(save_path, src_data, debug=True):
    '''
    generate a file which contains a 1-d numpy array data

    parameter:
        src_data:   a list of 1 element data, or a 1-d numpy array data
    '''
    save_path = safepath(save_path)

    if debug:
        if isnparray(src_data):
            assert src_data.ndim == 1, 'source data is incorrect'
        elif islist(src_data):
            assert all(np.array(data_tmp).size == 1 for data_tmp in src_data), 'source data is in correct'
        assert isfolder(save_path) or isfile(save_path), 'save path is not correct'
        
    if isfolder(save_path):
        save_path = os.path.join(save_path, 'datalist.txt')

    if debug:
        assert is_path_exists_or_creatable(save_path), 'the file cannot be created'

    with open(save_path, 'w') as file:
        for item in src_data:
            file.write('%f\n' % item)
    file.close()

######################################################### matrix related #########################################################

def save_2dmatrix_to_file(data, save_path, formatting='%.1f', debug=True):
    save_path = safepath(save_path)
    if debug:
        assert isnparray(data) and len(data.shape) == 2, 'input data is not 2d numpy array'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
        mkdir_if_missing(save_path)
        # assert isnparray(data) and len(data.shape) <= 2, 'the data is not correct'
        
    np.savetxt(save_path, data, delimiter=' ', fmt=formatting)

def load_2dmatrix_from_file(src_path, delimiter=' ', dtype='float32', debug=True):
    src_path = safepath(src_path)
    if debug:
        assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path

    data = np.loadtxt(src_path, delimiter=delimiter, dtype=dtype)
    nrows = data.shape[0]
    return data, nrows

######################################################### pts related #########################################################

# standard facial annotation format IO function
# note that, the top left point is (1, 1) in 300-W instead of zero-indexed
def anno_writer(pts_array, pts_savepath, num_pts=68, anno_version=1, debug=True):
    '''
    write the point array to a .pts file
    parameter:
        pts_array:      2 or 3 x num_pts numpy array
        
    '''
    if debug:
        assert is_path_exists_or_creatable(pts_savepath), 'the save path is not correct'
        assert (is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array)) and pts_array.shape[1] == num_pts, 'the input point is not correct'

    with open(pts_savepath, 'w') as file:
        file.write('version: %d\n' % anno_version)
        file.write('n_points: %d\n' % num_pts)
        file.write('{\n')

        # main content
        for pts_index in xrange(num_pts):
            if is2dptsarray(pts_array):
                file.write('%.3f %.3f %f\n' % (pts_array[0, pts_index], pts_array[1, pts_index], 1.0))      # all visible
            else:                           
                file.write('%.3f %.3f %f\n' % (pts_array[0, pts_index], pts_array[1, pts_index], pts_array[2, pts_index]))

        file.write('}')
        file.close()

# done
def anno_parser(anno_path, num_pts=None, anno_version=None, debug=True):
    '''
    parse the annotation for LS3D-W dataset, which has a fixed format for .pts file
    return:
        pts: 3 x num_pts (x, y, oculusion)          
    '''
    data, num_lines = load_txt_file(anno_path, debug=debug)
    if debug:
        assert data[0].find('version: ') == 0, 'version is not correct'
        assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'
        assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'
    version = str2num(data[0][len('version: '):])
    n_points = int(data[1][len('n_points: '):])

    if debug:
        # print('version of annotation is %d' % version)
        # print('number of points is %d' % n_points)
        assert num_lines == n_points + 4, 'number of lines is not correct'      # 4 lines for general information: version, n_points, start and end symbol
        if anno_version is not None:
            assert version == anno_version, 'version of annotation is not correct: %d vs %d' % (version, anno_version)
        if num_pts is not None:
            assert num_pts == n_points, 'number of points is not correct: %d vs %d' % (num_pts, n_points)

    # read points coordinate
    pts = np.zeros((3, n_points), dtype='float32')
    line_offset = 3     # first point starts at fourth line
    for point_index in xrange(n_points):
        try:
            pts_list = data[point_index + line_offset].split(' ')           # x y format
            if len(pts_list) > 2 and pts_list[2] == '':     # handle edge case where additional whitespace exists after point coordinates
                pts_list = remove_empty_item_from_list(pts_list)
            pts[0, point_index] = float(pts_list[0])
            pts[1, point_index] = float(pts_list[1])
            if len(pts_list) == 3:
                pts[2, point_index] = float(pts_list[2])
            else:
                pts[2, point_index] = float(1)          # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by LS3D-W
        except ValueError:
            print('error in loading points in %s' % anno_path)
    return pts

######################################################### image related #########################################################
def load_image(src_path, resize_factor=1.0, rotate=0, mode='numpy', debug=True):
    '''
    load an image from given path

    parameters:
        resize_factor:      resize the image (>1 enlarge)
        mode:               numpy or pil, specify the format of returned image
        rotate:             counterclockwise rotation in degree
    '''

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    src_path = safepath(src_path)
    if isinteger(resize_factor):
        resize_factor = float(resize_factor)

    if debug:
        assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path
        assert mode == 'numpy' or mode == 'pil', 'the input mode for returned image is not correct'
        assert isfloat(resize_factor) and resize_factor > 0, 'the resize factor is not correct'

    with open(src_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

            if rotate != 0:
                img = img.rotate(rotate, expand=True)
            width, height = img.size
            img = img.resize(size=(int(width*resize_factor), int(height*resize_factor)), resample=Image.BILINEAR)

            if mode == 'numpy':
                return np.array(img)
            elif mode == 'pil':
                return img
            else:
                assert False, 'the mode %s is not supported' % mode

def save_image_from_data(save_path, data, debug=True, vis=False):
    save_path = safepath(save_path)
    if debug:
        assert isimage(data), 'input data is not image format'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
    
    mkdir_if_missing(save_path)
    imsave(save_path, data)

######################################################### web related #########################################################
"""
BEFORE RUNNING:
---------------
1. If not already done, enable the Google Sheets API
   and check the quota for your project at
   https://console.developers.google.com/apis/api/sheets
2. Install the Python client library for Google APIs by running
   `pip install --upgrade `
"""

def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    try:
        import argparse
        flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
    except ImportError:
        flags = None

    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def get_sheet_service():
    # If modifying these scopes, delete your previously saved credentials
    # at ~/.credentials/sheets.googleapis.com-python-quickstart.json
    SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
    # Authorize using one of the following scopes:
    #     'https://www.googleapis.com/auth/drive'
    #     'https://www.googleapis.com/auth/drive.file'
    #     'https://www.googleapis.com/auth/drive.readonly'
    #     'https://www.googleapis.com/auth/spreadsheets'
    #     'https://www.googleapis.com/auth/spreadsheets.readonly'    
    #     'https://www.googleapis.com/auth/drive'
    #     'https://www.googleapis.com/auth/drive.file'
    #     'https://www.googleapis.com/auth/spreadsheets'

    CLIENT_SECRET_FILE = 'client_secret.json'
    APPLICATION_NAME = 'Google Sheets API Python'

    # TODO: Change placeholder below to generate authentication credentials. See
    # https://developers.google.com/sheets/quickstart/python#step_3_set_up_the_sample

    # credentials = None
    credentials = get_credentials()
    service = discovery.build('sheets', 'v4', credentials=credentials)

    return service


def update_patchs2sheet(service, sheet_id, starting_position, data, debug=True):
    '''
    update a list of list data to a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        starting_position:      a string existing in the sheet to represent the let-top corner of patch to fill in
        data:                   a list of list data to fill
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        isstring(starting_position), 'the starting position is not correct'
        islistoflist(data), 'the input data is not a list of list'

    # How the input data should be interpreted.
    value_input_option = 'RAW'  # TODO: Update placeholder value.

    value_range_body = {'values': data}
    request = service.spreadsheets().values().update(spreadsheetId=sheet_id, range=starting_position, valueInputOption=value_input_option, body=value_range_body)
    response = request.execute()

def update_row2sheet(service, sheet_id, row_starting_position, data, debug=True):
    '''
    update a list of data to a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        starting_position:      a string existing in the sheet to represent the let-top corner of patch to fill in
        data:                   a of list data to fill
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        isstring(row_starting_position), 'the starting position is not correct'
        islist(data), 'the input data is not a list'

    # How the input data should be interpreted.
    value_input_option = 'RAW'  # TODO: Update placeholder value.

    value_range_body = {'values': [data]}
    request = service.spreadsheets().values().update(spreadsheetId=sheet_id, range=row_starting_position, valueInputOption=value_input_option, body=value_range_body)
    response = request.execute()

def get_data_from_sheet(service, sheet_id, search_range, debug=True):
    '''
    get a list of data from a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        search_range:      a list of position queried 
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        islist(search_range), 'the search range is not a list'

    # print(search_range)
    # How the input data should be interpreted.
    # value_input_option = 'RAW'  # TODO: Update placeholder value.

    # value_range_body = {'values': [data]}
    request = service.spreadsheets().values().batchGet(spreadsheetId=sheet_id, ranges=search_range)

    while True:
        try:
            response = request.execute()
            break
        except:
            continue

    data = list()
    # print(response['valueRanges'])
    for raw_data in response['valueRanges']:
        if 'values' in raw_data:
            data.append(raw_data['values'][0][0])
        else:
            data.append('')

    return data