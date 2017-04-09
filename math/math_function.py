# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math
import numpy as np

def get_line(pts, slope):
    '''
    # slope is the angle in degree, this function takes a point and a
    '''
    if slope == 90 or -90:
        slope = slope + 0.00001
    slope = math.tan(math.radians(slope))
    # print('slope is ' + str(slope))
    dividor = slope * pts[0] - pts[1]
    if dividor == 0:
        dividor += 0.00001
    b = 1.0 / dividor
    a = -b * slope
    assert math.fabs(pts[0]*a + pts[1]*b + 1) < 0.0000001, 'Point is not on the line'
    return np.array([a, b, 1], dtype=float)

def get_slope(pts1, pts2):
    slope = (pts1[1] - pts2[1]) / (pts1[0] - pts2[0])
    # print(slope)
    slope = np.arctan(slope)
    # print(slope)
    slope = math.degrees(slope)
    # print(slope)
    return slope


def get_intersection(line1, line2):
    a1 = line1[0]
    b1 = line1[1]
    a2 = line2[0]
    b2 = line2[1]
    dividor = a2 * b1 - a1 * b2
    if dividor == 0:
        dividor += 0.00001
    y = (a1 - a2) / dividor
    if a1 == 0:
        a1 += 0.00001
    x = (-1.0 - b1 * y) / a1
    assert math.fabs(x*line1[0] + y*line1[1] + 1) < 0.0000001, 'Intersection point is not on the line'
    assert math.fabs(x*line2[0] + y*line2[1] + 1) < 0.0000001, 'Intersection point is not on the line'
    return np.array([x, y], dtype=float)

# convert the point coordinate in the image to the general coordinate, y axis is inverse
def convert_pts(pts):
    return np.array([pts[0], -pts[1]], dtype=float)

# convert the point from general coordinate to image coordinate
def convert_pts_back2image(pts):
    return np.array([pts[0], -pts[1]], dtype=float)
