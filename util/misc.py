import numpy as np
import errno
import os
import cv2
from shapely.geometry import Polygon
from util.config import config as cfg


def to_device(*tensors):
    if len(tensors) < 2:
        return tensors[0].to(cfg.device)
    return (t.to(cfg.device) for t in tensors)


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise

def rescale_result(image, contours, H, W):
    ori_H, ori_W = image.shape[:2]
    image = cv2.resize(image, (W, H))
    for cont in contours:
        cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
        cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
    return image, contours


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | input_mask.astype(np.uint8))


def regularize_sin_cos(sin, cos):
    # regularization
    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))
    return sin * scale, cos * scale


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))

def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))

def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[1] / l

def vector_cos(v):
    assert len(v) == 2
    # cos = x / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[0] / l

def find_bottom(pts):
    #print("nw trying to find bottoms",pts)
    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        #print("e",e)
        candidate = []
        #trying to find the pair of edges which are forming an obtuse angle

        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.7:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))
        #print("candidate",candidate)
        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]

    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    #finally return the two edges which can be the head and snake of our text instance.
    #print(bottoms[0])

    return bottoms


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while (i % n_pts != b2_end):
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while (i % n_pts != b1_end):
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def find_long_edges(points, bottoms):
    #text polygon assumed in a snake shape
    #takes heead and tail edges as input (bottom)
    #takes all the points also as input

    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []
    # print("b1_start",b1_start)
    # print("b1_end",b1_end)
    # print("b2_start",b2_start)
    # print("b2_end",b2_end)
    # print("i",i)
    # print("n_pts",n_pts)
    while (i % n_pts != b2_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts
    # print("long_edge",long_edge_1)
    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while (i % n_pts != b1_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    # print("long_edge_2",long_edge_2)
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):
    #print("splitting the edge",long_edge, " into", n_parts," parts")
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    #print("Edge_length",edge_length)
    point_cumsum = np.cumsum([0] + edge_length)

    #print("point_cumsum",point_cumsum)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while(cur_end > point_cumsum[cur_node + 1]):
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        #print("before",e1,e2)
        e1, e2 = points[e1], points[e2]
        #print("afteer",e1,e2)

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    #print(len(splited_result))
    #reeturns a lsit of points along the axis of the long and short edge
    return np.stack(splited_result)

def disjoint_find(x, F):
    if F[x] == x:
        return x
    F[x] = disjoint_find(F[x], F)
    return F[x]

def disjoint_merge(x, y, F):
    x = disjoint_find(x, F)
    y = disjoint_find(y, F)
    if x == y:
        return False
    F[y] = x
    return True


def merge_polygons(polygons, merge_map):

    def merge_two_polygon(p1, p2):
        p2 = Polygon(p2)
        merged = p1.union(p2)
        return merged

    merge_map = [disjoint_find(x, merge_map) for x in range(len(merge_map))]
    merge_map = np.array(merge_map)
    final_polygons = []

    for i in np.unique(merge_map):
        merge_idx = np.where(merge_map == i)[0]
        if len(merge_idx) > 0:
            merged = Polygon(polygons[merge_idx[0]])
            for j in range(1, len(merge_idx)):
                merged = merge_two_polygon(merged, polygons[merge_idx[j]])
            x, y = merged.exterior.coords.xy
            final_polygons.append(np.stack([x, y], axis=1).astype(int))

    return final_polygons
