from collections import deque
import numpy as np
import cv2
import time
from numpy import random


data_deque = {}
palette = (2 * 11 - 1, 2 * 15 - 1, 2 ** 20 - 1)

def point_in_rect(point, rect):
    xx1, yy1, w, h = rect
    xx2, yy2 = xx1 + w, yy1 + h
    x, y = point
    if xx1 < x < xx2:
        if yy1 < y < yy2:
            return True
    return False

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 1:  # Car
        color = (222, 82, 175)
    elif label == 2:  # Motobike
        color = (0, 204, 255)
    elif label == 3:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def calculate_distance(point1, point2):
    distance_pixels=np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return distance_pixels* 0.05

# Function to calculate speed
def calculate_speed(distance, time_elapsed):
    if time_elapsed > 0:
        return (distance/1000) / (time_elapsed/3600)
    else:
        return 0
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0), frame_start_time=None):
    if frame_start_time is None:
        frame_start_time = time.time()
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    total_width = 0
    total_height = 0
    total_boxes = 0

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # calculate width and height of the bounding box
        w = x2 - x1
        h = y2 - y1

        # add width and height to label
        label += f" Width: {w:.2f}, Height: {h:.2f}"

        # add width and height to total
        total_width += w
        total_height += h
        total_boxes += 1

        # add center to buffer
        data_deque[id].appendleft(center)
        speed = 0
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], (147, 20, 255), 2)
        if len(data_deque[id]) >= 2:  # 2
            distance = calculate_distance(data_deque[id][0], data_deque[id][1])
            time_elapsed = time.time() - frame_start_time
            speed = calculate_speed(distance, time_elapsed)
            label += f" Speed: {speed:.2f} m/s"

            UI_box(box, img, label=label, color=color, line_thickness=1)

    # calculate average width and height
    avg_width = total_width / total_boxes
    avg_height = total_height / total_boxes
    return img, avg_width, avg_height
