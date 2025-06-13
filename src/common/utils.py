"""A collection of functions and classes used across multiple modules."""

import math
import queue
import cv2
import threading
import numpy as np
from src.common import config, settings
from random import random


def run_if_enabled(function):
    """
    Decorator for functions that should only run if the bot is enabled.
    :param function:    The function to decorate.
    :return:            The decorated function.
    """

    def helper(*args, **kwargs):
        if config.enabled:
            return function(*args, **kwargs)
    return helper


def run_if_disabled(message=''):
    """
    Decorator for functions that should only run while the bot is disabled. If MESSAGE
    is not empty, it will also print that message if its function attempts to run when
    it is not supposed to.
    """

    def decorator(function):
        def helper(*args, **kwargs):
            if not config.enabled:
                return function(*args, **kwargs)
            elif message:
                print(message)
        return helper
    return decorator


def distance(a, b):
    """
    Applies the distance formula to two points.
    :param a:   The first point.
    :param b:   The second point.
    :return:    The distance between the two points.
    """

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def separate_args(arguments):
    """
    Separates a given array ARGUMENTS into an array of normal arguments and a
    dictionary of keyword arguments.
    :param arguments:    The array of arguments to separate.
    :return:             An array of normal arguments and a dictionary of keyword arguments.
    """

    args = []
    kwargs = {}
    for a in arguments:
        a = a.strip()
        index = a.find('=')
        if index > -1:
            key = a[:index].strip()
            value = a[index+1:].strip()
            kwargs[key] = value
        else:
            args.append(a)
    return args, kwargs

def single_match(frame, template):
    assert template is not None, "Template is None — check if image file path is correct."
    assert frame is not None, "Frame is None — screenshot may have failed."

    print(f"[DEBUG] template shape: {template.shape}, dtype: {template.dtype}")
    print(f"[DEBUG] frame shape: {frame.shape}, dtype: {frame.dtype}")

    # 處理 template 灰階轉換
    if template.ndim == 3:
        if template.shape[2] == 4:
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
        elif template.shape[2] == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        elif template.shape[2] == 1:
            template = np.squeeze(template, axis=2)  # (H, W, 1) → (H, W)
        else:
            raise ValueError(f"Unsupported number of template channels: {template.shape[2]}")

    # 處理 frame 灰階轉換
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[2] == 1:
            frame = np.squeeze(frame, axis=2)
        else:
            raise ValueError(f"Unsupported number of frame channels: {frame.shape[2]}")

    assert template.ndim == 2 and frame.ndim == 2, "Both images must be 2D grayscale"
    assert template.dtype == np.uint8 and frame.dtype == np.uint8, "Both must be 8-bit grayscale"

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF)
    _, _, _, top_left = cv2.minMaxLoc(result)
    w, h = template.shape[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right


def multi_match(frame, template, threshold=0.95):
    """
    Finds all matches in FRAME that are similar to TEMPLATE by at least THRESHOLD.
    :param frame:       The image in which to search.
    :param template:    The template to match with.
    :param threshold:   The minimum percentage of TEMPLATE that each result must match.
    :return:            An array of matches that exceed THRESHOLD.
    """

    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    results = []
    for p in locations:
        x = int(round(p[0] + template.shape[1] / 2))
        y = int(round(p[1] + template.shape[0] / 2))
        results.append((x, y))
    return results


def convert_to_relative(point, frame):
    """
    Converts POINT into relative coordinates in the range [0, 1] based on FRAME.
    Normalizes the units of the vertical axis to equal those of the horizontal
    axis by using config.mm_ratio.
    :param point:   The point in absolute coordinates.
    :param frame:   The image to use as a reference.
    :return:        The given point in relative coordinates.
    """

    x = point[0] / frame.shape[1]
    y = point[1] / config.capture.minimap_ratio / frame.shape[0]
    return x, y


def convert_to_absolute(point, frame):
    """
    Converts POINT into absolute coordinates (in pixels) based on FRAME.
    Normalizes the units of the vertical axis to equal those of the horizontal
    axis by using config.mm_ratio.
    :param point:   The point in relative coordinates.
    :param frame:   The image to use as a reference.
    :return:        The given point in absolute coordinates.
    """

    x = int(round(point[0] * frame.shape[1]))
    y = int(round(point[1] * config.capture.minimap_ratio * frame.shape[0]))
    return x, y


def filter_color(img, ranges):
    """
    Returns a filtered copy of IMG that only contains pixels within the given RANGES.
    on the HSV scale.
    :param img:     The image to filter.
    :param ranges:  A list of tuples, each of which is a pair upper and lower HSV bounds.
    :return:        A filtered copy of IMG.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for i in range(1, len(ranges)):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, ranges[i][0], ranges[i][1]))

    # Mask the image
    color_mask = mask > 0
    result = np.zeros_like(img, np.uint8)
    result[color_mask] = img[color_mask]
    return result


def draw_location(minimap, pos, color):
    """
    Draws a visual representation of POINT onto MINIMAP. The radius of the circle represents
    the allowed error when moving towards POINT.
    :param minimap:     The image on which to draw.
    :param pos:         The location (as a tuple) to depict.
    :param color:       The color of the circle.
    :return:            None
    """

    center = convert_to_absolute(pos, minimap)
    cv2.circle(minimap,
               center,
               round(minimap.shape[1] * settings.move_tolerance),
               color,
               1)


def print_separator():
    """Prints a 3 blank lines for visual clarity."""

    print('\n\n')


def print_state():
    """Prints whether Auto Maple is currently enabled or disabled."""

    print_separator()
    print('#' * 18)
    print(f"#    {'ENABLED ' if config.enabled else 'DISABLED'}    #")
    print('#' * 18)


def closest_point(points, target):
    """
    Returns the point in POINTS that is closest to TARGET.
    :param points:      A list of points to check.
    :param target:      The point to check against.
    :return:            The point closest to TARGET, otherwise None if POINTS is empty.
    """

    if points:
        points.sort(key=lambda p: distance(p, target))
        return points[0]


def bernoulli(p):
    """
    Returns the value of a Bernoulli random variable with probability P.
    :param p:   The random variable's probability of being True.
    :return:    True or False.
    """

    return random() < p


def rand_float(start, end):
    """Returns a random float value in the interval [START, END)."""

    assert start < end, 'START must be less than END'
    return (end - start) * random() + start


def find_color_template(frame, template, threshold=0.95):
    """
    Finds all matches in FRAME that are similar to TEMPLATE by at least THRESHOLD.
    :param frame:       The image in which to search.
    :param template:    The template to match with.
    :param threshold:   The minimum percentage of TEMPLATE that each result must match.
    :return:            An array of matches that exceed THRESHOLD.
    """

    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return []
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # **將 FRAME 轉換為 HSV**
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # **將 TEMPLATE 也轉換為 HSV**
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # 定義黃色的 HSV 範圍
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 建立遮罩，只保留黃色範圍
    mask_frame = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    mask_template = cv2.inRange(hsv_template, lower_yellow, upper_yellow)

    # **套用遮罩，只保留黃色部分**
    yellow_frame = cv2.bitwise_and(frame, frame, mask=mask_frame)
    yellow_template = cv2.bitwise_and(template, template, mask=mask_template)

    # **執行匹配**
    result = cv2.matchTemplate(yellow_frame[:, :, 1], yellow_template[:, :, 1], cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    # **計算匹配點的中心**
    results = []
    for p in locations:
        x = int(round(p[0] + yellow_template.shape[1] / 2))
        y = int(round(p[1] + yellow_template.shape[0] / 2))
        results.append((x, y))

    return results


def find_white_text_template(frame, template, threshold=0.4):
    """
    使用白色對比增強的方式，在 frame 中找出與 template 相似的文字區塊座標。
    :param frame:       待搜尋圖片 (np.array, e.g., screenshot).
    :param template:    名字牌模板圖片 (cv2.imread).
    :param threshold:   匹配分數的閥值 (0~1).
    :return:            中心點座標 list of (x, y).
    """

    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return []

    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # White mask in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask_frame = cv2.inRange(hsv_frame, lower_white, upper_white)
    mask_template = cv2.inRange(hsv_template, lower_white, upper_white)

    # Morphology to reduce noise
    kernel = np.ones((2, 2), np.uint8)
    mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_OPEN, kernel)
    mask_template = cv2.morphologyEx(mask_template, cv2.MORPH_OPEN, kernel)

    # Edge detection on masked regions
    edges_frame = cv2.Canny(mask_frame, 50, 150)
    edges_template = cv2.Canny(mask_template, 50, 150)

    # Match using edges
    result = cv2.matchTemplate(edges_frame, edges_template, cv2.TM_CCOEFF_NORMED)

    # Detect good matches
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))  # (x, y)

    # Convert to center points
    results = []
    for p in locations:
        x = int(round(p[0] + template.shape[1] / 2))
        y = int(round(p[1] + template.shape[0] / 2))
        results.append((x, y))

    return results

##########################
#       Threading        #
##########################
class Async(threading.Thread):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.queue = queue.Queue()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.function(*self.args, **self.kwargs)
        self.queue.put('x')

    def process_queue(self, root):
        def f():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                root.after(100, self.process_queue(root))
        return f


def async_callback(context, function, *args, **kwargs):
    "Returns a callback function that can be run asynchronously by the GUI."

    def f():
        task = Async(function, *args, **kwargs)
        task.start()
        context.after(100, task.process_queue(context))
    return f
