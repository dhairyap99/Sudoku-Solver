import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator
import tensorflow as tf
import sudoku
import math
import copy


def distance_between(p1, p2):
    """distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    """all the four corners of rectangle"""
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    """get all four sides"""
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    """destination points"""
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    """getPerspective is used to change the perspective of the image from src to destination point"""
    m = cv2.getPerspectiveTransform(src, dst)
    """warp perspective is used to take the image from A to image B with given size"""
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def find_corners_of_largest_polygon(img):
    """
    to find the largest continous shape in the binary image
    findCountours is used to find all the contours in the image
    chain_approx_simple removes redundant points and saves memory.
    for example for a line only end points wil be stores
    retr_external return outer most boundarys incase of overlapping shapes
    """
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """sorted the contours in desc order to get teh largest contour"""
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]  # Largest image
    """all the corners of largest contours are found and returned"""
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def show_image(img):
    """to show image"""
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def pre_process_image(img, skip_dilate=False):
    """preprocess the image"""
    """GaussianBlur is used to reduce the noise in the image and blur the image. it is used to apply low pass filter"""
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    """ 
    adaptiveThreshold gives different threshold values to different regions depending on the neighbours of point.
    255= maximum value which can be assigned to a image
    ADAPTIVE_THRESH_GAUSSIAN_C : threshold value of a point = weighted sum of neighbours - constant
    THRESH_BINARY: 0 if threshold<something or 1
    11 = blocksize of neighbours
    2 = constant to be subtracted
    """
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    """
    inverted the image
    """
    proc = cv2.bitwise_not(proc, proc)
    """below code is used to fined the boundary of the sudoku puzzle"""
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        print("kernel")
        print(kernel)
        proc = cv2.dilate(proc, kernel)
    return proc


def infer_grid(img):
    """this image is used to get 81 grids of sudoku"""
    squares = []
    side = img.shape[:1]
    """size of each grid will be sudoku_size/9"""
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            """coordinates of all teh 81 points are stored"""
            squares.append((p1, p2))
    return squares


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """this method is used for digits inside the grid"""
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    """below code is used to create border around every grid """
    if h > w:
        t_pad = int(margin / 2)  # top border width
        b_pad = t_pad  # bottom border width
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)  # left and right border width
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    print("..........w:", w, " h:", h, " size:", size, ".................")
    img = cv2.resize(img, (w, h))
    """copyMakeBorder is for creating border around the image and grids of sudoku"""
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    img = inp_img.copy()
    height, width = img.shape[:2]
    max_area = 0
    seed_point = (None, None)
    if scan_tl is None:
        scan_tl = [0, 0]
    if scan_br is None:
        scan_br = [width, height]
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)
    mask = np.zeros((height + 2, width + 2), np.uint8)
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)
    top, bottom, left, right = height, 0, width, 0
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right
    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """this method will extract digits from the image"""
    """digit gets sub block from the image """
    digit = cut_from_rect(img, rect)
    """height and weight of the grid"""
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    """to get largest feature in the grid(digit). uses floodfill to fill the cells.
    It will take the digit from the grid and copy in another image. 
    thus removes extra white pixels which are not part of digit"""
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    """if number of white pixels are greater than 100 than only digit is present 
    or else empty grid with all black points"""
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    digits = []
    """image will be preprocessed"""
    img = pre_process_image(img.copy(), skip_dilate=True)
    """squares are all the small grids with preprocessed, scaled and centered digits"""
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def equal_matrices(max_a, max_b):
    for i in range(81):
        if max_a[i] != max_b[i]:
            return False
    return True


def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True


def detect_sudoku(sudoku_image):
    print("---------------prev----------")
    # original = cv2.imread('frames/frame.jpg', cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(sudoku_image, cv2.IMREAD_GRAYSCALE)  # read image
    # original = sudoku_image
    print("original")
    print(original)
    processed = pre_process_image(original)  # preprocess the image, gaussian blur, adaptive threshold
    print("processed")
    print(processed)
    corners = find_corners_of_largest_polygon(processed)  # find sudoku box, returns corners
    print("corners")
    print(corners)
    cropped = crop_and_warp(original, corners)  # crop the sudoku box, change and warp perspective
    print("cropped")
    print(cropped)
    squares = infer_grid(cropped)  # small grids from the sudoku
    print("squares")
    print(squares)
    digits = get_digits(cropped, squares, 32)  # get cleaned digits box
    print("digit")
    print(type(digits[0]))
    # show_digits(digits)

    digits = np.stack(digits)  # change axis of digits

    model = tf.keras.models.load_model("models/model1.hdf5")  # loads teh model

    pred = np.array([])  # predicted values will be stored here

    for i in range(len(digits)):
        """checks if any of the pixels are present"""
        if np.any(digits[i]):
            element = digits[i]  # take digits one by one
            element = element.reshape(-1, 32, 32, 1)  # reshape according to how model was trained
            """model.predict gives probabilities of all the possibilities"""
            pred = np.append(pred, np.argmax(model.predict(element)) + 1)
        else:
            pred = np.append(pred, 0)

    pred = pred.reshape(9, 9)
    print("pred")
    print(pred)
    print(type(digits), type(cropped))
    return pred, digits, cropped


def get_sudoku(sudoku_image):
    """this method gets the answer for the sudoku"""
    method_return = detect_sudoku(sudoku_image)
    """digits : grids of numbers present in the image, matrix is predicted numbers"""
    matrix, digits, cropped = method_return  # cropped sudoku with changed and warped perspective
    print("matrix")
    matrix = matrix.tolist()
    answer = []
    count = 0
    """sudoku.py called here to solve the sudoku"""
    for solution in sudoku.solve_sudoku((3, 3), matrix):
        if count > 10:
            break
        count = count + 1
        answer.append([*solution])

    answer = np.array(answer)
    answer = answer.reshape(9, 9)
    return answer.flatten(), digits, cropped


def print_solution(digits, images, ans):
    """returns answered images"""
    width = images.shape[1] // 9  # width of every grid
    height = images.shape[0] // 9  # height of every grid
    for i in range(9):
        for j in range(9):
            if np.any(digits[i * 9 + j]):  # checks if the grid already has a number or not
                continue
            text = str(int(ans[i * 9 + j]))  # text: it will be written in the empty grids
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=1)

            font_scale = 0.5 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width * j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height * (i + 1) - math.floor((height - text_height) / 2) + off_set_y
            images = cv2.putText(images, text, (bottom_left_corner_x, bottom_left_corner_y),
                                 font, font_scale, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    # show_image(images)
    return images


def run_detection(sudoku_image):
    """this method is called from main.py"""
    method_return = get_sudoku(sudoku_image)
    """ans: answer matrix of sudoku, digits: grids of numbers of sudoku"""
    ans, digits, cropped = method_return  # cropped sudoku with changed and warped perspective
    print(ans)
    answer_matrix = print_solution(digits, cropped, ans)
    """answer_matrix: answered image of sudoku"""
    return answer_matrix