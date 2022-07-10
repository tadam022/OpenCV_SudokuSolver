import cv2
import numpy as np
#import matplotlib.pyplot as plt

import copy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from tensorflow import keras

from CNN import IMG_SIZE

PROBABILITY_THRESHOLD = 0.7


def read_in_image(image_file):
    if not image_file:
        print("No file path given")
        return None

    try:
        return cv2.imread(image_file)
    except:
        print("File not found. Please enter a valid file path.")
        return None


def process_image(original_image, dilate_amount, show_processing=False):
    grey_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(grey_img, (5, 5), 0, sigmaY=0)  # reduce noise
    # edges = cv2.Canny(img_blur, 80, 160, apertureSize=3)
    # threshold1 = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 9, 2)
    threshold2 = cv2.adaptiveThreshold(img_blur, 255, 1, 1, blockSize=3, C=2)

    if show_processing:
        cv2.imshow('Adaptive Thresh', threshold2)

    dilated = cv2.dilate(threshold2, np.ones((dilate_amount, dilate_amount), np.uint8))

    if show_processing:
        cv2.imshow('Dilated image', dilated)

    # eroded = cv2.erode(dilated,np.ones((3, 3), np.uint8))
    return dilated


def get_grid_image(processed_image, original_image, show_processing=False):
    contours = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if show_processing:
        contours_image = original_image.copy()
        cv2.drawContours(contours_image, contours, -1, 255, 3)
        cv2.imshow('Contours', contours_image)

    polygon = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(polygon, True)
    approximatedShape = cv2.approxPolyDP(polygon, 0.01 * perimeter, True)
    hull = cv2.convexHull(approximatedShape)

    # cv2.drawContours(original_image, [hull], 0, (255, 0, 0), 3)

    corners = hull
    moment = cv2.moments(corners)

    if moment['m00'] != 0:
        center_x = int(moment['m10'] / moment['m00'])
        # center_y = int(moment['m01'] / moment['m00'])
    else:
        print("Could Not get Grid Image. Moment has division by zero")
        return

    left_points = [point[0] for point in corners if point[0][0] < center_x]
    right_points = [point[0] for point in corners if point[0][0] > center_x]
    # print("# Corners: ", len(corners))
    # print(left_points)
    # print(right_points)

    top_left = min(left_points, key=lambda x: x[1])
    bottom_left = max(left_points, key=lambda x: x[1])
    top_right = min(right_points, key=lambda x: x[1])
    bottom_right = max(right_points, key=lambda x: x[1])

    # print(top_left, top_right, bottom_left, bottom_right)

    corners = [top_left, top_right, bottom_left, bottom_right]

    if show_processing:
        corner_labels = ["TL", "TR", "BL", "BR"]
        labelled_corners = list(zip(corners, corner_labels))

        highlight_grid_image = original_image.copy()

        for corner, label in labelled_corners:
            cv2.circle(highlight_grid_image, corner, 7, (0, 255, 0), cv2.FILLED)
            cv2.putText(highlight_grid_image, f"{label}:{corner}", corner - 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Corners of grid', highlight_grid_image)

    # width should be same as height for a sudoku since its square
    w = top_right[0] - top_left[0]
    # top left, top right, bottom left, bottom right
    corners = np.array(corners, dtype='float32')

    warp_to_points = np.array([[0, 0], [w, 0], [0, w], [w, w]], dtype='float32')
    warped_matrix = cv2.getPerspectiveTransform(corners, warp_to_points)
    warped_image = cv2.warpPerspective(original_image, warped_matrix, (w, w))

    return warped_image, corners


def get_largest_connected_component(cell, cropped_width, cropped_height, tolerance):
    # largest connected component should correspond to the number in the cell (and not the gridlines if any)
    # starts from center, the number in cell should be in the center of the cell

    max_area = float('-inf')

    no_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell, 4)
    number_img = None
    for label in range(1, no_of_labels):
        center = centroids[label]
        height, width = stats[label, cv2.CC_STAT_HEIGHT], stats[label, cv2.CC_STAT_WIDTH]
        x, y = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            if cropped_width // 2 - tolerance < center[0] < cropped_width // 2 + tolerance:

                if cropped_height // 2 - tolerance < center[1] < cropped_height // 2 + tolerance:
                    max_area = area

                    number_img = cell[y:y + height, x:x + width]
    return number_img


def get_grid_numbers(grid_image, CNN_model, Otsu=True, lower=110, tolerance=5, show_certainties=False,
                     show_processing=False):
    warped_image_grey = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    '''
    g = warped_image_grey.copy()
    threshold2 = cv2.adaptiveThreshold(g, 255, 1, 1, blockSize=11, C=2)
    threshold2 = np.invert(threshold2)
    '''
    if not Otsu:
        threshold_val, warped_image_binary = cv2.threshold(warped_image_grey, lower, 255,
                                                           cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    else:
        threshold_val, warped_image_binary = cv2.threshold(warped_image_grey, lower, 255,
                                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show_processing:
        cv2.imshow('warped binary image', warped_image_binary)
    # cv2.imshow('warped binary image', warped_image_binary)
    # cv2.imshow('thresh2', threshold2)
    # cv2.waitKey(0)
    img_height, img_width = warped_image_binary.shape[0:2]
    cell_height = img_height // 9
    cell_width = img_width // 9

    cells = []

    for i in range(0, 9):

        for j in range(0, 9):
            d = 0

            if j % 3 == 0 or i % 3 == 0:  # thicker lines every 3 squares
                d = 3

            crop_amount = tolerance
            # crop_amount = 0
            h, w = i * cell_height + crop_amount, j * cell_width + crop_amount

            # cropping at specific intervals first to remove grid lines
            cell = warped_image_binary[h + d:h + cell_height - crop_amount, w + d:w + cell_width - crop_amount]

            cropped_height, cropped_width = cell.shape[0:2]

            cell2 = np.invert(cell)

            # add border to ensure that gridlines, if any are connected and wont be conisdered a contour
            # which will throw off blank cells, e.g. a blank cell with a gridline still showing could be considered a one
            # if the gridlines have gaps in them
            # cell2 = cv2.copyMakeBorder(cell2, 1,1,1,1, cv2.BORDER_CONSTANT,value=[255,255,255])

            contours = cv2.findContours(cell2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            number_img = None
            max_contour_area = float("-inf")
            if contours:

                for contour in contours:

                    area = cv2.contourArea(contour)
                    (x, y, w, h) = cv2.boundingRect(contour)

                    # Check if contour is within the center of the cell with small tolerance

                    if x - tolerance < cropped_width // 2 < x + w + tolerance:
                        if y - tolerance < cropped_height // 2 < y + h + tolerance:
                            if h < cropped_height and w < cropped_width:  # most likely a grid line if this is not true
                                #  < area to get rid of some noise, particularly broken up gridlines
                                # cropped h
                                avg = (cropped_height + cropped_width) // 2
                                if avg // 2 < area > max_contour_area:  # max contour area (with above conditions) should correspond to the number, if any

                                    max_contour_area = area

                                    number_img = cell[y:y + h, x:x + w]

            if not number_img is None:
                # Add blank (white) padding around image
                h_offset = (cropped_height - number_img.shape[0]) // 2
                w_offset = (cropped_width - number_img.shape[1]) // 2
                # - (d//2)
                number_img = cv2.copyMakeBorder(number_img, h_offset, h_offset, w_offset, w_offset, cv2.BORDER_CONSTANT,
                                                value=[255, 255, 255])
            '''
            if (i,j) in [(4,2), (3,3)]:
                print(cell.shape)
                print(max_contour_area)
                plt.figure(1)
                plt.imshow(cell, cmap='gray')
                plt.title("Cell")
                if number_img is None:
                    number_img = 255 * np.ones_like(cell, dtype=np.uint8)
                plt.figure(2)
                plt.imshow(number_img)
                plt.title("num img ")
                plt.figure(3)
                plt.imshow(cell2, cmap= 'gray')
                plt.title("cell2")
                plt.show()
            '''
            # number_img = get_largest_connected_component(cell, cropped_width, cropped_height, tolerance)
            if number_img is None:  # if number_img not set from above
                # Most likely cell is empty, no number
                cell = 255 * np.ones_like(cell, dtype=np.uint8)
                cell = np.asarray(cell)
                cell = cv2.resize(cell, (IMG_SIZE, IMG_SIZE))

            else:
                try:
                    cell = cv2.resize(number_img, (IMG_SIZE, IMG_SIZE))
                except cv2.error:
                    print("Could not find all cells in the image. Grid was not detected or was not largest contour in "
                          "image")
                    return

            cell = cell / 255
            cell = cell.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            cells.append(cell)

    print("Batch predicting...")
    predictions = CNN_model.predict(np.vstack(cells))
    print("Batch predicting done.")

    sudoku_grid = [[] for i in range(9)]
    row = 0
    col = 0
    for prediction in predictions:

        if show_certainties:
            print("Predictions: ", prediction)

        highest_probability = np.amax(prediction)
        cell_num = np.where(prediction == highest_probability)[0][0]

        if show_certainties:
            print(f"Row, Column : {row}, {col},  Number :{cell_num}, Certainty: {highest_probability:.5f} % ")
            print("-------------------------------------------")

        if highest_probability < PROBABILITY_THRESHOLD:
            cell_num = 0

        sudoku_grid[row].append(cell_num)
        col += 1

        if col == 9:
            col = 0
            row += 1

    print("Read in Sudoku Grid as: ")
    for row in sudoku_grid:
        print(row)

    return sudoku_grid


def draw_numbers(sudoku_grid, solved_grid, original_corners, warped_img, final_output_image,
                 show_processing=False):
    img_height, img_width = warped_img.shape[0:2]

    cell_height = (img_height // 9)
    cell_width = (img_width // 9)

    num = "8"  # 8 is an arbitrary choice, all numbers should have the same bounded dimensions
    scale = 0.5
    # Find maximum scale size for font when drawing text on image that stays within each box of the sudoku grid
    for i in range(1, 9):

        (text_width, text_height), baseline = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        # center_x, center_y = (cell_width - text_width) // 2, (cell_height - text_height) // 2

        if text_width > cell_width - 10 or text_height > cell_height - 10:
            break
        scale += 0.25

    for i in range(9):
        for j in range(9):
            if sudoku_grid[i][j] != solved_grid[i][j]:
                d = 0
                if i % 3 == 0 or j % 3 == 0:
                    d = 2
                h, w = i * cell_height + d, j * cell_width + d
                text = str(solved_grid[i][j])
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                center_x, center_y = (cell_width - text_width) // 2, (cell_height - text_height) // 2
                # position parameter in putText is for bottom-left corner

                pos = center_x + w, center_y + h + text_height
                pos = np.array(pos)

                cv2.putText(warped_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # corners = [top_left, top_right, bottom_left, bottom_right]
    warp_to_points = original_corners
    warped_corners = np.array([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]], dtype='float32')
    warped_matrix = cv2.getPerspectiveTransform(warped_corners, warp_to_points)

    non_warped_image_filled = cv2.warpPerspective(warped_img, warped_matrix, final_output_image.shape[0:2][::-1])

    # Normalize intensity
    alpha_mask = non_warped_image_filled[:, :, 2] / 255

    # For each channel of our output image use the alpha blending formula: alpha * F + (1-alpha) * B
    # F and B being Foreground and Background image respectively
    for channel in range(3):  # 3 channels for rgb image
        f = (1 - alpha_mask) * final_output_image[:, :, channel]
        f2 = alpha_mask * non_warped_image_filled[:, :, channel]
        final_output_image[:, :, channel] = f + f2

    if show_processing:
        cv2.imshow('Filled in grid', warped_img)
        cv2.imshow('Filled in original grid', non_warped_image_filled)

    cv2.imshow("Final Image", final_output_image)


def sudoku_solver(sudoku_grid, solved_grids_list=None):
    if solved_grids_list is None:
        solved_grids_list = []
    for y in range(9):
        for x in range(9):
            if sudoku_grid[y][x] == 0:
                for n in range(1, 10):
                    if check_if_num_valid(x, y, n, sudoku_grid):
                        sudoku_grid[y][x] = n
                        sudoku_solver(sudoku_grid, solved_grids_list)
                        sudoku_grid[y][x] = 0  # reset cell to empty if above doesn't work
                return solved_grids_list

    solved_grids_list.append(copy.deepcopy(sudoku_grid))


def check_if_num_valid(x, y, n, grid):
    top_left_x = (x // 3) * 3
    top_left_y = (y // 3) * 3
    for i in range(9):
        if grid[top_left_y + i // 3][top_left_x + i % 3] == n:  # check the square

            return False
        if grid[y][i] == n or grid[i][x] == n:
            return False
    return True


def check_if_grid_full(grid):
    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:
                return False
    return True


def check_if_valid_grid(sudoku_grid):
    if not sudoku_grid:
        return False
    nums = 0
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i][j] != 0:
                nums += 1
    if nums < 18:
        return False, "Too few numbers found to solve"

    r = 0
    for row in sudoku_grid:
        l = []
        for num in row:
            if num != 0:
                l.append(num)
        if len(set(l)) < len(l):
            msg = f"Invalid row: {r}, with numbers: {l}"
            return False, msg
        r += 1

    c = 0
    for i in range(9):
        col = []
        for j in range(9):
            if sudoku_grid[j][i] != 0:
                col.append(sudoku_grid[j][i])
        if len(set(col)) < len(col):
            msg = f"Invalid column: {c}, with numbers: {col}"
            return False, msg
        c += 1
    s = 0
    for row in range(3):
        for col in range(3):
            square = []
            for y in range(col * 3, col * 3 + 3):
                for x in range(row * 3, row * 3 + 3):
                    if sudoku_grid[y][x] != 0:
                        square.append(sudoku_grid[y][x])

            if len(set(square)) < len(square):
                msg = f"Invalid Square: {s}, with numbers: {square}"
                return False, msg
            s += 1

    return True, None


# Returns True if at least one solution, False if no solution found
def get_sudoku(original_image, dilate_amount=1, Otsu=True, show_certainties=True, show_processing=False):
    print("Reading Image...\n")

    processed_image = process_image(original_image, dilate_amount=dilate_amount, show_processing=show_processing)

    try:
        CNN_model = keras.models.load_model("CNN_model")
    except (OSError, ValueError, FileNotFoundError) as e:
        print("Could not load CNN model from 'CNN_model' Directory")
        print(e)
        return False, None, None, None

    grid_image, corners = get_grid_image(processed_image, original_image, show_processing=show_processing)

    if not grid_image is None:

        unsolved_sudoku_grid = get_grid_numbers(grid_image, CNN_model, Otsu=Otsu, tolerance=5,
                                                show_certainties=show_certainties,
                                                show_processing=show_processing)

        valid_grid, msg = check_if_valid_grid(unsolved_sudoku_grid)

        if valid_grid:

            return True, unsolved_sudoku_grid, corners, grid_image
        else:

            print("Did not  read in a valid grid - cannot solve")
            print(msg)

    return False, None, None, None


def get_solution(sudoku_grid):
    if check_if_grid_full(sudoku_grid):
        print("Sudoku is already solved/filled in.")
        return

    list_solved_grids = sudoku_solver(sudoku_grid)
    # print(list_solved_grids)

    if not list_solved_grids:
        print("Invalid Sudoku - no solutions - possibly misread numbers")


    else:
        if len(list_solved_grids) > 1:
            print("Invalid Sudoku - multiple solutions - possibly misread numbers. \n")
        else:
            solved_grid = list_solved_grids[0]
            return solved_grid

    return


def main(image_path, show_processing=False, show_certainties=False):
    #image_path = 12.jpeg"
    original_image = read_in_image(image_path)
    while original_image is None:
        image_path = input("\nEnter image path: ")
        original_image = read_in_image(image_path)

    final_output_image = original_image.copy()

    solved_grid = None
    unsolved_sudoku_grid, corners, grid_image = None, None, None

    # Try different combinations of image processing to get a solved grid
    # Can also try changing PROBABILITY_THRESHOLD constant defined at top of file, e.g. to 0.5
    for i in range(4):
        print("Re-reading image with different processing...")
        if i == 0:
            valid, unsolved_sudoku_grid, corners, grid_image = get_sudoku(original_image,
                                                                          show_certainties=show_certainties,
                                                                          show_processing=show_processing)
        elif i == 1:  # Try with different thresholding of warped grid
            valid, unsolved_sudoku_grid, corners, grid_image = get_sudoku(original_image, Otsu=False,
                                                                          show_certainties=show_certainties,
                                                                          show_processing=show_processing)
        elif i == 2:  # Change dilation (helps if the grid's gridlines have gaps, as if they do then they can be seen as a valid contour)
            valid, unsolved_sudoku_grid, corners, grid_image = get_sudoku(original_image, Otsu=True, dilate_amount=3,
                                                                          show_certainties=show_certainties,
                                                                          show_processing=show_processing)
        else:  # Change dilation and thresholding type
            valid, unsolved_sudoku_grid, corners, grid_image = get_sudoku(original_image, Otsu=False,
                                                                          dilate_amount=3,
                                                                          show_certainties=show_certainties,
                                                                          show_processing=show_processing)
        if valid:
            print("Solving...")
            solved_grid = get_solution(unsolved_sudoku_grid)
            if solved_grid:
                print("\n--------------Solved Grid--------------\n")

                for row in solved_grid:
                    print(row)
                break

    if solved_grid:
        draw_numbers(unsolved_sudoku_grid, solved_grid, corners, grid_image, final_output_image,
                     show_processing=show_processing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(show_certainties=False, show_processing=False)