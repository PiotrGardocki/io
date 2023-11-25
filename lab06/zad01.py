import numpy as np
from matplotlib import pyplot as plt

img_size = 128
data = np.zeros((img_size, img_size), dtype=np.int16)

def draw(img, x, y, color):
    img[x, y] = color

def draw_matrix(matrix):
    matrix = np.array([
        [
            [pixel, pixel, pixel] for pixel in row
        ] for row in matrix
    ], dtype=np.uint8)

    plt.imshow(matrix, interpolation='nearest')
    plt.show()

draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

draw_matrix(data)

def mult_matrix_by_filter(matrix, filter, stride=1):
    matrix = np.array([
        [
            np.sum(data[row_idx:(row_idx+3), col_idx:(col_idx+3)] * filter)
            for col_idx in range(0, matrix.shape[1] - 2, stride)
        ]
        for row_idx in range(0, matrix.shape[0] - 2, stride)
    ], dtype=np.int16)
    return matrix

vertical_filter1 = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
], dtype=np.int16)

vertical_filter2 = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
], dtype=np.int16)

horizontal_filter1 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=np.int16)

horizontal_filter2 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.int16)

sobel1 = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0],
], dtype=np.int16)

sobel2 = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2],
], dtype=np.int16)

sobel3 = np.array([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2],
], dtype=np.int16)

filters = [
    vertical_filter1,
    vertical_filter2,
    horizontal_filter1,
    horizontal_filter2,
    sobel1,
    sobel2,
    sobel3,
]

print("Stride 1")
for f in filters:
    draw_matrix(mult_matrix_by_filter(data, f))

print("Stride 2")
for f in filters:
    draw_matrix(mult_matrix_by_filter(data, f, 2))
