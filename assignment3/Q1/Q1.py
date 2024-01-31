import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_zeros_like(array, dtype=None):
    if dtype is None:
        dtype = array.dtype
    shape = array.shape
    return create_zeros_array(shape, dtype)

def create_zeros_array(shape, dtype):
    if len(shape) == 1:
        return np.array([0] * shape[0], dtype=dtype)
    else:
        return np.array([create_zeros_array(shape[1:], dtype) for _ in range(shape[0])])

def custom_filter2D(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape

    result = custom_zeros_like(image, dtype=np.float32)

    # Pad the image to handle convolution at the borders
    padded_image = np.pad(image, ((k_height//2, k_height//2), (k_width//2, k_width//2)), mode='constant')

    for y in range(height):
        for x in range(width):
            roi = padded_image[y:y+k_height, x:x+k_width]
            result[y, x] = np.sum(roi * kernel)

    return result

def custom_sobel(img, axis='x'):
    if axis == 'x':
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif axis == 'y':
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:
        raise ValueError("Invalid axis. Use 'x' or 'y'.")

    return custom_filter2D(img, kernel)

def custom_vstack(arrays):
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.array([custom_vstack(arrays[1:])] + [arrays[0]])

def custom_lstsq(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    AT = A.T
    ATA = np.dot(AT, A)
    ATb = np.dot(AT, b)
    x = solve_linear_system(ATA, ATb)
    return x

def solve_linear_system(A, b):
    n = len(A)
    x = [0.0] * n  # Ensure x is initialized as a list of floats

    for i in range(n):
        pivot_row = max(range(i, n), key=lambda j: abs(A[j][i]))

        if abs(A[pivot_row][i]) < 1e-10:
            # Handle zero pivot by skipping this column
            continue

        A[i], A[pivot_row] = A[pivot_row], A[i]
        b[i], b[pivot_row] = b[pivot_row], b[i]

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-10:
            # Handle zero pivot by skipping this row
            continue

        x[i] = b[i] / A[i][i]
        for j in range(i):
            b[j] -= A[j][i] * x[i]

    return x

def lucas_kanade_optical_flow(I1, I2, window_size=5):
    Ix = custom_sobel(I1, axis='x')
    Iy = custom_sobel(I1, axis='y')
    
    It = I2 - I1

    half_window = window_size // 2

    u = custom_zeros_like(I1, dtype=np.float32)
    v = custom_zeros_like(I1, dtype=np.float32)

    for y in range(half_window, I1.shape[0] - half_window):
        for x in range(half_window, I1.shape[1] - half_window):
            Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            It_window = -It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

            A = custom_vstack([Ix_window, Iy_window]).T
            uv = custom_lstsq(A, It_window)

            u[y, x] = uv[0]
            v[y, x] = uv[1]

    return u, v

# Read the video file
cap = cv2.VideoCapture('sample.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    u, v = lucas_kanade_optical_flow(prvs, next_frame)

    # Display frames and optical flow
    plt.subplot(2, 2, 1), plt.imshow(prvs, cmap='gray')
    plt.title('Frame 1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(next_frame, cmap='gray')
    plt.title('Frame 2'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(u, cmap='viridis')
    plt.title('Optical Flow U'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(v, cmap='viridis')
    plt.title('Optical Flow V'), plt.xticks([]), plt.yticks([])

    plt.show()

    # Update the previous frame for the next iteration
    prvs = next_frame

cap.release()
cv2.destroyAllWindows()