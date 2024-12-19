from PIL import Image
import cv2
import numpy as np
import os

def reduce_colors(image, k=5):
    """
    Reduces the number of colors in an image using k-means clustering.
    """
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # K-means clustering
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    reduced = centers[labels.flatten()]
    reduced_img = reduced.reshape((image.shape))
    return reduced_img  # Return the reduced color image

def remove_background(image):
    """
    Removes the background by making near-white regions transparent.
    """
    img = Image.fromarray(image).convert("RGBA")
    arr = np.array(img)

    # Replace near-white background with transparency
    mask = (arr[:, :, :3] > [240, 240, 240]).all(axis=-1)
    arr[mask] = [0,0,0,0]
    img_out = Image.fromarray(arr, 'RGBA')
    return img_out  # Return the image with background removed

def main(input_dir, output_dir):
    for img_fp in os.listdir(input_dir):
        input_path = os.path.join(input_dir, img_fp)
        output_path = os.path.join(output_dir, f"final_{os.path.splitext(img_fp)[0]}.png")

        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reduced_colors_img = reduce_colors(img, k=50)
        final_img = remove_background(reduced_colors_img)

        final_img.save(output_path)

if __name__ == '__main__':
    input_dir = 'raw-art/'
    output_dir = 'processed-art/'
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir)