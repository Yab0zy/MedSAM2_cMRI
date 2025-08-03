import numpy as np
from PIL import Image
from skimage import measure, morphology
import matplotlib.pyplot as plt

def getLargestCC(segmentation):
    """
    Get the largest connected component

    Paramters:
        segmentation: array of the segmentation result

    Result:
        largestCC: array of the largest connected component of the segmentation result,
        with the same shape as input array
    """
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.

    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.

    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # [3, image_size, image_size]
        resized_array[i] = img_array

    return resized_array
def generate_box(labelSlice, target_obj, padding=10):
    """
    Generate a prompt box from gt

    Parameters:
       labelSlice: one slice of gt, array of shape(h, w).
       target_obj: target object id of this prompt.
       padding: describe the extension of the box, to make the prompt more subtle
    """
    rows, cols = np.where(labelSlice == target_obj)
    min_row, max_row = int(max(np.min(rows)-padding, 0)), int(min(np.max(rows)+padding, image_height-1))
    min_col, max_col = int(max(np.min(cols)-padding, 0)), int(min(np.max(cols)+padding, image_width-1))
    bbox = np.array([min_col, min_row, max_col, max_row])
    return bbox

def generate_point(labelSlice, target_obj, background=False):
    """
    Generate a prompt point from gt

    Parameters:
        labelSlice: one slice of gt, array of shape(h, w).
        target_obj: target object id of this prompt.
        background: True--the point is from the background,
                    False--the point is from the target

    Returns:
        points: array of shape (1, 2).
        label: array of shape (1, ), describing whether the point belong to background or target
    """
    rows, cols = np.where(labelSlice == target_obj)
    mid_idx = len(rows)//2
    point_x = cols[mid_idx]
    point_y = rows[mid_idx]
    points = np.array([[point_x, point_y]], dtype=np.float32)
    if not background:
        label = np.array([1], np.int32)
    else:
        label = np.array([0], np.int32)
    return points, label

