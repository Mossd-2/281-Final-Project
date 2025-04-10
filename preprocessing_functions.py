import numpy as np
import cv2

# Normalize images function
def normalize_images(images, normalization_range=[0, 1], fit_on=None):
    """
    Normalize an array of images to a specified range.

    Parameters:
    -----------
    images : numpy.ndarray
        Array of images with shape (n_samples, height, width) or (n_samples, height, width, channels)
    normalization_range : list, optional
        Target range for normalization, either [0, 1] or [-1, 1]
    fit_on : numpy.ndarray, optional
        If provided, normalization parameters will be computed from this array
        and applied to the images array (useful for normalizing test data with train statistics)

    Returns:
    --------
    numpy.ndarray
        Normalized images
    """
    # Store original shape and reshape to (n_samples, pixels)
    original_shape = images.shape
    if len(original_shape) == 4:  # Images with channels
        n_samples, height, width, channels = original_shape
        flattened = images.reshape(n_samples, height * width * channels)
    else:  # Grayscale images
        n_samples, height, width = original_shape
        flattened = images.reshape(n_samples, height * width)

    # Convert to float if not already
    flattened = flattened.astype(np.float32)

    if normalization_range == [0, 1]:
        # Simple min-max scaling to [0, 1]
        if fit_on is not None:
            # Reshape fit_on array to match
            if len(fit_on.shape) == 4:
                fit_on_flat = fit_on.reshape(fit_on.shape[0], -1)
            else:
                fit_on_flat = fit_on.reshape(fit_on.shape[0], -1)

            min_val = fit_on_flat.min()
            max_val = fit_on_flat.max()
        else:
            min_val = flattened.min()
            max_val = flattened.max()

        normalized = (flattened - min_val) / (max_val - min_val)

    elif normalization_range == [-1, 1]:
        # Scale to [-1, 1]
        if fit_on is not None:
            # Reshape fit_on array to match
            if len(fit_on.shape) == 4:
                fit_on_flat = fit_on.reshape(fit_on.shape[0], -1)
            else:
                fit_on_flat = fit_on.reshape(fit_on.shape[0], -1)

            min_val = fit_on_flat.min()
            max_val = fit_on_flat.max()
        else:
            min_val = flattened.min()
            max_val = flattened.max()

        normalized = 2 * ((flattened - min_val) / (max_val - min_val)) - 1

    else:
        raise ValueError("normalization_range must be either [0, 1] or [-1, 1]")

    # Reshape back to original shape
    normalized = normalized.reshape(original_shape)

    return normalized



# Convert images to grayscale function
def convert_to_grayscale(images):
    """
    Convert RGB images to grayscale.

    Parameters:
    -----------
    images : numpy.ndarray
        Array of RGB images with shape (n_samples, height, width, 3)

    Returns:
    --------
    numpy.ndarray
        Array of grayscale images with shape (n_samples, height, width)
    """
    # Check if images are already grayscale
    if len(images.shape) == 3:
        return images  # Already grayscale

    n_samples = images.shape[0]
    grayscale_images = np.zeros((n_samples, images.shape[1], images.shape[2]), dtype=np.uint8)

    for i in range(n_samples):
        # Convert RGB to grayscale
        grayscale_images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

    return grayscale_images


def apply_contrast_normalization(images, method='clahe', clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply contrast normalization to a batch of images.

    Parameters:
    -----------
    images : numpy.ndarray
        Array of images with shape (n_samples, height, width) for grayscale
        or (n_samples, height, width, channels) for color images
    method : str, optional
        Method for contrast normalization, either 'clahe' (default) or 'histeq'
    clip_limit : float, optional
        Threshold for contrast limiting in CLAHE (only used if method='clahe')
    tile_grid_size : tuple, optional
        Size of grid for histogram equalization in CLAHE (only used if method='clahe')

    Returns:
    --------
    numpy.ndarray
        Contrast-normalized images with same shape as input
    """
    is_color = len(images.shape) == 4 and images.shape[3] > 1
    result = np.zeros_like(images, dtype=np.uint8)

    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        for i, img in enumerate(images):
            if is_color:
                # For color images, apply CLAHE to each channel separately
                for c in range(img.shape[2]):
                    result[i, :, :, c] = clahe.apply(img[:, :, c].astype(np.uint8))
            else:
                # For grayscale images
                result[i] = clahe.apply(img.astype(np.uint8))

    elif method == 'histeq':
        for i, img in enumerate(images):
            if is_color:
                # Convert to LAB space, apply histogram equalization to L channel
                lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                result[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # For grayscale images
                result[i] = cv2.equalizeHist(img.astype(np.uint8))

    return result


# Function to apply noise reduction
def reduce_noise(images, strength=1):
    # Convert to uint8 if needed (required for OpenCV functions)
    orig_dtype = images.dtype
    if orig_dtype != np.uint8:
        # Scale if needed based on data range
        if orig_dtype == np.float32 or orig_dtype == np.float64:
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = images.astype(np.uint8)
        else:
            images = images.astype(np.uint8)

    processed_images = np.zeros_like(images)

    h_luminance = max(2, strength * 3)
    h_color = max(2, strength * 3)
    search_window = 21
    template_window = 7
    # Apply the selected noise reduction method to each image
    for i in range(images.shape[0]):
        processed_images[i] = cv2.fastNlMeansDenoisingColored(
            images[i], None, h_luminance, h_color, template_window, search_window)

    # Convert back to original dtype if needed
    if orig_dtype != np.uint8:
        if orig_dtype == np.float32 or orig_dtype == np.float64:
            if images.max() <= 1.0:
                processed_images = processed_images.astype(np.float32) / 255.0

    return processed_images




