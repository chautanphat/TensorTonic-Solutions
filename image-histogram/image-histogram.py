def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    histogram = [0] * 256
    for row in image:
        for col in row:
            histogram[col] += 1
    return histogram