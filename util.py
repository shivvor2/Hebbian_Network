import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

def random_memory(dims: np.ndarray):    
    # Generate random memory
    memory = np.random.randint(0,2,dims)
    
    # Change all 0s to -1s
    memory[memory <= 0] = -1 
    return memory

def image_memory(image_path: str, dims: np.ndarray = None):
    image = Image.open(image_path)
    if dims:
        image = image.resize(dims)
    image = image.convert("L")
    memory = np.array(image)
    memory = (memory >= 128).astype(float)
    memory[memory == 0] = -1 # Change all 0s to -1s
    return memory


def resample_with_interpolation(original_matrix, resampled_shape, threshold = 0.5):
    original_shape = original_matrix.shape
    scaling_factors = [resampled_dim / original_dim for resampled_dim, original_dim in zip(resampled_shape, original_shape)]
    resampled_matrix = np.zeros(resampled_shape)

    # Generate coordinates for each pixel in the resampled matrix
    coordinates = np.indices(resampled_shape)

    # Scale the coordinates to match the original matrix dimensions
    scaled_coordinates = coordinates / scaling_factors[:, np.newaxis, np.newaxis]

    # Perform bilinear interpolation
    resampled_matrix = map_coordinates(original_matrix, scaled_coordinates, order=1)
    
    # Threshold the resampled matrix
    resampled_matrix[resampled_matrix < threshold] = -1
    resampled_matrix[resampled_matrix >= threshold] = 1
    
    return resampled_matrix

def print_array(array: np.array): #Temporary, will change later, can only print 2d, binary matrices correctly
    array = array.tolist()
    array = [["□" if x > 0 else "■" for x in row] for row in array]
    for row in array:
        print(' '.join(map(str, row)))