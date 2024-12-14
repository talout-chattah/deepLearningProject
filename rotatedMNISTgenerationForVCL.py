import gzip
import pickle
import random
import numpy as np
def rotate_image_manual(image, angle):
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Image dimensions
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    
    # Create an empty array for the rotated image
    rotated_image = np.zeros_like(image)
    
    # Iterate over every pixel in the output image
    for x in range(w):
        for y in range(h):
            # Translate coordinates to origin (center of the image)
            trans_x = x - center_x
            trans_y = y - center_y
            
            # Apply the rotation matrix
            orig_x = int(center_x + (trans_x * cos_theta - trans_y * sin_theta))
            orig_y = int(center_y + (trans_x * sin_theta + trans_y * cos_theta))
            
            # Check if the original coordinates are within bounds
            if 0 <= orig_x < w and 0 <= orig_y < h:
                rotated_image[y, x] = image[orig_y, orig_x]
    
    return rotated_image

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

rotated_train_set_images = []
for image in train_set[0]:

    # Flattened MNIST image array
    image_flattened = image  

    # Step 1: Reshape to 28x28
    image_2d = image_flattened.reshape(28, 28)

    # Step 2: Choose a random angle from [0, 45, 90, 135, 180, 225, 270, 315]
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

    rotated_image_2d = rotate_image_manual(image_2d, angle)

    # Step 3: Flatten back to 1D if needed
    rotated_image_flattened = rotated_image_2d.flatten()

    rotated_train_set_images.append(rotated_image_flattened)

rotated_train_set = (np.array(rotated_train_set_images),train_set[1])

rotated_valid_set_images = []
for image in valid_set[0]:

    # Flattened MNIST image array
    image_flattened = image 

    # Step 1: Reshape to 28x28
    image_2d = image_flattened.reshape(28, 28)

    # Step 2: Choose a random angle from [0, 45, 90, 135, 180, 225, 270, 315]
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

    rotated_image_2d = rotate_image_manual(image_2d, angle)

    # Step 3: Flatten back to 1D if needed
    rotated_image_flattened = rotated_image_2d.flatten()

    rotated_valid_set_images.append(rotated_image_flattened)

rotated_valid_set = (np.array(rotated_valid_set_images), valid_set[1])

rotated_test_set_images = []
for image in test_set[0]:

    # Flattened MNIST image array
    image_flattened = image

    # Step 1: Reshape to 28x28
    image_2d = image_flattened.reshape(28, 28)

    # Step 2: Choose a random angle from [0, 45, 90, 135, 180, 225, 270, 315]
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

    rotated_image_2d = rotate_image_manual(image_2d, angle)

    # Step 3: Flatten back to 1D if needed
    rotated_image_flattened = rotated_image_2d.flatten()

    rotated_test_set_images.append(rotated_image_flattened)

rotated_test_set = (np.array(rotated_test_set_images),test_set[1])



with gzip.open('data/rotatedmnist.pkl.gz', 'wb', compresslevel=9) as f:
    pickle.dump((rotated_train_set, rotated_valid_set, rotated_test_set), f)

print("the generated file saved")