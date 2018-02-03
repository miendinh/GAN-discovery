import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

images = []
size = 210

def rgba2rbg(image):
    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

def load_imgs():
    global images
    for i in range(1, size + 1):
        image = plt.imread('dataset/flower_images/{0:04d}.png'.format(i))
        image = rgba2rbg(image)
        images.append(image)
    random.shuffle(images)
    return images

def next_batch(step, batch_size):
    global images
    if images == []:
        print('loading...')
        images = load_imgs()
    return np.array(images[step:step+batch_size])

# if __name__ == '__main__':
#     images = next_batch(0, 10)
#     plt.imshow(images[0], cmap='gray')
#     plt.show()
#     print(images.shape)
