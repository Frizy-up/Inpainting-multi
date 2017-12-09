import skimage.io
import skimage.transform
from PIL import ImageFile
import os
#import ipdb

import numpy as np

import cv2

#def load_image( path, height=128, width=128 ):

# Frizy changed : load image 128-->512
# def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):
def load_image( path, pre_height=274, pre_width=274, height=256, width=256 ):
    try:
        # img = skimage.io.imread( path ).astype( float )
        img = cv2.imread(path).astype(float)
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    # Frizy changed
    # short_edge = min( img.shape[:2] )
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    # resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )
    # resized_img = skimage.transform.resize(img, [pre_height, pre_width])
    resized_img = cv2.resize(img, (pre_width,pre_height) )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5

def load_image_random( path, pre_height=274, pre_width=274, height=256, width=256 ):
    try:
        # img = skimage.io.imread( path ).astype( float )
        img = cv2.imread(path).astype(float)
    except:
        return None

    # img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    # Frizy changed
    # short_edge = min( img.shape[:2] )
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    # resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )
    # resized_img = skimage.transform.resize(img, [pre_height, pre_width])
    resized_img = cv2.resize(img, (pre_width,pre_height) )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    # return (resized_img * 2)-1 #(resized_img - 127.5)/127.5
    return resized_img


# Frizy add for mul : load two images
def load_image_mul_random( path_left, path_right, pre_height=274, pre_width=274, height=256, width=256 ):
    try:
        img_left = cv2.imread(path_left).astype(float)
        img_right= cv2.imread(path_right).astype(float)
    except:
        return None

    # img_left  /= 255.
    # img_right /= 255.

    if img_left is None or img_right is None: return None
    if len(img_left.shape) < 2 or len(img_right.shape)<2 : return None
    if len(img_left.shape) == 4 or len(img_right.shape)==4: return None
    if len(img_left.shape) == 2: img_left = np.tile(img_left[:,:,None], 3)
    if len(img_right.shape) == 2: img_right = np.tile(img_right[:, :, None], 3)
    if img_left.shape[2] == 4: img_left=img_left[:,:,:3]
    if img_right.shape[2] == 4: img_right = img_right[:, :, :3]
    if img_left.shape[2] > 4 or img_right.shape[2]>4 : return None

    resized_img_left = cv2.resize(img_left, (pre_width,pre_height) )
    resized_img_right = cv2.resize(img_right, (pre_width, pre_height))

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img_left = resized_img_left[ rand_y:rand_y+height, rand_x:rand_x+width, : ]
    resized_img_right = resized_img_right[rand_y:rand_y + height, rand_x:rand_x + width, :]

    # resized_img_left = (resized_img_left * 2) - 1
    # resized_img_right = (resized_img_right * 2) - 1

    resized_img_all = np.concatenate((resized_img_left,resized_img_right),axis=2)

    return resized_img_all


# Frizy changed: width , height
def crop_random(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap, height-overlap) if x is None else x
    random_x = np.random.randint(overlap, width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y


def damageImage(image):
    height = image.shape[0]
    width = image.shape[1]

    p1x = np.random.randint(0, width)
    p1y = np.random.randint(0, height)

    p2x = np.random.randint(0, width)
    p2y = np.random.randint(0, height)

    p3x = np.random.randint(0, width)
    p3y = np.random.randint(0, height)

    p4x = np.random.randint(0, width)
    p4y = np.random.randint(0, height)

    points = np.array([[[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]]])

    color = np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])

    cv2.fillConvexPoly(image, points, color)

    # return image


def crop_random_damage(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap, height-overlap) if x is None else x
    random_x = np.random.randint(overlap, width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()

    damageImage(image)

    image /= 255.
    crop  /= 255.

    image = (image * 2) - 1
    crop = (crop * 2) - 1

    return image, crop, random_x, random_y


# Frizy add: for crop for mul images
def crop_random_all_damage(image_ori_all, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori_all is None: return None
    random_y = np.random.randint(overlap, height-overlap) if x is None else x
    random_x = np.random.randint(overlap, width-overlap) if y is None else y

    crop = image_ori_all[:,:,0:3].copy()

    leftImage = image_ori_all[:,:,0:3].copy()
    rightImage = image_ori_all[:,:,3:6].copy()

    damageImage(leftImage)
    image_all_damage = np.concatenate((leftImage, rightImage),axis=2)

    image_all_damage /= 255.
    crop /= 255.

    image_all_damage = (image_all_damage * 2) - 1
    crop = (crop * 2) - 1

    return image_all_damage, crop, random_x, random_y