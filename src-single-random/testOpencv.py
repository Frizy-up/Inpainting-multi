# from cv2 import cv2
import cv2
import skimage.io
import skimage.transform
import numpy as np


path = '/home/lab/PycharmProjects/Inpainting-multi/test/1.png'
def load_image( path, pre_height=274, pre_width=274, height=256, width=256 ):
    try:
        # img = skimage.io.imread( path ).astype( float )
        img = cv2.imread(path).astype(float)
        print "srcImage:", img.shape, type(img)
    except:
        print "Can't load image!"
        return None

    # img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4: img = img[:, :, :3]
    if img.shape[2] > 4: return None

    # Frizy changed
    # short_edge = min( img.shape[:2] )
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    # resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )
    # resized_img = skimage.transform.resize(img, [pre_height, pre_width])
    resized_img = cv2.resize(img, (pre_width, pre_height))

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[rand_y:rand_y + height, rand_x:rand_x + width, :]

    # Frizy changed
    # return (resized_img * 2) - 1  # (resized_img - 127.5)/127.5
    return resized_img

def damageImage(image):
    print "damage Image",image.shape, type(image)
    height = image.shape[0]
    width = image.shape[1]
    print height, width

    p1x = np.random.randint(0, width)
    p1y = np.random.randint(0, height)

    p2x = np.random.randint(0, width)
    p2y = np.random.randint(0, height)

    p3x = np.random.randint(0, width)
    p3y = np.random.randint(0, height)

    p4x = np.random.randint(0, width)
    p4y = np.random.randint(0, height)

    points = np.array([[ [p1x,p1y],[p2x,p2y],[p3x,p3y],[p4x,p4y] ]])

    color = np.array([np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)])

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


dst_image = load_image(path)
# test_image = (255. * (dst_image+1)/2.).astype(float)
# cv2.imshow("test",test_image)

# test_image2 = damageImage(test_image)

image, crop, _, _ = crop_random_damage(dst_image)

image = (255. * (image+1)/2.).astype(int)
crop = (255. * (crop+1)/2.).astype(int)
cv2.imwrite('/home/lab/PycharmProjects/Inpainting-multi/test/image.png',image)
cv2.imwrite('/home/lab/PycharmProjects/Inpainting-multi/test/crop.png',crop)
