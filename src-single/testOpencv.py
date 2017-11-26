# from cv2 import cv2
import cv2
import skimage.io
import skimage.transform
import numpy as np


path = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/trainData/Left/1-left-image_00000000_0.png'
def load_image( path, pre_height=274, pre_width=274, height=256, width=256 ):
    try:
        # img = skimage.io.imread( path ).astype( float )
        img = cv2.imread(path).astype(float)
    except:
        print "Can't load image!"
        return None

    img /= 255.

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
    return (resized_img * 2) - 1  # (resized_img - 127.5)/127.5
    # return resized_img


dst_image = load_image(path)
test_image = (255. * (dst_image+1)/2.).astype(int)
cv2.imshow("test",test_image)
# cv2.waitKey(0)
cv2.imwrite('../results/Single/dst.png',test_image)
