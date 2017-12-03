# Frizy :added
from __future__ import absolute_import

import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *


n_epochs = 100000
# Frizy changed: 0.0002->0.002
# learning_rate_val = 0.0002
# learning_rate_val = 0.002
learning_rate_val = 0.00148

# Frizy changed: 0.0001->0.001
# weight_decay_rate =  0.0001
weight_decay_rate =  0.001

momentum = 0.9

# Frizy
batch_size = 100
# batch_size = 3

lambda_recon = 0.999
lambda_adv = 0.001

overlap_size = 7
hiding_size = 64

# Frizy add: fill
resultDim = 256
fill_one_side = (resultDim-hiding_size)/2


trainset_path = '../data/MulData/mul_trainset.pickle'
testset_path  = '../data/MulData/mul_testset.pickle'
#dataset_path = '/media/storage3/Study/data/Paris/'
dataset_path = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/'

model_path = '../models/Mul/'
result_path= '../results/Mul/'
pretrained_model_path = None # '/home/lab/PycharmProjects/Inpainting-multi/models/Mul-old/model-150'

if not os.path.exists(model_path):
    os.makedirs( model_path )

if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):

    # Frizy add for mul
    # trainset_dir = os.path.join(dataset_path, 'trainData/Left')
    trainset_dir_left  = os.path.join( dataset_path, 'trainData/Left' )
    trainset_dir_right = os.path.join(dataset_path, 'trainData/Right')

    testset_dir_left  = os.path.join(dataset_path, 'testData/Left')
    testset_dir_right = os.path.join( dataset_path, 'testData/Right' )

    # Frizy changed for mul dataset
    # trainset = pd.DataFrame({'image_path': map(lambda x: os.path.join( trainset_dir, x ), os.listdir(trainset_dir))})
    # testset = pd.DataFrame({'image_path': map(lambda x: os.path.join( testset_dir, x ), os.listdir(testset_dir))})
    trainset_list_left = map(lambda x: os.path.join(trainset_dir_left, x), os.listdir(trainset_dir_left))
    trainset_list_left.sort()

    trainset_list_right = map(lambda x: os.path.join(trainset_dir_right, x),os.listdir(trainset_dir_right))
    trainset_list_right.sort()

    trainset = pd.DataFrame({'image_path_left': trainset_list_left, 'image_path_right': trainset_list_right})


    testset_list_left = map(lambda x: os.path.join(testset_dir_left, x), os.listdir(testset_dir_left))
    testset_list_left.sort()

    testset_list_right = map(lambda x: os.path.join(testset_dir_right, x),os.listdir(testset_dir_right))
    testset_list_right.sort()

    testset = pd.DataFrame({'image_path_left': testset_list_left, 'image_path_right': testset_list_right })

    trainset.to_pickle( trainset_path )
    testset.to_pickle( testset_path )
else:
    trainset = pd.read_pickle( trainset_path )
    testset = pd.read_pickle( testset_path )

testset.index = range(len(testset))
is_train = tf.placeholder( tf.bool, shape=None )

learning_rate = tf.placeholder( tf.float32, [])

# Frizy changed : input image 128-->
# images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")
images_tf = tf.placeholder( tf.float32, [batch_size, resultDim, resultDim, 6], name="images")

images_hiding = tf.placeholder( tf.float32, [batch_size, hiding_size, hiding_size, 3], name='images_hiding')

model = Model()

bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

adversarial_pos = model.build_adversarial(images_hiding, is_train)
adversarial_neg = model.build_adversarial(reconstruction, is_train, reuse=True)
#adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])
adversarial_all = tf.concat([adversarial_pos, adversarial_neg], 0)

# Applying bigger loss for overlapping region
mask_recon = tf.pad(tf.ones([hiding_size - 2*overlap_size, hiding_size - 2*overlap_size]), [[overlap_size,overlap_size], [overlap_size,overlap_size]])
mask_recon = tf.reshape(mask_recon, [hiding_size, hiding_size, 1])
#mask_recon = tf.concat(2, [mask_recon]*3)
mask_recon = tf.concat([mask_recon]*3,2)
mask_overlap = 1 - mask_recon

loss_recon_ori = tf.square( images_hiding - reconstruction )
loss_recon_center = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1,2,3])))  # Loss for non-overlapping region
loss_recon_overlap = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1,2,3]))) * 10. # Loss for overlapping region
loss_recon = loss_recon_center + loss_recon_overlap

def get_adv_loss():
    return tf.reduce_mean( tf.log(1e-6+tf.nn.sigmoid(adversarial_pos)) + tf.log(1.+1e-6 - tf.nn.sigmoid(adversarial_neg)) )

loss_adv = get_adv_loss()
loss_G = loss_adv * lambda_adv + loss_recon * lambda_recon
loss_D = -loss_adv #* lambda_adv

var_G = filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables())
var_D = filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables())

W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

#loss_G += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_G)))
#loss_D += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_D)))
loss_G += weight_decay_rate * tf.reduce_mean(tf.stack( map(lambda x: tf.nn.l2_loss(x), W_G)))
loss_D += weight_decay_rate * tf.reduce_mean(tf.stack( map(lambda x: tf.nn.l2_loss(x), W_D)))

sess = tf.InteractiveSession()

optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -20., 20.), gv[1]], grads_vars_G)
train_op_G = optimizer_G.apply_gradients( grads_vars_G )

optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -20., 20.), gv[1]], grads_vars_D)
train_op_D = optimizer_D.apply_gradients( grads_vars_D )

saver = tf.train.Saver(max_to_keep=3)

tf.initialize_all_variables().run()

#if pretrained_model_path is not None and os.path.exists( pretrained_model_path ):
#    saver.restore( sess, pretrained_model_path )
# saver.restore( sess, pretrained_model_path )

iters = 0

loss_D_val = 0.
loss_G_val = 0.

for epoch in range(n_epochs):
    trainset.index = range(len(trainset))
    trainset = trainset.iloc[np.random.permutation(len(trainset))]

    for start,end in zip(
            range(0, len(trainset), batch_size),
            range(batch_size, len(trainset), batch_size)):

        # Frizy add for mul
        image_paths_left = trainset[start:end]['image_path_left'].values
        image_paths_right = trainset[start:end]['image_path_right'].values
        images_ori_all = map(lambda left, right: load_image_mul( left, right, height=resultDim,width=resultDim),
                         image_paths_left,image_paths_right )

        # images_ori_left = map(lambda img: img[:, :, 0:3], images_ori_all)
        # images_ori_right = map(lambda img: img[:, :, 3:6], images_ori_all)

        if iters % 2 == 0:
            images_ori_all = map(lambda img: img[:,::-1, :], images_ori_all)

        is_none_all = np.sum(map(lambda x: x is None, images_ori_all))
        if is_none_all > 0 : continue

        # Frizy
        # images_crops = map(lambda x: crop_random(x, x=32, y=32), images_ori)
        images_allAndcrops = map(lambda x: crop_random_all(x, x=fill_one_side, y=fill_one_side), images_ori_all)
        images_all, crops,_,_ = zip(*images_allAndcrops)

        # Printing activations every 10 iterations
        if iters % 300 == 0:
            test_image_paths_left = testset[:batch_size]['image_path_left'].values
            test_image_paths_right = testset[:batch_size]['image_path_right'].values
            test_images_ori_all = map(lambda left,right: load_image_mul(left,right,height=resultDim, width=resultDim),
                                  test_image_paths_left, test_image_paths_right)

            # Frizy
            # test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
            test_images_allAndcrop = map(lambda x: crop_random_all(x, width=hiding_size, height=hiding_size, x=fill_one_side, y=fill_one_side), test_images_ori_all)
            test_images_all, test_crops, xs,ys = zip(*test_images_allAndcrop)

            reconstruction_vals, adv_pos_val, adv_neg_val, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                    [reconstruction, adversarial_pos, adversarial_neg, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
                    feed_dict={
                        images_tf: test_images_all,
                        images_hiding: test_crops,
                        is_train: False
                        })

            # Generate result every 1000 iterations
            if iters % 300 == 0:
                ii = 0
                for rec_val, img,x,y in zip(reconstruction_vals, test_images_all, xs, ys):
                    rec_hid = (255. * (rec_val+1)/2.).astype(int)
                    rec_con = (255. * (img[:,:,0:3] +1)/2.).astype(int)

                    rec_con[y:y+64, x:x+64] = rec_hid
                    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.png'), rec_con)
                    ii += 1

                if iters == 0:
                    ii = 0
                    for test_image in test_images_ori_all:
                        test_image = (255. * ( test_image[:,:,0:3] + 1 )/2.).astype(int)

                        # Frizy changed
                        # test_image[32:32+64,32:32+64] = 0
                        test_image[fill_one_side:fill_one_side + hiding_size, fill_one_side:fill_one_side + hiding_size] = 0
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.png'), test_image)
                        ii += 1

            # ipdb.set_trace()

            print "========================================================================"
            print bn1_val.max(), bn1_val.min()
            print bn2_val.max(), bn2_val.min()
            print bn3_val.max(), bn3_val.min()
            print bn4_val.max(), bn4_val.min()
            print bn5_val.max(), bn5_val.min()
            print bn6_val.max(), bn6_val.min()
            print debn4_val.max(), debn4_val.min()
            print debn3_val.max(), debn3_val.min()
            print debn2_val.max(), debn2_val.min()
            print debn1_val.max(), debn1_val.min()
            print recon_ori_vals.max(), recon_ori_vals.min()
            print reconstruction_vals.max(), reconstruction_vals.min()
            print adv_pos_val.max(), adv_pos_val.min()
            print adv_neg_val.max(), adv_neg_val.min()
            print loss_G_val, loss_D_val
            print "========================================================================="

            if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                print "NaN detected!!"
                ipdb.set_trace()

        # Generative Part is updated every iteration
        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_val, reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
                feed_dict={
                    images_tf: images_all,
                    images_hiding: crops,
                    learning_rate: learning_rate_val,
                    is_train: True
                    })

        _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
                [train_op_D, loss_D, adversarial_pos, adversarial_neg],
                feed_dict={
                    images_tf: images_all,
                    images_hiding: crops,
                    learning_rate: learning_rate_val/100.,
                    is_train: True
                    })

        print "Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_val,  "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max()
        print "cur_learning_rate:", learning_rate_val

        iters += 1


    if epoch != 0 and epoch % 100 == 0:
        saver.save(sess, model_path + 'model', global_step=epoch)
        learning_rate_val *= 0.99
        print "cur_learning_rate:", learning_rate_val

    if epoch != 0 and epoch % 10 == 0:
        saver.save(sess, model_path + 'model', global_step=epoch)



