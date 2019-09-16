from __future__ import print_function
import os
import glob
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
from faceReconstruction.load_data import *
from faceReconstruction.reconstruct_mesh import Reconstruction


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = args.levels
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels
        self.n_frames =5

        # if args.phase == 'train':
        self.crop_size = args.patch_size
        # self.data_dir = args.data_dir
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)

        # self.train_dir = os.path.join('./checkpoints', args.face)
        self.train_dir = os.path.join('./checkpoints', '%s_%d_%d'%(args.face, args.levels, args.patch_size))
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = combine_img()
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./dataset/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.cast(img_b, tf.int32)
            img_b = tf.concat([img_b, img_b, img_b, img_b, img_b, img_b], axis=2)

            coeff = self.data_queue[6]


            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b, coeff

        def combine_img():
            ll_img = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[1]])),
                                          channels=3)
            l_img  = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[2]])),
                                          channels=3)
            m_img  = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[3]])),
                                          channels=3)
            r_img  = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[4]])),
                                          channels=3)
            rr_img = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[5]])),
                                          channels=3)
            render_img = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[7]])),
                                          channels=3)
            com_img = tf.concat([tf.cast(ll_img, tf.int32), tf.cast(l_img, tf.int32), tf.cast(m_img, tf.int32), 
                                 tf.cast(r_img, tf.int32), tf.cast(rr_img, tf.int32), tf.cast(render_img, tf.int32)], axis=2)

            return com_img

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]

            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns*(self.n_frames+1)]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            # List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            List_all = np.array(self.data_list)
            gt_list = List_all[:, 0]
            ll_list = List_all[:, 1]
            l_list = List_all[:, 2]
            m_list = List_all[:, 3]
            r_list = List_all[:, 4]
            rr_list = List_all[:, 5]
            coe_list = List_all[:, 6]
            render_list = List_all[:, 7]

            coes_list = []
            for i in range(len(coe_list)):
                coef = np.loadtxt(os.path.join('./training_set/', coe_list[i]))
                coef = coef[:81] # load one more data
                coef[80] = 0.0 # set the last data equals 0
                coef = coef.reshape(9,9)
                coes_list.append(coef)
            coes_list = np.array(coes_list)
            self.data_queue = tf.train.slice_input_producer([gt_list, ll_list, l_list, m_list, r_list, rr_list, coes_list, render_list], capacity=20)
            imag_in, imag_gt, coeff = read_data()
            # batch_in, batch_gt = tf.train.batch([imag_in, imag_gt], batch_size=batch_size, num_threads=8, capacity=20)
        batch_in, batch_gt, batch_coeff = tf.train.batch([imag_in, imag_gt, coeff], batch_size=batch_size, num_threads=8, capacity=20)
        return batch_in, batch_gt, batch_coeff

    def generator(self, inputs, inputs_render, coeff, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        # pre-handle coeff
        def pad_coeff(coeff_p, h_h, w_w):
            h_r = int(round((h_h-9) * 0.5))
            w_r = int(round((w_w-9) * 0.5))
            coeff_pm = tf.pad(coeff_p, [[0, 0],[h_r, h_h-9-h_r],[w_r, w_w-9-w_r]])
            coeff_pm = tf.expand_dims(coeff_pm, 3)
            coeff_pm = tf.cast(coeff_pm, tf.float32)

            return coeff_pm

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in xrange(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')

                    conv1_1_c_nums = 32
                    # add render
                    if self.args.face == 'render' or self.args.face == 'both':
                        inp_render = tf.image.resize_images(inputs_render, [hi, wi], method=0)
                        conv1_1 = tf.concat([conv1_1, inp_render], axis=3)
                        conv1_1_c_nums = 35

                    conv1_2 = ResnetBlock(conv1_1, conv1_1_c_nums, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, conv1_1_c_nums, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, conv1_1_c_nums, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    # add coeff
                    channel_nums = 128
                    if self.args.face == 'coeff' or self.args.face == 'both':
                        n_c, h_c, w_c, c_c = deconv3_4.get_shape().as_list()
                        coeff_m = pad_coeff(coeff, h_c, w_c)
                        # coeff = tf.reshape(coeff,[n_c, 81])
                        # coeff = tf.cast(coeff,tf.float32)
                        # name = 'Fc_' + str(i)
                        # print(tf.get_variable_scope().reuse)
                        # coeff_m = tf.layers.dense(inputs=coeff, units=h_c*w_c, activation=None, name=name, reuse=tf.AUTO_REUSE)
                        # coeff_m = tf.reshape(coeff_m, [n_c, h_c, w_c])
                        # coeff_m=tf.expand_dims(coeff_m,axis=3)
                        # print(coeff_m.shape)
                        deconv3_4 = tf.concat([deconv3_4, coeff_m], axis=3)
                        channel_nums = 129

                    # decoder
                    deconv3_3 = ResnetBlock(deconv3_4, channel_nums, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, channel_nums, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, channel_nums, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, conv1_1_c_nums, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, conv1_1_c_nums, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, conv1_1_c_nums, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, conv1_1_c_nums, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

                    inp_pred_temp = inp_pred
                    for x in xrange(1, self.n_frames):
                    	inp_pred = tf.concat([inp_pred, inp_pred_temp], axis=3)

            return x_unwrap

    def build_model(self):
        image_in, image_gt, coeff = self.input_producer(self.batch_size)
        img_in = image_in[:,:,:,:15]
        img_gt = image_gt[:,:,:,6:9]
        img_render = image_in[:,:,:,15:]
        print('img_in, img_gt, img_render', img_in.get_shape(), img_gt.get_shape(), img_render.get_shape())

        # the intermediate process visualization
        tf.summary.image('img_in', im2uint8(img_in[:,:,:,6:9]))
        tf.summary.image('img_gt', im2uint8(img_gt))

        # generator
        x_unwrap = self.generator(img_in, img_render, coeff, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            # medium_x_unwrap = x_unwrap[i][:,:,:,6:9]
            # loss = tf.reduce_mean((gt_i - medium_x_unwrap) ** 2)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # load checkpoints 
        checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
        self.load(sess, checkpoint_path)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps:
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path):
        input_paths = os.listdir(input_path)
        input_pathss = [os.path.join(input_path, input_p) for input_p in input_paths]

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3

        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns*self.n_frames], dtype=tf.float32)
        inputs_render = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        coeff_P = tf.placeholder(shape=[self.batch_size, 9, 9], dtype=tf.float32)
        outputs = self.generator(inputs, inputs_render, coeff_P, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir)

        def scale_match(blur, rot):
            h, w, c = blur.shape
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            return blurPad, rot, resize, h, w

        for i in range(len(input_pathss)):
            out_path = os.path.join(output_path, input_paths[i])
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            imgsName = sorted(os.listdir(input_pathss[i]+'/blur_0'))

            for imgName in imgsName:
                blur_ll = scipy.misc.imread(os.path.join(input_pathss[i]+'/blur_-2', imgName))
                blur_l = scipy.misc.imread(os.path.join(input_pathss[i]+'/blur_-1', imgName))
                blur_m = scipy.misc.imread(os.path.join(input_pathss[i]+'/blur_0', imgName))
                blur_r = scipy.misc.imread(os.path.join(input_pathss[i]+'/blur_1', imgName))
                blur_rr = scipy.misc.imread(os.path.join(input_pathss[i]+'/blur_2', imgName))
                blur = np.concatenate((blur_ll, blur_l, blur_m, blur_r, blur_rr), axis=2)

                render = scipy.misc.imread(os.path.join(input_pathss[i]+'/render', imgName))

                coeff = np.loadtxt(os.path.join(input_pathss[i]+'/face', imgName.replace('png','txt')))
                coeff = coeff[:81] # load one more data
                coeff[80] = 0.0 # set the last data equals 0
                coeff = coeff.reshape(9,9)
                coeff = np.expand_dims(coeff, axis=0)

                # make sure the width is larger than the height
                rot = False
                blurPad, rot, resize, h, w = scale_match(blur, rot)
                renderPad, rot_r, resize_r, h_r, w_r = scale_match(render, rot)
                
                start = time.time()
                deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0, inputs_render: renderPad / 255.0, coeff_P: coeff})
                duration = time.time() - start
                
                res = deblur[-1]
                if self.args.model != 'color':
                    res = np.transpose(res, (3, 1, 2, 0))
                res = im2uint8(res[0, :, :, :])
                # crop the image into original size
                if resize:
                    res = res[:new_h, :new_w, :]
                    res = scipy.misc.imresize(res, [h, w], 'bicubic')
                else:
                    res = res[:h, :w, :]

                if rot:
                    res = np.transpose(res, [1, 0, 2])
                # img_res = res[:,:,6:9]
                scipy.misc.imsave(os.path.join(out_path, imgName), res)
                print('Saving results: %s ... %4.3fs' % (os.path.join(out_path, imgName), duration))
