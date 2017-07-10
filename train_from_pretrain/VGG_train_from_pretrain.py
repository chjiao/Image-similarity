import numpy as np
import pdb
import pickle,random
from tensorflow.python.framework import ops

from nets import vgg

from VGG_input import *

slim = tf.contrib.slim
image_size = vgg.vgg_16.default_image_size
batch_size = 50
class_number = 200


#---------------------- read images and generate training data -------------------------#
label_file = '/home/j0c023d/visual_search/Data/tiny-imagenet-200/words.txt'
train_image_folder='/home/j0c023d/visual_search/Data/tiny-imagenet-200/train/'
checkpoints_dir = '/home/j0c023d/visual_search/Trained_models/VGG/'
label_dict = read_label(label_file)

with tf.Graph().as_default():
    image_full_list, label_full_list = [], []
    classes = os.listdir(train_image_folder)
    class_list = []

    class_label = 0
    for sub_folder in classes:
        if sub_folder.startswith('.'):
            continue
        image_path = os.path.join(train_image_folder, sub_folder) + '/images/'
        image_names = os.listdir(image_path)
        image_list = [os.path.join(image_path, x) for x in image_names]
        image_full_list.extend(image_list)
        label_full_list.extend([class_label] * len(image_list))
        class_list.append(sub_folder)
        class_label += 1
        #print(sub_folder)

    all_images = ops.convert_to_tensor(image_full_list, dtype=tf.string)
    all_labels = ops.convert_to_tensor(label_full_list, dtype=tf.int32)

    ## partition the test data set as 10% of all the data
    total_size = len(image_full_list)
    test_size = int(0.1*total_size)
    partitions = [0]*total_size
    partitions[:test_size] = [1]*test_size
    random.shuffle(partitions)

    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    # create input queues
    train_input_queue = tf.train.slice_input_producer([train_images, train_labels], capacity = 6*batch_size)
    test_input_queue = tf.train.slice_input_producer([test_images, test_labels], num_epochs=1, shuffle=False, capacity = 6*batch_size)

    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = vgg_preprocessing.preprocess_image(train_image, image_size, image_size, is_training=True)
    train_label = train_input_queue[1]
    train_image_batch, train_label_batch = generate_image_and_label_batch(train_image, train_label, 2, batch_size, False)
    train_label_batch = tf.one_hot(train_label_batch, class_number)
    #pdb.set_trace()
    #"""
    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content,channels=3)
    test_image = vgg_preprocessing.preprocess_image(test_image, image_size, image_size, is_training=False)
    test_label = test_input_queue[1]
    test_image_batch, test_label_batch = generate_image_and_label_batch(test_image, test_label, 2, batch_size, False)
    test_label_batch = tf.one_hot(test_label_batch, class_number)
    #"""

    #-------------------------- load the pre-trained model ------------------------------------#
    train_log_dir = './train_from_pretrain_checkpoints/'
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)
    
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(train_image_batch)
    net = end_points['vgg_16/fc7']
    net = slim.dropout(net, 0.5, is_training=True, scope='dropout7')
    net = slim.conv2d(net, class_number, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='fc8')
    logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')

    probabilities = tf.nn.softmax(logits)
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'), variables_to_restore)

    loss = tf.losses.softmax_cross_entropy(train_label_batch, logits)

    #cross_entropy = tf.reduce_mean(loss)

    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(probabilities, axis=1), tf.argmax(train_label_batch, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for i in range(20000):
            if i%10==0:
                train_accuracy, loss_r, _ = sess.run([accuracy, loss, train_step])
                print("Step: %d, train loss is: %f, train accuracy is %f" % (i, loss_r, train_accuracy))
            else:
                sess.run(train_step)
                #probabilities_r, train_label_batch_r, train_accuracy, loss_r, _ = sess.run([probabilities, train_label_batch, accuracy, loss, train_step])
                #print(train_accuracy)
                #pdb.set_trace()

            if i%100==0 and i>0:
                saver.save(sess, train_log_dir+'VGG_miniImagenet', global_step=i)
                
            #train_image_batch_r, train_label_batch_r = sess.run([train_image_batch, train_label_batch])
            #pdb.set_trace()
            #train_input_queue_r = sess.run(train_input_queue)
            #probabilities_r, predictions_r = sess.run([probabilities, predictions])
            #probabilities_r = sess.run(probabilities)
            #pdb.set_trace()

            #loss_r = sess.run(loss)
            #train_accuracy = sess.run(accuracy)
            #pdb.set_trace()

    # test the model
        
    coord.request_stop()
    coord.join(threads)




