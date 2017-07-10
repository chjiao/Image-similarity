import numpy as np
import pdb
import pickle,random
from tensorflow.python.framework import ops
from nets import vgg
from VGG_input import *

slim = tf.contrib.slim
image_size = vgg.vgg_16.default_image_size
batch_size = 100
class_number = 200


#---------------------- read images and generate training data -------------------------#
label_file = '/home/j0c023d/visual_search/Data/tiny-imagenet-200/words.txt'
train_image_folder='/home/j0c023d/visual_search/Data/tiny-imagenet-200/train/'
checkpoints_dir = '/home/j0c023d/visual_search/miniImagenet/train_checkpoints/'
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
        label_full_list.extend(image_names)
        image_list = [os.path.join(image_path, x) for x in image_names]
        image_full_list.extend(image_list)
        #label_full_list.extend([class_label] * len(image_list))
        class_list.append(sub_folder)
        class_label += 1
        #print(sub_folder)
    
    #label_full_list = image_full_list[:]
    total_size = len(image_full_list)
    all_images = ops.convert_to_tensor(image_full_list, dtype=tf.string)
    all_labels = ops.convert_to_tensor(label_full_list, dtype=tf.string)

    # create input queues
    train_input_queue = tf.train.slice_input_producer([all_images, all_labels], capacity = 6*batch_size)

    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = vgg_preprocessing.preprocess_image(train_image, image_size, image_size, is_training=True)
    train_label = train_input_queue[1]
    train_image_batch, train_label_batch = generate_image_and_label_batch(train_image, train_label, 2, batch_size, False)

    #-------------------------- load the pre-trained model ------------------------------------#
    predictions, end_points = vgg.vgg_16(train_image_batch, num_classes=class_number)
    fc7 = end_points['vgg_16/fc7']
    fc7 = tf.squeeze(fc7, [1,2])
    fc_norm = tf.norm(fc7, ord=2, axis=1) # dim: batch_size
    fc_norm = tf.expand_dims(fc_norm, 1) # dimension: batch_size x 1
    fc7 = tf.div(fc7, fc_norm)

    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    image_names_list=[]  # train images names, containing the class label
    feature_matrix = np.zeros((1,4096), dtype=np.float32)
    with tf.Session() as sess:
        sess.run(init)
        #saver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, 'VGG_miniImagenet-18000.meta'))
        saver.restore(sess, os.path.join(checkpoints_dir, 'VGG_miniImagenet-500000'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        print(int(total_size/batch_size))
        for i in range(int(total_size/batch_size)):
            print(i)
            image_list, fc7_out = sess.run([train_label_batch, fc7])
            image_names_list.extend(image_list)
            #pdb.set_trace()
            feature_matrix=np.concatenate((feature_matrix, fc7_out), axis=0)
        
        coord.request_stop()
        coord.join(threads)

    feature_matrix = feature_matrix[1:]
    results = (image_names_list, feature_matrix)
    pickle.dump(results, open('Train_results_scratch', 'wb'))



