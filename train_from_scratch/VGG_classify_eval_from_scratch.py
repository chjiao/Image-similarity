import numpy as np
import pdb
import pickle,random
from tensorflow.python.framework import ops

from nets import vgg

from VGG_input import *

slim = tf.contrib.slim

def eval_images(image_batch, label_batch, num_class):
    predictions, end_points = vgg.vgg_16(image_batch, num_classes=num_class)
    probabilities = tf.nn.softmax(predictions)
    predict_labels = tf.argmax(probabilities, axis=1)
    real_labels = tf.argmax(label_batch, axis=1)
    return predict_labels, real_labels

image_size = vgg.vgg_16.default_image_size
batch_size = 50
class_number = 200


#---------------------- read images and generate training data -------------------------#
label_file = '/home/j0c023d/visual_search/Data/tiny-imagenet-200/val/val_annotations.txt'
train_image_folder='/home/j0c023d/visual_search/Data/tiny-imagenet-200/train/'
eval_image_folder='/home/j0c023d/visual_search/Data/tiny-imagenet-200/val/images/'
label_dict = read_label(label_file)

image_full_list = os.listdir(eval_image_folder)
eval_size = len(image_full_list)
classes = os.listdir(train_image_folder)
class_list = []
class_dict = {}

class_label = 0
for sub_folder in classes:
    if sub_folder.startswith('.'):
        continue
    class_list.append(sub_folder)
    class_dict[sub_folder] = class_label # key: class name, value: class number
    class_label += 1

label_full_list = [class_dict[label_dict[x]] for x in image_full_list]
image_full_list = [os.path.join(eval_image_folder, x) for x in image_full_list]
#pdb.set_trace()

with tf.Graph().as_default():
    all_images = ops.convert_to_tensor(image_full_list, dtype=tf.string)
    all_labels = ops.convert_to_tensor(label_full_list, dtype=tf.int32)

    # create input queues
    eval_input_queue = tf.train.slice_input_producer([all_images, all_labels], capacity = 6*batch_size)

    file_content = tf.read_file(eval_input_queue[0])
    eval_image = tf.image.decode_jpeg(file_content, channels=3)
    eval_image = vgg_preprocessing.preprocess_image(eval_image, image_size, image_size, is_training=False)
    eval_label = eval_input_queue[1]
    eval_image_batch, eval_label_batch = generate_image_and_label_batch(eval_image, eval_label, 2, batch_size,False)
    eval_label_batch = tf.one_hot(eval_label_batch, class_number)

    #-------------------------- create the computation graph ------------------------------------#
    checkpoints_dir = '/home/j0c023d/visual_search/miniImagenet/train_checkpoints/'
    
    # test
    predict_labels, real_labels = eval_images(eval_image_batch, eval_label_batch, class_number)
    l1=tf.placeholder(shape=(None), dtype=tf.float32)
    l2=tf.placeholder(shape=(None), dtype=tf.float32)
    test_correct_prediction = tf.equal(l1, l2)
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        #saver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, 'VGG_miniImagenet-49900.meta'))
        saver.restore(sess, os.path.join(checkpoints_dir, 'VGG_miniImagenet-500000'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        """
        for i in range(100001):
            if i%10==0:
                summary, train_accuracy, loss_r, _ = sess.run([merged, accuracy, loss, train_step])
                train_writer.add_summary(summary, i)
                print("Step: %d, train loss is: %f, train accuracy is %f" % (i, loss_r, train_accuracy))
            else:
                summary,_ = sess.run([merged,train_step])
                train_writer.add_summary(summary, i)
            if i%100==0 and i>0:
                saver.save(sess, train_log_dir+'VGG_miniImagenet', global_step=i)
        """
        #train_writer.close()

        # test the model
        test_labels = np.array([])
        test_predict_labels = np.array([])

        for j in range(int(eval_size/batch_size)):
            print(j)
            predict_labels_r, real_labels_r = sess.run([predict_labels, real_labels]) 
            test_predict_labels = np.concatenate((test_predict_labels, predict_labels_r))
            test_labels = np.concatenate((test_labels, real_labels_r))
            #pdb.set_trace()
        #pdb.set_trace()
        test_accuracy_r = sess.run(test_accuracy, feed_dict={l1:test_predict_labels, l2:test_labels})
        print("The evaluation accuracy is: %f" % (test_accuracy_r))
            
        
    coord.request_stop()
    coord.join(threads)

