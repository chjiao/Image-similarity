import numpy as np
import pdb
import pickle
from tensorflow.python.framework import ops
from nets import vgg
from VGG_input import *

def k_nearest_neighbor(query, features):
    # query: the embedding of query image
    # features: the embeddings of training images
    z = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(query, features)), axis=1))
    return z

def partition(vector, left, right, pivotIndex):
    pivotValue = vector[pivotIndex]
    vector[pivotIndex], vector[right] = vector[right], vector[pivotIndex]
    storeIndex = left
    for i in range(left, right):
        if vector[i]<pivotValue:
            vector[storeIndex], vector[i] = vector[i], vector[storeIndex]
            storeIndex+=1
    vector[right], vector[storeIndex] = vector[storeIndex], vector[right]
    return storeIndex

def _select(vector, left, right, k):
    "Returns the k-th smallest, (k>=0), element of vector within vector[left:right+1] inclusive"
    while True:
        pivotIndex = np.random.randint(left, right)
        pivotNewIndex = partition(vector, left, right, pivotIndex)
        pivotDist = pivotNewIndex - left
        if pivotDist == k:
            return pivotNewIndex, vector[pivotNewIndex]
        elif k<pivotDist:
            right = pivotNewIndex - 1
        else:
            k -= pivotDist +1
            left = pivotNewIndex +1

def select(vector, k, left=None, right=None):
    if left is None:
        left = 0
    lv1 = len(vector) - 1
    if right is None:
        right = lv1
    assert vector and k>=0
    assert 0<=left<=lv1
    assert left<=right<=lv1
    return _select(vector, left, right, k)

slim = tf.contrib.slim
image_size = vgg.vgg_16.default_image_size
batch_size = 50
class_number = 200

#---------------------- read images -------------------------#
val_label_file = "/home/j0c023d/visual_search/Data/tiny-imagenet-200/val/val_annotations.txt"
val_label_dict = read_label(val_label_file)

## load training data matrix
class_list, feature_matrix = pickle.load(open("/home/j0c023d/visual_search/miniImagenet/Train_results", "rb"))

eval_image_folder='/home/j0c023d/visual_search/Data/tiny-imagenet-200/val/images/'

image_full_list = os.listdir(eval_image_folder)
label_full_list = image_full_list[:]
image_full_list = [os.path.join(eval_image_folder, x) for x in image_full_list]
eval_size = len(image_full_list) # image number in the folder

f_out = open('Visual_search_result.txt', 'w')

with tf.Graph().as_default():
    all_images = ops.convert_to_tensor(image_full_list, dtype=tf.string)
    all_labels = ops.convert_to_tensor(label_full_list, dtype=tf.string)

    # create input queues
    eval_input_queue = tf.train.slice_input_producer([all_images, all_labels], capacity = 6*batch_size, shuffle=False)

    file_content = tf.read_file(eval_input_queue[0])
    eval_image = tf.image.decode_jpeg(file_content, channels=3)
    eval_image = vgg_preprocessing.preprocess_image(eval_image, image_size, image_size, is_training=False)
    eval_label = eval_input_queue[1]
    eval_image_batch, eval_label_batch = generate_image_and_label_batch(eval_image, eval_label, 2, batch_size,False)
    #eval_label_batch = tf.one_hot(eval_label_batch, class_number)

    #-------------------------- create the computation graph ------------------------------------#
    checkpoints_dir = '/home/j0c023d/visual_search/miniImagenet/train_from_pretrain_checkpoints/'
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(eval_image_batch, is_training=False)
    fc7 = end_points['vgg_16/fc7']
    net = slim.dropout(fc7, 0.5, is_training=True, scope='dropout7')
    net = slim.conv2d(net, class_number, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='fc8')
    logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    fc7 = tf.squeeze(fc7, [1, 2])
    fc_norm = tf.norm(fc7, ord=2, axis=1) # dim: batch_size
    fc_norm = tf.expand_dims(fc_norm, 1) # dimension: batch_size x 1
    fc7=tf.div(fc7, fc_norm)
    probabilities = tf.nn.softmax(logits)
    
    saver=tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, os.path.join(checkpoints_dir, 'VGG_miniImagenet-500000'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        for i in range(int(eval_size/batch_size)):
            print(i)
            image_names, fc7_out = sess.run([eval_label_batch, fc7])
            image_names = [x.decode('utf-8') for x in image_names]
            eval_class_list = [val_label_dict[x] for x in image_names]

            x = tf.placeholder(tf.float32, shape=(4096))
            y = tf.placeholder(tf.float32, shape=[None, 4096])
            distances = k_nearest_neighbor(x, y)
            for j in range(batch_size):
                image_vector = fc7_out[j]
                distances_r = sess.run(distances, feed_dict={x:image_vector, y:feature_matrix})
                idx_sort = np.argsort(distances_r)
                #pdb.set_trace()
                
                top10_neighbors = [class_list[x].decode('utf-8') for x in idx_sort[:10]]
                f_out.write(image_names[j]+'\t'+eval_class_list[j]+':\t'+"\t".join(top10_neighbors)+'\n')
            #pdb.set_trace()
        coord.request_stop()
        coord.join(threads)

f_out.close()

# generate the correct results counting
f1 = open('Visual_search_count.txt','w')
with open('Visual_search_result.txt', 'r') as f:
    for line in f:
        lmap = line.split('\t')
        eval_class = lmap[1][:-1]
        count = 0
        for i in range(2,len(lmap)):
            image_class = lmap[i].split('_')[0]
            if image_class==eval_class:
                count+=1
        f1.write(lmap[0]+'\t'+lmap[1]+'\t'+str(count)+'\n')
f1.close()

