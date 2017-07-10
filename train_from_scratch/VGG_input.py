import sys,os
import tensorflow as tf
from preprocessing import vgg_preprocessing

def read_label(label_path):
    label_dict= {}
    with open(label_path) as f:
        for line in f:
            lmap=line.strip().split('\t')
            label_dict[lmap[0]]=lmap[1] # key: ID, value: label
    return label_dict

def read_folder(image_path, image_size_x, image_size_y):
    # type: (image_path, object, object) -> object
    filename_list = tf.train.match_filenames_once(image_path)
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=1, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, channels=3)
    #image_resize = tf.random_crop(image, [image_size_x, image_size_y, 3])
    image_resize = vgg_preprocessing.preprocess_image(image, image_size_x, image_size_y, is_training=False)
    #image_resize = tf.image.resize_images(image, [image_size_x, image_size_y])
    return filename_list, image_resize

def gen_image_batch(image, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 2
    if shuffle:
        image_batch = tf.train.shuffle_batch([image],
                                             batch_size=batch_size,
                                             num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3*batch_size,
                                             min_after_dequeue=min_queue_examples)
    else:
        image_batch = tf.train.batch([image], batch_size=batch_size, num_threads=num_preprocess_threads)
                                     #capacity=min_queue_examples+3*batch_size)
    return image_batch

def generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, [label]],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 5 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, [label]],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 5 * batch_size)

  # Display the training images in the visualizer.
  #tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

