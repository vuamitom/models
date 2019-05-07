import logging
import tensorflow as tf
import PIL
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

def create_single_tf_example(faces, img_name, label_map_dict, base_dir):
    img_path = os.path.join(base_dir, img_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format not == 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    image = np.asarray(image)
    width = int(image.shape[1])
    height = int(image.shape[0])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    for obj in faces:
        x, y, w, h = obj[0], obj[1], obj[2], obj[3]
        xmins.append(float(x) / width)
        xmaxs.append((float(x) + w) / width)
        ymins.append(float(y) / height)
        ymaxs.append((float(y) + h) / height)
        classes_text.append('face')
        classes.append(label_map_dict['face'])


    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
          img_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
          img_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
        # 'image/object/view': dataset_util.bytes_list_feature(poses),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_filename, txt_path, 
                    num_shards, 
                    label_map_dict,
                    image_dir):
    objects, imgs = [], []
    with open(txt_path, 'r') as f:
        imgs.append(f.readline())
        n_obj = int(f.readline())
        objects.append([(int(t) for t in f.readline().split(' '))
                            for o in range(0, n_obj)])

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
        for idx, img_path in enumerate(imgs):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(imgs))
            try:
                tf_example = create_single_tf_example(objects[idx], img_path, 
                                                        label_map_dict, image_dir)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignored', img_path)    


def main(_):
    logging.info('Reading from WILDER face dataset.')
    label_map_dict = dict(0='background', 1='face')
    # create train set
    create_tf_record(
        '',
        '',
        FLAGS.num_shards,
        label_map_dict,
        '')
    # create validation set
    create_tf_record(
        '',
        '',
        FLAGS.num_shards,
        label_map_dict,
        '')

if __name__ == '__main__':
    tf.app.run()