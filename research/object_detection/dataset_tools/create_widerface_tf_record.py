import os
import sys
import io
import hashlib
# import logging
import tensorflow as tf
import PIL
import numpy as np
from object_detection.utils import label_map_util
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
import contextlib2

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw face dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/face_label_map.pbtxt',
                    'Path to label map proto')
# flags.DEFINE_integer('num_shards', 50, 'Number of TFRecord shards')

FLAGS = flags.FLAGS
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def create_single_tf_example(faces, img_name, label_map_dict, base_dir):
    img_path = os.path.join(base_dir, img_name)
    # print ('">>', img_path, '<<"')
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if not image.format == 'JPEG':
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
    # print ('faces ', faces)
    for obj in faces:
        x, y, w, h = obj[0], obj[1], obj[2], obj[3]
        blur, invalid = obj[4], obj[7]
        if (not invalid == 1) and w >= 50 and h >= 50:
            xmins.append(max(0.005, float(x) / width))
            xmaxs.append(min(0.995, (float(x) + w) / width))
            ymins.append(max(0.005, float(y) / height))
            ymaxs.append(min(0.995, (float(y) + h) / height))
            classes_text.append('face'.encode('utf8'))
            classes.append(label_map_dict['face'])
    if len(classes) == 0:
        return None

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
    valid_imgs = 0
    max_faces = 0
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            imgs.append(line.strip(' \n'))
            n_obj = int(f.readline().strip(' \n'))
            # print('read n_obj', n_obj)
            if n_obj == 0:
                n_obj = 1
            # 
            faces = [[int(t) for t in f.readline().strip(' \n').split(' ')]
                        for o in range(0, n_obj)]
            # valid_count = sum([1 for f in faces if f[2]>=50 and f[3]>=50])
            # max_faces = valid_count if valid_count > max_faces else max_faces
            objects.append(faces)
            line = f.readline()            
    print('about to prepare', len(imgs), ' records')
    # if True:
    #     print ('max faces = ', max_faces)
    #     return

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
        for idx, img_path in enumerate(imgs):
            if idx % 100 == 0:
                print('On image', idx, 'of',len(imgs))
            try:                
                tf_example = create_single_tf_example(objects[idx], img_path, 
                                                        label_map_dict, image_dir)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                    valid_imgs += 1
                
            except ValueError:
                print('Invalid example: %s, ignored', img_path)    
    print ('no. valid images = ', valid_imgs)


def main(_):
    data_dir = FLAGS.data_dir
    
    train_output_path = os.path.join(FLAGS.output_dir, 'wider_faces_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'wider_faces_val.record')
    train_image_dir = os.path.join(data_dir, 'WIDER_train', 'images')
    val_image_dir = os.path.join(data_dir, 'WIDER_val', 'images')
    train_image_txt = os.path.join(data_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    val_image_txt = os.path.join(data_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    print('Reading from WILDER face dataset.')
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    # create train set
    create_tf_record(
        train_output_path,
        train_image_txt,
        100,
        label_map_dict,
        train_image_dir)
    # create validation set
    create_tf_record(
        val_output_path,
        val_image_txt,
        10,
        label_map_dict,
        val_image_dir)

if __name__ == '__main__':
    tf.app.run()