DATA_DIR='/home/tamvm/Projects/tensorflow-models/research/object_detection'
OUTPUT_DIR='/home/tamvm/Projects/tensorflow-models/research/object_detection/wider_face_tfrecords'
LABEL_MAP='/home/tamvm/Projects/tensorflow-models/research/object_detection/data/face_label_map.pbtxt'
python dataset_tools/create_widerface_tf_record.py \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR}\
    --label_map_path=${LABEL_MAP}