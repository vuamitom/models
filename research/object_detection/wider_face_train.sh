# From the tensorflow/models/research/ directory
# PIPELINE_CONFIG_PATH=./wider_face_models/ssd224/ssd_mobilenet_v2_quantized_224x224_widerface.config
# MODEL_DIR=./wider_face_models/ssd224

PIPELINE_CONFIG_PATH=./wider_face_models/ssdcust/ssdcust_mobilenet_v2_quantized_224x224_widerface.config
MODEL_DIR=./wider_face_models/ssdcust
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr