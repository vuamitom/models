INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/home/tamvm/Downloads/facessd_mobilenet_v2_quantized_320x320_open_image_v4/pipeline.config'
TRAINED_CKPT_PREFIX='/home/tamvm/Downloads/facessd_mobilenet_v2_quantized_320x320_open_image_v4/model.ckpt'
EXPORT_DIR=.
python export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}