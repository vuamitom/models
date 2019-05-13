YOUR_GCS_BUCKET='tamvm-face'
gcloud ml-engine jobs submit training `whoami`_object_detection_wider_faces_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir=gs://${YOUR_GCS_BUCKET}/models/ssd224 \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region asia-east1 \
    --config object_detection/samples/cloud/cloud.yml \
    -- \
    --model_dir=gs://${YOUR_GCS_BUCKET}/models/ssd224 \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v2_quantized_224x224_widerface.config