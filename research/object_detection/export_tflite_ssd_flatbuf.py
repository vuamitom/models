import tensorflow as tf
import subprocess
import os

def regen_frozen_graph(checkpoint_dir, pipeline='pipeline.config', ckpt='model.ckpt'):
    config_file = os.path.join(checkpoint_dir, pipeline)
    ckpt_path = os.path.join(checkpoint_dir, ckpt)
    cmds = [
        'python',
        'export_tflite_ssd_graph.py',
        '--pipeline_config_path=' + config_file,
        '--trained_checkpoint_prefix=' + ckpt_path,
        '--output_directory=.',
        '--add_postprocessing_op=true',
        '--max_detections=10'
    ]
    subprocess.call(cmds)


def gen_tflite(crop_size):
    graph_def_file = "/home/tamvm/Projects/tensorflow-models/research/object_detection/tflite_graph.pb"
    # graph_def_file = "/home/tamvm/Downloads/facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb"
    input_arrays = ["normalized_input_image_tensor"]
    # output_arrays = ["detection_boxes", "detection_classes", "detection_scores", 'num_detections']
    # output_arrays = ['raw_outputs/box_encodings', 'raw_outputs/class_predictions']
    output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
    input_shapes={}
    input_shapes[input_arrays[0]] = [1, crop_size, crop_size, 3]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays, input_shapes=input_shapes)
    converter.inference_type = tf.uint8
    converter.allow_custom_ops = True
    converter.quantized_input_stats = {}
    converter.quantized_input_stats[input_arrays[0]] = (128, 128) # (mean, std)
    # converter.post_training_quantize = True
    tflite_model = converter.convert()
    open('face' + str(crop_size) + 'x' + str(crop_size) + '.tflite', "wb").write(tflite_model)

def export_from_session():
    pass
# def gen_tflite_toco():
#     cmds = [
#         'bazel',
#         'run',
#         '--config=opt tensorflow/lite/toco:toco --'
#     ]
#     subprocess.call(cmds, cwd=tensorflow_dir)

if __name__ == '__main__':
    crop_size = 48
    regen_frozen_graph('/home/tamvm/Projects/tensorflow-models/research/object_detection/wider_face_models/ssdlite48_3anchors',
                        pipeline='ssdlite_mobilenet_v2_quantized_48x48_3anchors_widerface.config',
                        ckpt='models_ssdlite48_3anchors_model.ckpt-542296')
    gen_tflite(crop_size)