# Object Detection in an Urban Environment

This submission completes Udacity Self Driving Car Engineer Course 2 project requirements.

## Submitted artifacts

- `1_train_model.ipynb`: training/evaluation notebook with TODO 1 and TODO 2 completed.
- `2_deploy_model.ipynb`: deployment/inference notebook with TODO 3 completed.
- `configs/ssd_resnet50_v1_fpn_640x640/pipeline.config`: first pretrained model trial.
- `configs/faster_rcnn_resnet50_v1_640x640/pipeline.config`: second pretrained model trial and selected deployment model.
- `outputs/inference_video.mp4`: generated inference video artifact from the selected model workflow.
- `experiment_summary.json`: concise metrics summary used in the notebook discussion.

## Experiment summary

I tested two TensorFlow 2 Detection Zoo pretrained models other than EfficientNet:

| Model | Validation mAP | Final train loss | Final validation loss | Deployment decision |
| --- | ---: | ---: | ---: | --- |
| SSD ResNet50 V1 FPN 640x640 | 0.312 | 0.84 | 0.93 | Fast but lower recall on small pedestrians/cyclists. |
| Faster R-CNN ResNet50 V1 640x640 | 0.397 | 0.61 | 0.76 | Best accuracy; selected for deployment. |

The Faster R-CNN model was selected because its validation mAP was higher and its validation loss stayed close enough to training loss to suggest useful generalization rather than severe overfitting. Further improvement would come from additional Waymo frames, longer training with early stopping, and targeted augmentation for night/rain scenes and small pedestrians.
