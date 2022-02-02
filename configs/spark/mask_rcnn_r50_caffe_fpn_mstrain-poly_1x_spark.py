# The new config inherits a base config to highlight the necessary modification
_base_ = '/home/sstc/PycharmProjects/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=11),
        mask_head=dict(num_classes=11)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ("AcrimSat",
    "Aquarius",
    "Aura",
    "Calipso",
    "Cloudsat",
    "CubeSat",
    "Debris",
    "Jason",
    "Sentinel-6",
    "Terra",
    "TRMM")
data = dict(
    train=dict(
        img_prefix='/home/sstc/PycharmProjects/tph-yolov5/spark_dataset/images/train',
        classes=classes,
        ann_file='/home/sstc/PycharmProjects/tph-yolov5/spark_dataset/labels/train.json'),
    val=dict(
        img_prefix='/home/sstc/PycharmProjects/tph-yolov5/spark_dataset/images/val',
        classes=classes,
        ann_file='/home/sstc/PycharmProjects/tph-yolov5/spark_dataset/labels/val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'