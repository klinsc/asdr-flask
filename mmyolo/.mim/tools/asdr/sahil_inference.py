# import required functions, classes
# mmdet requirements
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.mmdet import (
    download_mmdet_cascade_mask_rcnn_model,
    download_mmdet_config,
)

# download cascade mask rcnn model&config
model_path = (
    "models/yolov5_s-p6-v62_syncbn_fast_4xb8-130e_asdr6-2-200-split82_1280_anchOptm.pth"
)
config_path = "configs/yolov5/yolov5_s-p6-v62_syncbn_fast_4xb8-150e_asdr6-2-200-split82_1280_anchOptm.py"


detection_model = AutoDetectionModel.from_pretrained(
    model_type="mmdet",
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.5,
    image_size=640,
    device="cpu",  # 'cpu' or 'cuda:0'
)
