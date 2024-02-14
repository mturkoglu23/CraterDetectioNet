import json
import os
from ultralytics import RTDETR, YOLO

from build_pred_coco import build_pred_coco
from three_ensembler import three_ensemble
from test_models import test_models
from validate_coco import validate_coco

if __name__ == '__main__':
    dataset_name = "gomulu966_new"
    yolov8_model = YOLO(f'RESULTS/{dataset_name}/yolo/train/weights/best.pt')
    yolov5_model = YOLO(f'RESULTS/{dataset_name}/yolov5/train/weights/best.pt')
    rtdetr_model = RTDETR(f'RESULTS/{dataset_name}/rtdetr/train/weights/best.pt')
    yolov8_test_coco = test_models(f"RESULTS/{dataset_name}/datasets/test.json", f"RESULTS/{dataset_name}/datasets/test/images", yolov8_model)
    yolov5_test_coco = test_models(f"RESULTS/{dataset_name}/datasets/test.json", f"RESULTS/{dataset_name}/datasets/test/images", yolov5_model)
    rtdetr_test_coco = test_models(f"RESULTS/{dataset_name}/datasets/test.json", f"RESULTS/{dataset_name}/datasets/test/images", rtdetr_model)
    three_ensemble_test_coco = three_ensemble(f"RESULTS/{dataset_name}/datasets/test.json", f"RESULTS/{dataset_name}/datasets/test/images",
                                         rtdetr_model, yolov8_model, yolov5_model)
    yolov8_test_json_data = json.dumps(yolov8_test_coco)
    yolov5_test_json_data = json.dumps(yolov5_test_coco)
    rtdetr_test_json_data = json.dumps(rtdetr_test_coco)
    three_ensemble_test_json_data = json.dumps(three_ensemble_test_coco)
    """
    def convert_to_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value


    # old_ensemble_test_coco içindeki her bir elemanın içindeki tüm sayısal değerleri float türüne dönüştür
    for item in old_ensemble_test_coco:
        for key, value in item.items():
            item[key] = convert_to_float(value)

    # Şimdi old_ensemble_test_coco değişkenini JSON olarak seri hale getirebilirsiniz
    old_ensemble_test_json_data = json.dumps(old_ensemble_test_coco)
    """
    if os.path.exists('yolov8_test.coco.json'):
        os.remove('yolov8_test.coco.json')
    if os.path.exists('yolov5_test.coco.json'):
        os.remove('yolov5_test.coco.json')
    if os.path.exists('rtdetr_test.coco.json'):
        os.remove('rtdetr_test.coco.json')
    if os.path.exists('three_ensemble_test.coco.json'):
        os.remove('three_ensemble_test.coco.json')
    with open('yolov8_test.coco.json', 'w') as json_file:
        json_file.write(yolov8_test_json_data)
    with open('yolov5_test.coco.json', 'w') as json_file:
        json_file.write(yolov5_test_json_data)
    with open('rtdetr_test.coco.json', 'w') as json_file:
        json_file.write(rtdetr_test_json_data)
    with open('three_ensemble_test.coco.json', 'w') as json_file:
        json_file.write(three_ensemble_test_json_data)

    print("YOLOV8 TEST RESULTS")
    yolov8_test_results = validate_coco(f'RESULTS/{dataset_name}/datasets/test.json', "yolov8_test.coco.json")
    print("YOLOV5 TEST RESULTS")
    yolov5_test_results = validate_coco(f'RESULTS/{dataset_name}/datasets/test.json', "yolov5_test.coco.json")
    print("RTDETR TEST RESULTS")
    rtdetr_test_results = validate_coco(f'RESULTS/{dataset_name}/datasets/test.json', "rtdetr_test.coco.json")
    print("THREE ENSEMBLE TEST RESULTS")
    old_ensemble_test_results = validate_coco(f'RESULTS/{dataset_name}/datasets/test.json', "three_ensemble_test.coco.json")
