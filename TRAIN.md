<br>
<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

## Train commands

1. To train the model for single detection task, run the following command:
    
    ```bash
    yolo detect train data=weaponsensev2.yaml model=yolov8s.yaml pretrained=yolov8s.pt epochs=300 imgsz=640 batch=8 optimizer=SGD project=runs/DELETE/yolo8s/ name=YOLOSRCross
    ```

2. To train the model for single segmentation task, run the following command:

    ```bash
    yolo segment train data=weaponsensev2.yaml model=yolo11n-seg.yaml pretrained=yolo11n-seg.pt epochs=300 imgsz=640 batch=8 optimizer=SGD project=runs/DELETE/yolo8s/ name=YOLOSRCross
    ```

3. To train the model for multi-task detection and super-resolution, run the following command:

    ```bash
    yolo detsr train data=multi-mothsynth-weaponsensev2.yaml model=yolo-detsrv8s.yaml pretrained=yolov8s.pt epochs=300 imgsz=640 batch=16 optimizer=SGD project=runs/yolo8s/ name=YOLOSRCross
    ```
