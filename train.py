from ultralytics import YOLO
import os

model = YOLO("yolov8n.yaml")  


results = model.train(data="data\data.yaml", epochs=100,)

