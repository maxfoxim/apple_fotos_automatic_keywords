# https://lopezyse.medium.com/computer-vision-object-detection-with-python-14b241f97fd8
#  .venv/bin/python Main.py

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

import os.path
import photoscript
import osxphotos

def detect_objects(image_path,confidence_value):
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Load the model
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(image_rgb)[0]
    boxes = results.boxes  
    class_names = results.names
    
    tags = []
    for box in boxes:
        confidence = float(box.conf[0])
        
        if confidence > confidence_value:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            tags.append(class_name)
        
    return tags

db = os.path.expanduser("/Users/stephan/Pictures/Fotomediathek_ab_2025.photoslibrary")
photosdb = osxphotos.PhotosDB(db)

# Ausgabe infos
print(photosdb.keywords)
print(photosdb.persons)
print(photosdb.albums)

print(photosdb.keywords_as_dict)
print(photosdb.persons_as_dict)
print(photosdb.albums_as_dict)

# filter photo
photos = photosdb.photos()

for p in photos[:]:
    print("------------------------------")
    print(
        p.uuid,
        p.filename,
        p.keywords,
        p.path,
    )
    if p.path is not None:
        try:
            keywords = detect_objects(p.path,0.1)
            print(keywords)
            p = photoscript.Photo(p.uuid)
            
            p.keywords = p.keywords + keywords 
        except Exception as e:
            print("Problem: ",e)
            


    