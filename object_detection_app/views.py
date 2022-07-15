from ast import If
from django.shortcuts import render, HttpResponse
from matplotlib.style import context
from .object_detection.ObjectDetectionSystem import ObjectDetectionSystem as ObjectDetectionSystem
from django.core.files.storage import FileSystemStorage
import os
import  time
from pathlib import Path


ALLOWED_FILE_FORMAT = [".jpg", ".png", ".jpeg", ".webp"]
# Create your views here.
def index(request):
    delete_uploaded_file()
    return render(request, 'index.html')


def upload_and_test(request):
    dir_path = os.path.join(Path(__file__).resolve().parent.parent,'static/uploaded')
    delete_uploaded_file()
    message = ""
    detection_result = ""
    show_img = ""
    if request.method == "POST":
        uploaded_file = request.FILES['image']
        split_filename = os.path.splitext(uploaded_file.name)
        file_ext = split_filename[1]
        if file_ext in ALLOWED_FILE_FORMAT:
            milliseconds = int(round(time.time() * 1000))
            new_file_name = str(milliseconds)+file_ext
            fs = FileSystemStorage()
            fs.save(new_file_name, uploaded_file)
            imagepath = os.path.join(dir_path, new_file_name)
            detection_result, save_status = detect_object_and_save_image(imagepath)
            if save_status:
                show_img = new_file_name
        else:
            message = "This file is not acceptable"
    context = {
        "message": message,
        "detection_result": detection_result,
        "show_img": show_img
    }
    return render(request, 'upload_and_test.html', context)


def detect_object_and_save_image(imagepath):
    o1 = ObjectDetectionSystem()
    LoadJson = True
    detect_objects, saved = o1.DetectObjectsAndSaveImage(imagepath, imagepath, LoadJson)
    return detect_objects, saved

def delete_uploaded_file():
    dir_path = os.path.join(Path(__file__).resolve().parent.parent,'static/uploaded')
    files = os.listdir(dir_path)
    for file in files:
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, file)):
            split_filename = os.path.splitext(file)
            file_time = int(split_filename[0]) + 200000
            currenttime = int(round(time.time() * 1000))
            if currenttime >= file_time:
                print(currenttime, file_time)
                os.remove(os.path.join(dir_path, file))
