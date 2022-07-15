from ObjectDetectionSystem import ObjectDetectionSystem

o1 = ObjectDetectionSystem()
# detect_objects = o1.DetectObjects('models/research/object_detection/test_images/image2.jpg')

# to detect and save and genrate result.
detect_objects, saved = o1.DetectObjectsAndSaveImage('./images/test_images/p1.webp', "./images/detected/image2.webp")
if saved:
    print(detect_objects);