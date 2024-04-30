import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, prototxt_path, caffemodel_path):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def detect_objects(self, color_image_bgr, CLASSES, object_of_interest):
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        object_center = None
        object_point = None

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == object_of_interest:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [color_image_bgr.shape[1], color_image_bgr.shape[0], color_image_bgr.shape[1],
                         color_image_bgr.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    object_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
                    object_point = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
                    break

        return object_center, object_point
