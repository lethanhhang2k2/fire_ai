#nhap thu vien

import time #tgian
import cv2 #thu vien xu ly hinh anh
import argparse
import numpy as np #thu vien xu ly hinh anh




def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


#ham danh label va ve duong bao quanh vat

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = (0,0,255)
    if label =='Fire':
        print(" CO LUAAAAAAAAAAA")
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    
cap = cv2.VideoCapture('assets/video/fire1.mp4')




classes = None

# nhap ten vat
with open("ml/objects.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# load tri tue nhan tao
print("load modelll")

net = cv2.dnn.readNet("ml/yolov3.weights","ml/yolov3.cfg")



class_ids = []
confidences = []
boxes = []
conf_threshold = 0.1
nms_threshold = 0.1

#xu ly anh

while(True):
    #doc anh tu cam

    #tien xu ly
    ret, image = cap.read()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    #xu li tung pixel
    outs = net.forward(get_output_layers(net))
    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    #quet tung diem anh va tim vat
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #so sanh do chinh xac va ve duong bao quanh vat
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    img_resized = cv2.resize(src=image, dsize=(1200, 800))
    cv2.imshow("object detection", img_resized)
    end = time.time()
    print("YOLO Execution time: " + str(end-start))
    if cv2.waitKey(1)== 32:
        break

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
