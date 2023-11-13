import cv2
import numpy as np
import argparse
import time
import datetime
import os

model = 'best.onnx'
img_w = 640
img_h = 640
classes_file = 'classes.txt'

def class_names():
    classes = []
    with open(classes_file, 'r') as file:
        for line in file:
            name = line.strip('\n')
            classes.append(name)
    return classes

width_frame = 640
net = cv2.dnn.readNetFromONNX(model)
classes = (0)

# Dahua CCTV camera settings
rtsp_url = 'rtsp://admin:DCTGroup18@10.10.12.14:554/cam/realmonitor?channel=1&subtype=0'  # Replace with your Dahua CCTV camera RTSP URL

cap = cv2.VideoCapture(rtsp_url)

total_slots = 15
cars_count = 0
empty_slots = 0
last_detection_time = datetime.datetime.now()

# Specify the directory path to save the image results
output_directory = r'D:\Database\Test\Test10nov'

while True:
    current_time = datetime.datetime.now()
    time_difference = current_time - last_detection_time

    if time_difference.total_seconds() >= (15 * 60):
        detection_start_time = datetime.datetime.now()
        detection_end_time = detection_start_time + datetime.timedelta(seconds=5)
        last_detection_time = detection_end_time

        while datetime.datetime.now() <= detection_end_time:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            height = int(frame.shape[0] * (width_frame / frame.shape[1]))
            dim = (width_frame, height)
            img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            blob = cv2.dnn.blobFromImage(img, 1/255, (img_w, img_h), swapRB=True, mean=(0, 0, 0), crop=False)
            net.setInput(blob)
            t1 = time.time()
            outputs = net.forward(net.getUnconnectedOutLayersNames())
            t2 = time.time()
            out = outputs[0]
            n_detections = out.shape[1]
            height, width = img.shape[:2]
            x_scale = width / img_w
            y_scale = height / img_h
            conf_threshold = 0.7
            score_threshold = 0.5
            nms_threshold = 0.5

            class_ids = [0]
            score = []
            boxes = []

            for i in range(n_detections):
                detect = out[0][i]
                confidence = detect[4]
                if confidence >= conf_threshold:
                    class_score = detect[5:]
                    class_id = np.argmax(class_score)
                    if class_id == 0 and class_score[class_id] > score_threshold:
                        score.append(confidence)
                        class_ids.append(class_id)
                        x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                        left = int((x - w/2) * x_scale)
                        top = int((y - h/2) * y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
                        classes = {0: "cars"}


            indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)

            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                class_id = class_ids[i]


                if class_id == 0:
                    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 2)
                    label = "{}".format(classes[class_id])
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(img, (left, top - 20), (left + dim[0], top + dim[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, label, (left, top + dim[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    


                cars_count = len(indices)
                
                empty_slots = total_slots - cars_count

            # Calculate the width and height of the text
            text_width_count, text_height_count = cv2.getTextSize(f"Cars Count: {cars_count}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_width_slots, text_height_slots = cv2.getTextSize(f"Empty Slots: {empty_slots}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Calculate the coordinates to place the text in the top right corner
            text_x_count = img.shape[1] - text_width_count - 5  # 5 pixels offset from the right edge
            text_y_count = 65

            text_x_slots = img.shape[1] - text_width_slots - 5  # 5 pixels offset from the right edge
            text_y_slots = 87


            cv2.putText(img, f"Cars Count: {cars_count}", (text_x_count, text_y_count),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Empty Slots: {empty_slots}", (text_x_slots, text_y_slots),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Save the image results
            filename = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, img)

            # Display the captured frame
            cv2.imshow("Object Detection", img)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Display the captured frame
        cv2.imshow("Camera Feed", frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()