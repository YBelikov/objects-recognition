import cv2
from ultralytics import YOLO
import easyocr

CAR_DETECTION_MODEL_PATH = '../models/yolov8n.pt'
LICENSE_PLATE_RECOGNITION_MODEL_PATH = '../models/license_plate_recognition_model.pt'
DEBUG_FRAME_LIMIT = 300

class VideoStreamObjectRecognizer:
    def __init__(self, source_video: str, result_dir: str):
        self.source = source_video
        self.result = result_dir
        self.car_recognition_model = YOLO(CAR_DETECTION_MODEL_PATH)
        self.license_plate_recognition_model = YOLO(LICENSE_PLATE_RECOGNITION_MODEL_PATH)
        self.car_detection_info = {}
        self.license_reader = easyocr.Reader(['en'], gpu=False)

    def process_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print('Unable to open file!')
            return
        frame_counter = 0
        while cap.isOpened() and frame_counter < DEBUG_FRAME_LIMIT:
            has_frame_to_read, frame = cap.read()
            if not has_frame_to_read:
                break
            car_model_detections = self.car_recognition_model(frame)[0].boxes.data.tolist()
            detected_cars_parameters = []
            for detected_car in car_model_detections:
                car_x1, car_y1, car_x2, car_y2, confidence_score, class_id = detected_car
                if class_id != 2:
                    break
                detected_cars_parameters.append([car_x1, car_y1, car_x2, car_y2, confidence_score])
                license_plate_detections = self.license_plate_recognition_model(frame)[0].boxes.tolist()
               # plates_on_frame_info = process_plates(license_plate_detections)
               # self.car_detection_info[frame_counter] = {'cars_info' : detected_cars_parameters, 'plates_info' : license_plate_detections}
                self.car_detection_info[frame_counter] = detected_cars_parameters
            frame_counter += 1
        cap.release()

    def process_plates(self, frame, recognized_plates_detections):
        for detection in recognized_plates_detections:
            license_x1, license_y1, license_x2, license_y2, confidence_score, _ = detection
            cropped_license_image = frame[int(license_x1):int(license_x2), int(license_y1):int(license_y2), :]
            license_plate_crop_gray = cv2.cvtColor(cropped_license_image, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            text_detections = self.license_reader.readtext(license_plate_crop_gray)

    def release_processed_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print('Unable to save processed video!')
            return
        frame_counter = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
        while cap.isOpened() and frame_counter < DEBUG_FRAME_LIMIT:
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter not in self.car_detection_info.keys():
                continue
            car_bounding_box_for_frame = self.car_detection_info[frame_counter] # {key : value for (key, value) in self.car_detection_info.items() if frame_counter == key}[frame_counter]['bbox']
            for tracking_info in car_bounding_box_for_frame:
                car_x1, car_y1, car_x2, car_y2 = tracking_info[0], tracking_info[1], tracking_info[2], tracking_info[3]
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 0, 255), 12)
                cv2.putText(frame, 'Car', (int(car_x1), int(car_y1) - 10), cv2.FONT_HERSHEY_PLAIN, 4.2, (255, 255, 255), 8)
            out.write(frame)
        cap.release()