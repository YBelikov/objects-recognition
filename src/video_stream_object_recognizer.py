import cv2
from ultralytics import YOLO
import easyocr

CAR_DETECTION_MODEL_PATH = '../models/yolov8n.pt'
LICENSE_PLATE_RECOGNITION_MODEL_PATH = '../models/license_plate_recognition_model.pt'
DEBUG_FRAME_LIMIT = 200

class VideoStreamObjectRecognizer:
    def __init__(self, source_video: str, result_dir: str):
        self.source = source_video
        self.result = result_dir
        self.car_recognition_model = YOLO(CAR_DETECTION_MODEL_PATH)
        self.license_plate_recognition_model = YOLO(LICENSE_PLATE_RECOGNITION_MODEL_PATH)
        self.car_detection_info = {}
        self.license_reader = easyocr.Reader(['en'], gpu=False)

        # actually we are interested in vehicles with class '2' (car) from COCO dataset
        self.vehicle_classes_to_track = [2]

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

            # Find and store bounding cars' bounding boxes and their corresponding confidence scores in one array
            # First four elements of the array are bbox parameters, the fifth is the confidence score
            detected_vehicles = self.car_recognition_model(frame)[0].boxes.data.tolist()
            detected_cars_info = []
            for vehicle in detected_vehicles:
                vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, confidence_score, class_id = vehicle
                if class_id not in self.vehicle_classes_to_track:
                    break
                detected_cars_info.append([vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, confidence_score])

            license_plate_detections = self.license_plate_recognition_model(frame)[0].boxes.data.tolist()
            detected_license_plates_info = []
            for detection in license_plate_detections:
                license_x1, license_y1, license_x2, license_y2, confidence_score, _ = detection
                cropped_license_image = frame[int(license_y1):int(license_y2), int(license_x1):int(license_x2), :]
                license_plate_crop_gray = cv2.cvtColor(cropped_license_image, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                result = self.license_reader.readtext(license_plate_crop_gray)
                if not result:
                    continue
                _, text, confidence_score = result[0]
                detected_license_plates_info.append([license_x1, license_y1, license_x2, license_y2, text, confidence_score])

            self.car_detection_info[frame_counter] = {'car_info' : detected_cars_info, 'plate_info' : detected_license_plates_info}
            frame_counter += 1
        cap.release()

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
            bounding_boxes_info = self.car_detection_info[frame_counter] # {key : value for (key, value) in self.car_detection_info.items() if frame_counter == key}[frame_counter]['bbox']
            cars_info = bounding_boxes_info['car_info']
            plates_info = bounding_boxes_info['plate_info']
            for car in cars_info:
                car_x1, car_y1, car_x2, car_y2 = car[0], car[1], car[2], car[3]
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 0, 255), 4)
                cv2.putText(frame, 'Car', (int(car_x1), int(car_y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2.2, (255, 255, 255), 4)
            for plate in plates_info:
                plate_x1, plate_y1, plate_x2, plate_y2, license = plate[0], plate[1], plate[2], plate[3], plate[4]
                cv2.rectangle(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 4)
                cv2.putText(frame, license, (int(plate_x1), int(plate_y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2.2, (255, 255, 255), 4)
            out.write(frame)
        cap.release()