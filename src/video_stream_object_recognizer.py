from ultralytics import YOLO
from sort.sort import *
import cv2
import easyocr
import pandas as pd
import re

DEBUG_FRAME_LIMIT = 1000

class VideoStreamObjectRecognizer:
    def __init__(self,
                 source_video,
                 result_path,
                 car_recognition_model_path='../models/yolov8n.pt',
                 license_plate_recognition_model_path='../models/license_plate_recognition_model.pt'):
        self.source = source_video
        self.result = result_path
        self.car_recognition_model = YOLO(car_recognition_model_path)
        self.license_plate_recognition_model = YOLO(license_plate_recognition_model_path)
        self.car_detection_dataframe = pd.DataFrame(columns=['frame_number', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_confidence_score', 'license_number', 'license_text_confidence_score'])
        self.motion_objects_tracker = Sort()
        self.license_reader = easyocr.Reader(['en'], gpu=False)
        # actually we are interested in vehicles with the class '2' (car) from the COCO dataset
        self.vehicle_classes_to_track = [2]

    def process_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print('Unable to open file!')
            return
        frame_counter = 0
        while cap.isOpened() and frame_counter <= DEBUG_FRAME_LIMIT:
            has_frame_to_read, frame = cap.read()
            if not has_frame_to_read:
                break
            frame_counter += 1
            # Find and store bounding cars' bounding boxes and their corresponding confidence scores in one array
            # First four elements of the array are bbox parameters, the fifth is the confidence score
            detected_vehicles = self.car_recognition_model(frame)[0].boxes.data.tolist()
            detected_cars_info = []
            for vehicle in detected_vehicles:
                vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, confidence_score, class_id = vehicle
                if class_id not in self.vehicle_classes_to_track:
                    break
                detected_cars_info.append([vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2])
            if not detected_cars_info:
                self.update_data(frame_number=frame_counter, car_id=-1, car_bbox=[0, 0, 0, 0],
                                 license_plate_bbox=[0, 0, 0, 0], license_plate_bbox_confidence_score=1,
                                 license_number='N/R', license_text_confidence_score=0.0)
                continue
            tracking_ids = self.motion_objects_tracker.update(np.asarray(detected_cars_info))
            license_plate_detections = self.license_plate_recognition_model(frame)[0].boxes.data.tolist()

            for detection in license_plate_detections:
                license_x1, license_y1, license_x2, license_y2, license_bbox_confidence_score, _ = detection
                cropped_license_image = frame[int(license_y1):int(license_y2), int(license_x1):int(license_x2), :]
                license_plate_crop_gray = cv2.cvtColor(cropped_license_image, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                corresponding_car_info = self.find_corresponding_car_id((license_x1, license_y1, license_x2, license_y2), tracking_ids)
                if corresponding_car_info is None:
                    # unable to find corresponding car
                    self.update_data(frame_number=frame_counter, car_id=-1, car_bbox=[0, 0, 0, 0], license_plate_bbox=[0, 0, 0, 0], license_plate_bbox_confidence_score=1, license_number='N/R', license_text_confidence_score=0.0)
                    continue

                car_x1, car_y1, car_x2, car_y2, car_id = corresponding_car_info
                result = self.license_reader.readtext(license_plate_crop_gray)

                # unable to recognize the license number
                if not result:
                    self.update_data(frame_number=frame_counter, car_id=car_id, car_bbox=[car_x1, car_y1, car_x2, car_y2], license_plate_bbox=[license_x1, license_y1,
                                                                                            license_x2, license_y2], license_plate_bbox_confidence_score=license_bbox_confidence_score, license_number='N/R', license_text_confidence_score=0.0)
                    continue
                _, text, text_confidence_score = result[0]
                text = text.upper().replace(' ', '')
                if not re.search('[A-Z]{2}\d{4}[A-Z]{2}', text):
                    self.update_data(frame_number=frame_counter, car_id=car_id,
                                     car_bbox=[car_x1, car_y1, car_x2, car_y2],
                                     license_plate_bbox=[license_x1, license_y1,
                                                         license_x2, license_y2],
                                     license_plate_bbox_confidence_score=license_bbox_confidence_score, license_number='N/R',
                                     license_text_confidence_score=0.0)
                    continue
                self.update_data(frame_number=frame_counter, car_id=car_id, car_bbox=[car_x1, car_y1, car_x2, car_y2],
                                 license_plate_bbox=[license_x1, license_y1,
                                                     license_x2, license_y2],
                                 license_plate_bbox_confidence_score=license_bbox_confidence_score, license_number=text,
                                 license_text_confidence_score=text_confidence_score)
        cap.release()

    def find_corresponding_car_id(self, license_plate_bbox, track_ids):
        for idx in range(len(track_ids)):
            car_x1, car_y1, car_x2, car_y2, car_id = track_ids[idx]
            license_x1, license_y1, license_x2, license_y2 = license_plate_bbox
            if car_x1 < license_x1 and car_y1 < license_y1 and car_x2 > license_x2 and car_y2 > license_y2:
                return track_ids[idx]
        return None

    def update_data(self, frame_number, car_id, car_bbox, license_plate_bbox, license_plate_bbox_confidence_score, license_number, license_text_confidence_score):
        self.car_detection_dataframe.at[frame_number, 'frame_number'] = frame_number
        self.car_detection_dataframe.at[frame_number, 'car_id'] = car_id
        self.car_detection_dataframe.at[frame_number, 'car_bbox'] = car_bbox
        self.car_detection_dataframe.at[frame_number, 'license_plate_bbox'] = license_plate_bbox
        self.car_detection_dataframe.at[frame_number, 'license_plate_bbox_confidence_score'] = license_plate_bbox_confidence_score
        self.car_detection_dataframe.at[frame_number, 'license_number'] = license_number
        self.car_detection_dataframe.at[frame_number, 'license_text_confidence_score'] = license_text_confidence_score

    def release_processed_video(self):
        license_plate = {}
        for car_id in np.unique(self.car_detection_dataframe['car_id']):
            max_confidence = np.amax(self.car_detection_dataframe[self.car_detection_dataframe['car_id'] == car_id]['license_text_confidence_score'])
            license_plate[car_id] = {'license_plate_number': self.car_detection_dataframe[(self.car_detection_dataframe['car_id'] == car_id) &
                                                                     (self.car_detection_dataframe['license_text_confidence_score'] == max_confidence)][
                                         'license_number'].iloc[0] }

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print('Unable to save processed video!')
            return
        frame_counter = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.result, fourcc, fps, (width, height))
        while cap.isOpened() and frame_counter <= DEBUG_FRAME_LIMIT:
            ret, frame = cap.read()
            if not ret:
                out.release()
                break
            frame_counter += 1
            data_for_frame = self.car_detection_dataframe[self.car_detection_dataframe['frame_number'] == frame_counter]
            for data_sample_idx in range(len(data_for_frame)):
                car_bbox = data_for_frame.iloc[data_sample_idx]['car_bbox']
                car_x1, car_y1, car_x2, car_y2 = car_bbox[0], car_bbox[1], car_bbox[2], car_bbox[3]
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 0, 255), 4)
                cv2.putText(frame, 'Car', (int(car_x1), int(car_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                plate_bbox = data_for_frame.iloc[data_sample_idx]['license_plate_bbox']
                plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox[0], plate_bbox[1], plate_bbox[2], plate_bbox[3]
                cv2.rectangle(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 4)
                cv2.putText(frame, str(license_plate[data_for_frame.iloc[data_sample_idx]['car_id']]['license_plate_number']), (int(plate_x1), int(plate_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        cap.release()