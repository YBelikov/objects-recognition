from argparse import ArgumentParser, Namespace
from pathlib import Path

import PySimpleGUI as sg
import numpy as np

import torch
import cv2


from ultralytics import YOLO
import easyocr


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FrameSource:
    def __init__(self, video: Path) -> None:
        self.video = video
        self.cap = cv2.VideoCapture(str(self.video))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self) -> int:
        return self.total_frames
    
    def __getitem__(self, idx) -> np.ndarray:
        idx = np.clip(idx, 0, self.total_frames - 1)
        
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        
        suc, img = self.cap.read()
        return img
    
    def close(self) -> None:
        self.cap.release()


class CarRecognizer:
    def __init__(self):
        car_recognition_model_path=Path(__file__).parent.parent / 'models' / 'yolov8n.pt'
        self.car_recognition_model = YOLO(car_recognition_model_path)

        license_plate_recognition_model_path=Path(__file__).parent.parent / 'models' / 'license_plate_recognition_model.pt'
        self.license_plate_recognition_model = YOLO(license_plate_recognition_model_path)

        self.license_reader = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'))

    def __call__(self, frame: np.ndarray):
        detected_cars_info = []
        detected_persons_info = []
        detected_license_plates = []

        detected_vehicles = self.car_recognition_model(frame)[0].boxes.data.tolist()
        
        for vehicle in detected_vehicles:
            x1, y1, x2, y2, confidence_score, class_id = vehicle
            if self.car_recognition_model.names[class_id] == 'car':
                detected_cars_info.append([[x1, y1, x2, y2], []])
            
            if self.car_recognition_model.names[class_id] == 'person':
                detected_persons_info.append([x1, y1, x2, y2])

        license_plate_detections = self.license_plate_recognition_model(frame)[0].boxes.data.tolist()
        # license_plate_detections = self.license_plate_recognition_model.predict(frame, imgsz=max(frame.shape))[0].boxes.data.tolist()

        for detection in license_plate_detections:
            license_x1, license_y1, license_x2, license_y2, license_bbox_confidence_score, _ = detection
            cropped_license_image = frame[int(license_y1):int(license_y2), int(license_x1):int(license_x2), :]
            license_plate_crop_gray = cv2.cvtColor(cropped_license_image, cv2.COLOR_BGR2GRAY)

            owner_cars_idx = []
            for car_idx, (car_rect, license_info) in enumerate(detected_cars_info):
                if CarRecognizer.intersection_area(car_rect, (license_x1, license_y1, license_x2, license_y2)) >= 0.95:
                    owner_cars_idx.append(car_idx)
            
            if len(owner_cars_idx) != 1:
                continue

            result = self.license_reader.readtext(license_plate_crop_gray)
            if len(result) == 0:
                text = ""
            else:
                _, text, _ = result[0]
            license_info.extend([license_x1, license_y1, license_x2, license_y2, text])
        
        return detected_cars_info, detected_persons_info

    @staticmethod
    def intersection_area(rect1, rect2):
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))

        return x_overlap * y_overlap

class GUI:
    IMG_KEY = "-Image-"
    PROGRESSBAR_KEY = "-Image-Progress-"
    BUTTON_NEXT_ONE = "-Next-one-button-"
    BUTTON_NEXT_TEN = "-Next-ten-button-"
    BUTTON_PREV_ONE = "-Prev-one-button-"
    BUTTON_PREV_TEN = "-Prev-ten-button-"
    IMG_H = 600

    def __init__(self):
        self.frames: FrameSource = None
        self.window: sg.Window = None
        self.car_recognizer: CarRecognizer = None

        self.img: np.ndarray = None
        self.detected_cars_info = None
        self.detected_persons_info = None

    def set_frames(self, frames: FrameSource):
        self.frames = frames
    
    def set_car_recognizer(self, car_recognizer: CarRecognizer) -> None:
        self.car_recognizer = car_recognizer
    
    def run(self) -> None:
        try:
            self._build_window()
            self._start_event_loop()
        finally:
            if self.window is not None and not self.window.is_closed():
                self.window.close()
    
    def _build_window(self) -> None:
        video_layout = [
            [sg.Image(key=GUI.IMG_KEY, filename="")],
            [
            sg.Button(button_text="<<<", key=GUI.BUTTON_PREV_TEN),
            sg.Button(button_text="<", key=GUI.BUTTON_PREV_ONE),
            sg.Slider(
                key=GUI.PROGRESSBAR_KEY,
                orientation='h', 
                range=(0, len(self.frames)),
                resolution=1,
                enable_events=True,
                expand_x=True
            ),
            sg.Button(button_text=">", key=GUI.BUTTON_NEXT_ONE),
            sg.Button(button_text=">>>", key=GUI.BUTTON_NEXT_TEN),
            ]
        ]

        layout = [
            [
                video_layout
            ]
        ]

        self.window = sg.Window(
            title="Saitarly, Belikov, Rymanov Demo", 
            layout=layout, 
            margins=(100, 50),
            finalize=True
        )

        self._update_image(0)

    
    def _start_event_loop(self) -> None:
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break

            if event == GUI.PROGRESSBAR_KEY:
                self._update_image(values[GUI.PROGRESSBAR_KEY])
            
            if event == GUI.BUTTON_NEXT_ONE:
                self.window[GUI.PROGRESSBAR_KEY].update(value=values[GUI.PROGRESSBAR_KEY] + 1)
                self._update_image(values[GUI.PROGRESSBAR_KEY] + 1)
            
            if event == GUI.BUTTON_NEXT_TEN:
                self.window[GUI.PROGRESSBAR_KEY].update(value=values[GUI.PROGRESSBAR_KEY] + 10)
                self._update_image(values[GUI.PROGRESSBAR_KEY] + 10)
            
            if event == GUI.BUTTON_PREV_ONE:
                self.window[GUI.PROGRESSBAR_KEY].update(value=values[GUI.PROGRESSBAR_KEY] - 1)
                self._update_image(values[GUI.PROGRESSBAR_KEY] - 1)
            
            if event == GUI.BUTTON_PREV_TEN:
                self.window[GUI.PROGRESSBAR_KEY].update(value=values[GUI.PROGRESSBAR_KEY] - 10)
                self._update_image(values[GUI.PROGRESSBAR_KEY] - 10)
            

    def _draw_ui_image(self):
        img = self.img.copy()

        for human in self.detected_persons_info:
            GUI.draw_bb(img, *human, (30, 250, 30))
        
        for car_bb, license_bb in self.detected_cars_info:
            GUI.draw_bb(img, *car_bb, (250, 30, 30))

            if len(license_bb) == 5:
                GUI.draw_bb(img, *license_bb[:4], (30, 30, 255), license_bb[4])

        self.window[GUI.IMG_KEY].update(data=self._get_byte_frame(img))

    @staticmethod
    def draw_bb(image: np.ndarray, x1, y1, x2, y2, color, text: str = None):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        if text is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 - 5

            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    def _update_image(self, value: float) -> None:
        self.img = self.frames[int(value)]
        self.detected_cars_info, self.detected_persons_info = self.car_recognizer(self.img)
        
        self._draw_ui_image()
        
    def _get_byte_frame(self, img: np.ndarray) -> bytes:
        
        img = cv2.resize(img, (int(GUI.IMG_H * img.shape[1] / img.shape[0]), GUI.IMG_H))
        return cv2.imencode(".png", img)[1].tobytes()




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--video_file', type=Path, required=True)
    args = parser.parse_args()
    return args

def start_gui() -> None:
    args = parse_args()
    frames = FrameSource(args.video_file)
    car_detector = CarRecognizer()
    
    window = GUI()
    window.set_frames(frames)
    window.set_car_recognizer(car_detector)

    try:
        window.run()
    finally:
        frames.close()


if __name__ == "__main__":
    start_gui()

