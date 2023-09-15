from ultralytics import YOLO

import click
import cv2


DEFAULT_SOURCE_PATH = '../test-data/jj-jordan-jtZsFCvXS_g-unsplash.jpg'

@click.command()
@click.option('--source-path', prompt='Please, provide a path to a file for processing', help='A path to the target file')
@click.option('--result-path', default='../results', help='A path to which the result is saved')
def process_file(source_path: str, result_path: str):
    # CAR_DETECTION_MODEL_PATH = '../models/yolov8n.pt'
    # car_detection_model = YOLO(CAR_DETECTION_MODEL_PATH)
    LICENSE_PLATE_RECOGNITION_MODEL_PATH = '../models/license_plate_recognition_model.pt'
    results = license_plate_detection_model = YOLO(LICENSE_PLATE_RECOGNITION_MODEL_PATH)
    license_plate_detection_model(source=source_path, save=True, project=result_path)


def read_video():
    cap = cv2.VideoCapture('../test-data/license_plate_recognition_sample.mp4')
    if cap.isOpened() == False:
        print('Unable to open file!')
        return
    while cap.isOpened():
        has_frame_to_read, frame = cap.read()
        if has_frame_to_read == False:
            break
        cv2.imshow('Test video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
  read_video()

if __name__ == '__main__':
    main()