from ultralytics import YOLO
import click
from video_stream_object_recognizer import VideoStreamObjectRecognizer


# @click.command()
# @click.option('--source-path', prompt='Please, provide a path to a file for processing', help='A path to the target file')
# @click.option('--result-path', default='../results', help='A path to which the result is saved')
# def process_file(source_path: str, result_path: str):
#     # CAR_DETECTION_MODEL_PATH = '../models/yolov8n.pt'
#     # car_detection_model = YOLO(CAR_DETECTION_MODEL_PATH)
#     LICENSE_PLATE_RECOGNITION_MODEL_PATH = '../models/license_plate_recognition_model.pt'
#     license_plate_detection_model = YOLO(LICENSE_PLATE_RECOGNITION_MODEL_PATH)
#     license_plate_detection_model(source=source_path, save=True, project=result_path)
#

def main():
    video_stream_object_recognizer = VideoStreamObjectRecognizer('../test-data/sample.mkv', '../results')
    video_stream_object_recognizer.process_video()
    video_stream_object_recognizer.release_processed_video()

if __name__ == '__main__':
    main()
