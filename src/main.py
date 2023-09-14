from ultralytics import YOLO

def main():
    car_detection_model = YOLO('../models/yolov8n.pt')
    license_plate_detection_model = YOLO('../models/license_plate_recognition_model.pt')
    license_plate_detection_model(source='../test-data/jj-jordan-jtZsFCvXS_g-unsplash.jpg', save=True, project='../results')

if __name__ == '__main__':
    main()