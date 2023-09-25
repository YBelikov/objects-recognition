import ntpath
import click
import os

from video_stream_object_recognizer import VideoStreamObjectRecognizer

def process_file(source_path, result_path):
    if not os.path.exists(source_path):
        print('The given input path does not exist')
        return
    if not os.path.exists(result_path):
        print('Given result directory does not exist. Will create it')
        os.makedirs(result_path)
    video_stream_object_recognizer = VideoStreamObjectRecognizer(source_video=source_path, result_path=os.path.join(result_path, path_last_component(source_path)))
    video_stream_object_recognizer.process_video()
    video_stream_object_recognizer.release_processed_video()

def path_last_component(path):
    head, tail = os.path.split(path)
    return tail or ntpath.basename(path)
def main():
    process_file()
if __name__ == '__main__':
    main()
