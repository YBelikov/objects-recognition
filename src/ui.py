from argparse import ArgumentParser, Namespace
from pathlib import Path

import PySimpleGUI as sg
import numpy as np

import cv2


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

    def set_frames(self, frames: FrameSource):
        self.frames = frames
    
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

        self.window[GUI.IMG_KEY].update(data=self._get_byte_frame(0))

    
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
            
            

    def _update_image(self, value: float) -> None:
        self.window[GUI.IMG_KEY].update(data=self._get_byte_frame(value))

    def _get_byte_frame(self, idx: int) -> bytes:
        img = self.frames[int(idx)]
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

    window = GUI()
    window.set_frames(frames)

    try:
        window.run()
    finally:
        frames.close()


if __name__ == "__main__":
    start_gui()

