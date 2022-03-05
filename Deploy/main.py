from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.utils import platform
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.camera import Camera
from os import listdir, getcwd
from os.path import isfile, join
#from kivy.graphics.texture import Texture

import cv2
import io
import numpy as np
#import threading
import time
import requests

"""
class CameraRead(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        vid = cv2.VideoCapture(0)
        while 1:
            ret, frame = vid.read()
            cv2.imshow('vid', frame)
            print("Here")
            time.sleep(1)
"""

class KivyCamera(Image):
    def __init__(self, capture=None, fps=0, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        # self.capture = cv2.VideoCapture("/sdcard2/python-apk/2.mp4")
        # print "file path exist :" + str(os.path.exists("/sdcard2/python-apk/1.mkv"))
        # self.capture = cv2.VideoCapture(0)
        self.capture = Camera(index=0, resolution=(640, 480), play=True)
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        self.texture = self.capture.texture
        #ret, frame = self.capture.read()
        # print str(os.listdir('/sdcard2/'))
        #if ret:
        # Example
        """
            url = 'https://i.pinimg.com/564x/2a/3b/17/2a3b175c8b6752a62a6f6915ff472f8c.jpg'
            bimage = requests.get(url).content
            file_extension = 'jpg'
            buf = io.BytesIO(bimage)
            cim = CoreImage(buf, ext=file_extension)
            self.image = Image(texture=cim.texture)
            self.add_widget(self.image)
        """

        """
            #buf = cv2.flip(frame, 2)
            bytes = io.BytesIO(cv2.imencode('.jpg', frame)[1].tobytes())
            cim = CoreImage(bytes, ext="jpg")
            image_texture = cim.texture
            self.texture = image_texture
        """

        """
            # image is a Kivy Image widget
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
        """
"""
class AndroidCamera(Camera):
    camera_resolution = (640, 480)
    counter = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = 0
        resolution: self.camera_resolution
        allow_stretch: True
        play: True

    def _camera_loaded(self, *largs):
        self.texture = Texture.create(size=np.flip(self.camera_resolution, 0), colorfmt='rgb')
        self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        if self._camera._buffer is None:
            print("None here")
            return None
        frame = self.frame_from_buf()
        self.frame_to_screen(frame)
        super(AndroidCamera, self).on_tex(*l)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_rgb, str(self.counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.counter += 1
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tostring()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
"""

class Main(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.button = Button(text="Click me", on_press=self.button_pressed, size_hint=(1, 0.1))

        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.INTERNET, Permission.WRITE_EXTERNAL_STORAGE])

        #self.camera = KivyCamera(fps=12)
        self.camera = Camera(index=0, resolution=(640, 480), play=True)
        #self.image = Image()
        #onlyfiles = [f for f in listdir("/storage/0123-4567") if isfile(join("/storage/0123-4567", f))]
        #print(getcwd())
        #print(onlyfiles)

        self.add_widget(self.button)
        self.add_widget(self.camera)


        url = 'https://i.pinimg.com/564x/2a/3b/17/2a3b175c8b6752a62a6f6915ff472f8c.jpg'
        bimage = requests.get(url).content
        print(type(bimage))
        file_extension = 'jpg'
        buf = io.BytesIO(bimage)
        print(type(buf))
        cim = CoreImage(buf, ext=file_extension)
        self.image = Image(texture=cim.texture)
        self.add_widget(self.image)

        #self.add_widget(self.image)

        #Clock.schedule_interval(self.update, 1.0)

    def update(self, dt):
        if self.camera.texture:
            self.camera.export_to_png('/storage/0123-4567/tmp.png')
            frame = cv2.imread("/storage/0123-4567/tmp.png", cv2.IMREAD_COLOR)
            #self.camera.export_to_png('tmp.png')
            #frame = cv2.imread("tmp.png", cv2.IMREAD_COLOR)
            """
            cv2.putText(frame, "Hello", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            buf = cv2.flip(frame, 0)
            file_extension = 'png'
            my_buf = io.BytesIO(open("tmp.png", "rb").read())
            cim = CoreImage(my_buf, ext=file_extension)
            print(cim)

            image_texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
            """
            self.image.texture = cim.texture

    def button_pressed(self, obj):
        pass
        #self.camera.log = not self.camera.log


class mainApp(App):
    def build(self):
        return Main()

if __name__ == "__main__":
    #CameraRead().start()
    mainApp().run()
