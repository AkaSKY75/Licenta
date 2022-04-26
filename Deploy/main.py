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
from kivy.graphics import Rotate, PushMatrix, PopMatrix, Translate, Color, Rectangle, BindTexture
from kivy.clock import Clock
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.metrics import dp
from os import listdir, getcwd
from os.path import isfile, join
#from kivy.graphics.texture import Texture
from base64 import b64encode

import cv2
import io
import numpy as np
#import threading
import _thread as thread
import time
import random
import requests

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, uuid

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
training_key = "5887d793471f48dc8417e84c699d561d"
prediction_key = "5887d793471f48dc8417e84c699d561d"
prediction_resource_id = "/subscriptions/2c0c2800-5672-4899-963d-d9c6e8984e61/resourceGroups/appsvc_windows_centralus/providers/Microsoft.CognitiveServices/accounts/Licenta"
project_id = "2fbc8c69-c0c5-42f7-83c1-63e91c12751b"

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

class Main(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        with self.canvas.before:
            PushMatrix()
            Rotate(origin=self.center, angle=45)
            Color(rgb=(1,255,0))
            Rectangle(pos=(0, 0), size=(50, 50))
            PopMatrix()
        """

        self.boundingBox_coords = [(-1, -1), (-1, -1)]

        self.is_process_ongoing = False

        self.button = Button(text="Click me", on_press=self.button_pressed, size=(Window.size[0], Window.size[1]/10), pos=(0, Window.size[1]-Window.size[1]/10))
        self.add_widget(self.button)

        Window.bind(on_resize=self.on_window_resize)

        self.camera = Camera(index=0, resolution=(640, 480), play=True, allow_stretch=True, pos=(0, 0), size=(Window.size[0], 480/640*Window.size[0]), on_touch_down=self.camera_click)
#       self.camera._camera.on_load = self.camera_on_load
        self.camera._camera.unbind(on_texture=self.camera.on_tex)
        self.camera._camera.bind(on_texture=self.on_texture_change)

        self.boundingBox = Image(allow_stretch=True, pos=(0, 0), size=(Window.size[0], 480/640*Window.size[0]))
        self.nparr = np.zeros((480, 640, 4), dtype=np.uint8)
        image_texture = Texture.create(size=(self.nparr.shape[1], self.nparr.shape[0]), colorfmt='rgba')
        image_texture.blit_buffer(self.nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.boundingBox.texture = image_texture

        """
        with self.camera.canvas.before:
            Color(255, 0, 0, 1)
            Rectangle(pos=self.camera.pos, size=self.camera.size)
        """

        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.INTERNET, Permission.WRITE_EXTERNAL_STORAGE])

            """
            center = (Window.size[0]/2, Window.size[1]/2)

            with self.camera.canvas.before:
                PushMatrix()
                Rotate(origin=center, angle=-90)
            with self.camera.canvas.after:
                PopMatrix()
            """
            #self.camera.size=(Window.size[1]*9/10, Window.size[0])

        """
        with self.camera.canvas.before:
            PushMatrix()
            Rotate(origin=self.center, angle=90)
            #Translate(-480, -640)

        with self.camera.canvas.after:
            PopMatrix()
        """

        #self.camera = KivyCamera(fps=12)
        #self.camera = Camera(index=0, resolution=(640, 480), play=True, allow_stretch=True)
        #self.image = Image()
        #onlyfiles = [f for f in listdir("/storage/0123-4567") if isfile(join("/storage/0123-4567", f))]
        #print(getcwd())
        #print(onlyfiles)

#        self.credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
#        self.trainer = CustomVisionTrainingClient(ENDPOINT, self.credentials)
#        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
#        predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

        self.add_widget(self.camera)

        self.add_widget(self.boundingBox)

        self.boxLayout = BoxLayout(pos=(0, dp(200)), size=(Window.size[0]/2, Window.size[1]*9/10-dp(200)), orientation="vertical")
        with self.boxLayout.canvas:
            Color(1, 0, 0, 0.5)
            Rectangle(pos=self.boxLayout.pos, size=self.boxLayout.size)

        self.add_widget(self.boxLayout)

        self.boxLayout.window_size_x_label = Label(text="Window x size: "+str(Window.size[0]), color="black")
        self.boxLayout.window_size_y_label = Label(text="Window y size: "+str(Window.size[1]), color="black")
        self.boxLayout.camera_pos_x_label = Label(text="Camera x pos: "+str(self.camera.pos[0]), color="black")
        self.boxLayout.camera_pos_y_label = Label(text="Camera y pos: "+str(self.camera.pos[1]), color="black")
        self.boxLayout.camera_size_x_label = Label(text="Camera x size: "+str(self.camera.size[0]), color="black")
        self.boxLayout.camera_size_y_label = Label(text="Camera y size: "+str(self.camera.size[1]), color="black")



        self.plus_button = Button(text="+", on_press=self.image_grow, size=(dp(100), dp(100)), pos=(Window.size[0]-dp(100), dp(100)), background_color=(1, 0, 0, 0.5))
        self.minus_button = Button(text="-", on_press=self.image_shrink, size=(dp(100), dp(100)), pos=(Window.size[0]-dp(100), 0), background_color=(1, 0, 0, 0.5))

        self.move_left_button = Button(text="Left", on_press=self.move_left, size=(dp(100), dp(100)), pos=(0, 0), background_color=(1, 0, 0, 0.5))
        self.move_right_button = Button(text="Right", on_press=self.move_right, size=(dp(100), dp(100)), pos=(dp(200), 0), background_color=(1, 0, 0, 0.5))
        self.move_up_button = Button(text="Up", on_press=self.move_up, size=(dp(100), dp(100)), pos=(dp(100), dp(100)), background_color=(1, 0, 0, 0.5))
        self.move_down_button = Button(text="Down", on_press=self.move_down, size=(dp(100), dp(100)), pos=(dp(100), 0), background_color=(1, 0, 0, 0.5))

        # Buttons
        self.add_widget(self.plus_button)
        self.add_widget(self.minus_button)
        self.add_widget(self.move_left_button)
        self.add_widget(self.move_right_button)
        self.add_widget(self.move_up_button)
        self.add_widget(self.move_down_button)

        #Labels
        self.boxLayout.add_widget(self.boxLayout.window_size_x_label)
        self.boxLayout.add_widget(self.boxLayout.window_size_y_label)
        self.boxLayout.add_widget(self.boxLayout.camera_pos_x_label)
        self.boxLayout.add_widget(self.boxLayout.camera_pos_y_label)
        self.boxLayout.add_widget(self.boxLayout.camera_size_x_label)
        self.boxLayout.add_widget(self.boxLayout.camera_size_y_label)




        """
        url = 'https://i.pinimg.com/564x/2a/3b/17/2a3b175c8b6752a62a6f6915ff472f8c.jpg'
        bimage = requests.get(url).content
        print(type(bimage))
        file_extension = 'jpg'
        buf = io.BytesIO(bimage)
        print(type(buf))
        cim = CoreImage(buf, ext=file_extension)
        self.image = Image(texture=cim.texture)
        self.add_widget(self.image)
        """

#        self.add_widget(self.image)

#        Clock.schedule_interval(self.update, 1.0)

    def update(self, dt):
        if self.camera.texture:
            print("Here...")
            #self.camera.export_to_png('/storage/0123-4567/tmp.png')
            #frame = cv2.imread("/storage/0123-4567/tmp.png", cv2.IMREAD_COLOR)
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

            """
            texture = Texture.create(size=(640, 480))

            size = 640 * 480 * 3
            buf = [random.randint(0, 255) for x in range(size)]

            # then, convert the array to a ubyte string
            buf = bytes(buf)

            # then blit the buffer
            texture.blit_buffer(self.camera.texture.pixels, colorfmt='rgba', bufferfmt='ubyte')

            self.image.texture = texture
            """

    def on_texture_change(self, camera):
        if self.is_process_ongoing or platform == "android":
            nparr = np.frombuffer(camera.texture.pixels, np.uint8).reshape((480, 640, 4))
            nparr = nparr.copy()
            nparr.setflags(write=1)

            if self.is_process_ongoing:
                nparr[:,0:10,1::2] = 255 # left border [ : => select all lines,0:10 => select columns 0 to 10,1::2 => select nested array and set element 1 and 3 to 255 ( R G B A => G and A = 255)]
                nparr[:,-10:,1::2] = 255 # right border
                nparr[0:10,:,1::2] = 255 # upper border
                nparr[-10:,:,1::2] = 255 # bottom border

            if platform != "android":
                nparr = nparr[::-1]
            else:
                nparr = cv2.rotate(nparr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
            image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.camera.texture = texture = image_texture
            self.camera.texture_size = list(image_texture.size)
            self.camera.canvas.ask_update()
            #cv2.imshow("Halo", nparr)
            #self.is_process_ongoing = False
        else:
            self.camera.texture = texture = camera.texture
            self.camera.texture_size = list(texture.size)
            self.camera.canvas.ask_update()

    """
    def camera_on_load(self):
        print("camera_on_load")
    """

    def camera_click(self, obj, args):
        if args.pos[0] >= self.camera.pos[0] and args.pos[0] <= self.camera.size[0]+self.camera.pos[0] and args.pos[1] >= self.camera.pos[1] and args.pos[1] <= self.camera.size[1]+self.camera.pos[1]:
            print("Mouse coords are: ("+str(args.pos[0])+", "+str(args.pos[1])+")")
            print("Clickable image area is from ("+str(self.camera.pos[0])+ ", "+str(self.camera.pos[1])+")"+" to ("+str(self.camera.size[0])+", "+str(self.camera.size[1])+")")
            if self.boundingBox_coords[0] == (-1, -1):
                self.boundingBox_coords[0] = (int(args.pos[0]-self.camera.pos[0]), int(args.pos[1]))
            elif self.boundingBox_coords[1] == (-1, -1):
                self.boundingBox_coords[1] = (int(args.pos[0]-self.camera.pos[0]), int(args.pos[1]))
                nparr = np.zeros((480, 640, 4), dtype=np.uint8)
                nparr[self.boundingBox_coords[0][1]:self.boundingBox_coords[1][1],self.boundingBox_coords[0][0]:self.boundingBox_coords[0][0]+10,1::2] = 255
                nparr[self.boundingBox_coords[0][1]:self.boundingBox_coords[1][1],self.boundingBox_coords[1][0]-10:self.boundingBox_coords[1][0],1::2] = 255
                nparr[self.boundingBox_coords[1][1]-10:self.boundingBox_coords[1][1],self.boundingBox_coords[0][0]:self.boundingBox_coords[1][0],1::2] = 255
                nparr[self.boundingBox_coords[0][1]:self.boundingBox_coords[0][1]+10,self.boundingBox_coords[0][0]:self.boundingBox_coords[1][0],1::2] = 255


                image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
                image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
                self.boundingBox.texture = image_texture
            else:
                self.boundingBox_coords = [(-1, -1), (-1, -1)]
                print(self.boundingBox_coords)
                image_texture = Texture.create(size=(self.nparr.shape[1], self.nparr.shape[0]), colorfmt='rgba')
                image_texture.blit_buffer(self.nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
                self.boundingBox.texture = image_texture


    def update_metrics(self):
        self.boxLayout.window_size_x_label.text = "Window x size: "+str(Window.size[0])
        self.boxLayout.window_size_y_label.text = "Window y size: "+str(Window.size[1])
        self.boxLayout.camera_pos_x_label.text = "Camera x pos: "+str(self.camera.pos[0])
        self.boxLayout.camera_pos_y_label.text = "Camera y pos: "+str(self.camera.pos[1])
        self.boxLayout.camera_size_x_label.text = "Camera x size: "+str(self.camera.size[0])
        self.boxLayout.camera_size_y_label.text = "Camera y size: "+str(self.camera.size[1])

    def image_grow(self, obj):
        self.camera.size[0] = self.camera.size[0]+1
        self.camera.size[1] = self.camera.size[1]+1
        self.update_metrics()
        if self.plus_button.state == 'down':
            Clock.schedule_once(self.image_grow)

    def image_shrink(self, obj):
        self.camera.size[0] = self.camera.size[0]-1
        self.camera.size[1] = self.camera.size[1]-1
        self.update_metrics()
        if self.minus_button.state == 'down':
            Clock.schedule_once(self.image_shrink)

    def move_left(self, obj):
        self.camera.pos[0] = self.camera.pos[0] - 1
        self.update_metrics()
        if self.move_left_button.state == 'down':
            Clock.schedule_once(self.move_left)

    def move_right(self, obj):
        self.camera.pos[0] = self.camera.pos[0] + 1
        self.update_metrics()
        if self.move_right_button.state == 'down':
            Clock.schedule_once(self.move_right)

    def move_up(self, obj):
        self.camera.pos[1] = self.camera.pos[1] + 1
        self.update_metrics()
        if self.move_up_button.state == 'down':
            Clock.schedule_once(self.move_up)

    def move_down(self, obj):
        self.camera.pos[1] = self.camera.pos[1] - 1
        self.update_metrics()
        if self.move_down_button.state == 'down':
            Clock.schedule_once(self.move_down)

    def button_pressed_2(self, obj):
        with self.camera.canvas.before:
            PushMatrix()
            Rotate(origin=self.center, angle=45)
        with self.camera.canvas.after:
            PopMatrix()

    def image_upload_thread(self, pixels, size):
        self.is_process_ongoing = True

        #texture = self.camera.texture
        #size=texture.size
        #pixels = texture.pixels


        nparr = np.frombuffer(pixels, np.uint8).reshape((size[1], size[0], 4))

        #print(nparr.shape)
        img = cv2.cvtColor(nparr, cv2.COLOR_RGBA2BGR)
        #print(texture)
        #print(size)
        #print(img_gray)

        retval, buffer = cv2.imencode('.jpg', img)

        # Upload base64 jpg file to file
        """
        file = open("base64.txt", "w")
        file.write(str(b64encode(buffer)))
        file.close()
        """
        """
        upload_result = self.trainer.create_images_from_data(project_id, image_data=buffer, tag_ids=["raspberry"])
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
        """


        url = ENDPOINT+'customvision/v3.3/training/projects/'+project_id+'/images'

        #use the 'headers' parameter to set the HTTP headers:
        x = requests.post(url, data = bytes(buffer), headers = {"Content-Type": "application/octet-stream", "Training-Key": training_key})

        print(x.content)

        #self.camera.log = not self.camera.log
        self.is_process_ongoing = False

    def button_pressed(self, obj):
        #print(self.ids.camera)
        thread.start_new_thread( self.image_upload_thread, (self.camera.texture.pixels, self.camera.texture.size) )
        #x = threading.Thread(target=self.image_upload_thread, args=(self.camera.texture.pixels, self.camera.texture.size))
        #x.start()
        #x.join()



    def on_window_resize(self, window, width, height):

        self.update_metrics()

        self.boxLayout.pos = (0, dp(200))
        self.boxLayout.size = (Window.size[0]/2, Window.size[1]*9/10-dp(200))

        for item in self.boxLayout.canvas.children:
            if isinstance(item, Rectangle):
                item.pos = self.boxLayout.pos
                item.size = self.boxLayout.size

        self.plus_button.size = (dp(100), dp(100))
        self.plus_button.pos = (Window.size[0]-dp(100), dp(100))

        self.minus_button.size = (dp(100), dp(100))
        self.minus_button.pos= (Window.size[0]-dp(100), 0)

        self.move_left_button.size = (dp(100), dp(100))
        self.move_left_button.pos = (0, 0)

        self.move_right_button.size = (dp(100), dp(100))
        self.move_right_button.pos = (dp(200), 0)

        self.move_up_button.size = (dp(100), dp(100))
        self.move_up_button.pos = (dp(100), dp(100))

        self.move_down_button.size = (dp(100), dp(100))
        self.move_down_button.pos = (dp(100), 0)


        self.button.size = (width, height/10)
        self.button.pos = (0, height-height/10)

        self.camera.size = (640/480*height*9/10, height*9/10)
        self.camera.pos = ((width-self.camera.size[0])/2, 0)

        self.boundingBox.size = self.camera.size
        self.boundingBox.pos = self.camera.pos

        """
        for item in self.camera.canvas.before.children:
            if isinstance(item, Rectangle):
                item.size = self.camera.size
        """



class MainApp(App):
    def build(self):
        return Main()

if __name__ == "__main__":
    #CameraRead().start()
    MainApp().run()
