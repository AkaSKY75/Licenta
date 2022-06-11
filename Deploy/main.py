from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.pagelayout import PageLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.animation import Animation
from kivy.clock import mainthread
from kivy.uix.checkbox import CheckBox
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.utils import platform
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.graphics import Rotate, PushMatrix, PopMatrix, Translate, Color, Rectangle, RoundedRectangle, BindTexture, Line, Ellipse, InstructionGroup
from kivy.clock import Clock
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.metrics import dp
from os import listdir, getcwd
from os.path import isfile, join
from functools import partial
#from kivy.graphics.texture import Texture
from base64 import b64encode
from datetime import datetime

import cv2
import io
import numpy as np
import threading
import json
#import _thread as thread
import time
import random
import requests

#from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
#from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
#from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
#from msrest.authentication import ApiKeyCredentials
import os, uuid

#ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
#training_key = "5887d793471f48dc8417e84c699d561d"
#prediction_key = "5887d793471f48dc8417e84c699d561d"
#prediction_resource_id = "/subscriptions/2c0c2800-5672-4899-963d-d9c6e8984e61/resourceGroups/appsvc_windows_centralus/providers/Microsoft.CognitiveServices/accounts/Licenta"
#project_id = "2fbc8c69-c0c5-42f7-83c1-63e91c12751b"

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
training_key = "5887d793471f48dc8417e84c699d561d"
prediction_key = "5887d793471f48dc8417e84c699d561d"
prediction_resource_id = "/subscriptions/2c0c2800-5672-4899-963d-d9c6e8984e61/resourceGroups/appsvc_windows_centralus/providers/Microsoft.CognitiveServices/accounts/Licenta"
project_id = "1aa5fcfe-5fa4-494d-96a1-2208b8c148e7"
version = "v3.4-preview"

colors = [(0, 72, 186), (176, 191, 26), (124, 185, 232), (178, 132, 190),
        (114, 160, 193), (219, 45, 67), (196, 98, 16), (159, 43, 104), (59, 122, 87),
        (255, 191, 0), (61, 220, 132), (205, 149, 117), (102, 93, 30), (132, 27, 45),
        (141, 182, 0), (0, 255, 255)]

class Main(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tags = {}
        self.iterations = {"selected": None}


        self.boundingBox_coords = [(-1, -1), (-1, -1)]
        self.probabilities = {"class":"", "probability":.0,"left":.0,"top":.0,"width":.0,"height":.0, "startx":0, "starty":0, "stopx":0, "stopy":0}
        self.boundingBox_label = None

        self.process_state = 0

        self.button = Button(text="Label", on_press=self.button_pressed, size_hint=(1, .1), pos_hint={"x": 0, "y": .9})
        self.add_widget(self.button)

        self.train_button = Button(text="Train", on_press=self.train_button_pressed, size_hint=(1, .1), pos_hint={"x": 0, "y": 0})
        self.add_widget(self.train_button)

        self.camera = Camera(index=0, resolution=(640, 480), play=True, allow_stretch=True, pos_hint={"x":0,"y":.1}, size_hint_y=.8)
        self.camera._camera.unbind(on_texture=self.camera.on_tex)
        self.camera._camera.bind(on_texture=self.on_texture_change)

        self.training_popup = None
        self.ask_ok_detection_popup = None
        self.coords = None

        self.boundingBox = Image(allow_stretch=True, pos_hint=self.camera.pos_hint, size_hint=self.camera.size_hint)
        self.boundingBox.borderWidth = 10
        nparr = np.zeros((self.camera.resolution[1], self.camera.resolution[0], 4), dtype=np.uint8)
        #nparr[:,:] = [0, 255, 0, 127]
        self.blank_transparent_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
        self.blank_transparent_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.boundingBox.texture = self.blank_transparent_texture
        #self.clear_image()

        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.INTERNET, Permission.WRITE_EXTERNAL_STORAGE])

        self.add_widget(self.camera)

        self.add_widget(self.boundingBox)

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/iterations'

        response = requests.get(url, headers = {"Training-Key": training_key})

        response = json.loads(response.content)

        for i in response:
            if i["status"] == "Completed":
                self.iterations[i["id"]] = {"name": i["name"], "lastModified": datetime.strptime(i["lastModified"], '%Y-%m-%dT%H:%M:%S.%fZ'), "tags": []}

                url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/iterations/'+i["id"]+'/performance'
                response_tags_from_iterations = requests.get(url, headers = {"Training-Key": training_key})
                response_tags_from_iterations = json.loads(response_tags_from_iterations.content)
                for j in response_tags_from_iterations["perTagPerformance"]:
                    self.iterations[i["id"]]["tags"].append(j["id"])

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/tags'

        response = requests.get(url, headers = {"Training-Key": training_key})

        response = json.loads(response.content)

        for i in response:
            self.tags[i["id"]] = {"name": i["name"], "selected": True}
        self.object_detection_thread = threading.Thread(target=self.detect_objects_thread)
        self.object_detection_thread.daemon = True
        self.object_detection_thread.start()

    @mainthread
    def create_boundingBox(self, startx=-1, starty=-1, stopx=-1, stopy=-1):
        if self.process_state == 1 or self.process_state == 3:
            startx,starty,stopx,stopy = self.get_calculated_coords()
        elif self.process_state == 10:
            if self.boundingBox_label is not None:
                self.remove_widget(self.boundingBox_label)
            ratio = self.camera.resolution[0]/self.camera.resolution[1]
            width = ratio*self.camera.size[1]
            camera_blank_space = (self.camera.size[0]-width)/2
            original_coords = (int(startx/self.camera.resolution[0]*width)+camera_blank_space, int(stopy/self.camera.resolution[1]*self.camera.size[1])+self.camera.pos[1])
            self.boundingBox_label = Label(text=self.probabilities["class"]+"\n{:.2f}%".format(self.probabilities["probability"]*100), color=(1, 1, 1, 1), size_hint=(None, None), size=(dp(150), dp(50)), pos=(original_coords[0], original_coords[1]))
            with self.boundingBox_label.canvas.before:
                Color(0, 1, 0, 0.5)
                Rectangle(pos=self.boundingBox_label.pos, size=self.boundingBox_label.size)
            self.add_widget(self.boundingBox_label)

        if self.process_state == 1 or self.process_state == 3 or self.process_state == 10:
            nparr = np.zeros((480, 640, 4), dtype=np.uint8)
            nparr[starty:stopy,startx:startx+self.boundingBox.borderWidth,1::2] = 255
            nparr[starty:stopy,stopx-self.boundingBox.borderWidth:stopx,1::2] = 255
            nparr[stopy-self.boundingBox.borderWidth:stopy,startx:stopx,1::2] = 255
            nparr[starty:starty+self.boundingBox.borderWidth,startx:stopx,1::2] = 255

            image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
            image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.boundingBox.texture = image_texture
            self.boundingBox.canvas.ask_update()

    @mainthread
    def clear_image(self):
        self.boundingBox.texture = self.blank_transparent_texture
        self.boundingBox.canvas.ask_update()

    def on_texture_change(self, camera):
        if self.process_state == 14 or ((self.process_state == 0 or self.process_state == 10) and platform == "android"):
            nparr = np.frombuffer(camera.texture.pixels, np.uint8).reshape((480, 640, 4))
            nparr = nparr.copy()
            nparr.setflags(write=1)

            if self.process_state == 14:
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
            self.camera_thread_params = (texture.pixels, texture.size)

        if (self.process_state == 0 or self.process_state == 10) and platform != "android":
            if platform != "android":
                self.camera.texture = texture = camera.texture
                self.camera.texture_size = list(texture.size)
                self.camera.canvas.ask_update()
                self.camera_thread_params = (texture.pixels, texture.size)

    def change_state(self, state):
        valid_state = False
        # This state should be set when process is in label selection (state 2)
        if state == 0:
            if self.process_state == 2: #sanity check, we can go to state 0 only from state 2 or 3
                self.button.text = "Label"
                self.camera.play = True
                valid_state = True
            elif self.process_state == 5:
                self.button.text = "Label"
                self.boundingBox_coords = [(-1, -1), (-1, -1)]
                self.camera.play = True
                valid_state = True
            elif self.process_state == 7 or self.process_state == 8 or self.process_state == 9 or self.process_state == 11 or self.process_state == 10:
                self.button.text = "Label"
                self.train_button.text = "Train"
                self.training_popup.clear_widgets()
                self.remove_widget(self.training_popup)
                valid_state = True
            elif self.process_state == 14:
                valid_state = True
        elif state == 1:
            if self.process_state == 2:
                self.button.text = "Upload"
                valid_state = True
        elif state == 2:
            if self.process_state == 0 or self.process_state == 10:
                self.remove_ok_detection_popup()
                self.clear_image()
                if self.boundingBox_label is not None:
                    self.remove_widget(self.boundingBox_label)
                self.button.text = "Cancel"
                self.camera.play = False
                valid_state = True
            elif self.process_state == 1:
                self.button.text = "Cancel"
                valid_state = True
            elif self.process_state == 4:
                self.clear_image()
                self.remove_widget(self.boundingBox_input)
                self.boundingBox_coords = [(-1, -1), (-1, -1)]
                self.button.text = "Cancel"
                valid_state = True
        elif state == 3:
            if self.process_state == 1:
                valid_state = True
        elif state == 4:
            if self.process_state == 3:
                valid_state = True
        elif state == 5:
            if self.process_state == 4:
                self.clear_image()
                self.button.text = "Uploading..."
                self.boundingBox_input_text = self.boundingBox_input.text
                self.remove_widget(self.boundingBox_input)
            #thread.start_new_thread( self.image_upload_thread, (self.camera.texture.pixels, self.camera.texture.size) )
                x = threading.Thread(target=self.image_upload_thread, args=(self.camera.texture.pixels, self.camera.texture.size))
                x.daemon = True
                x.start()
                #self.camera.play = True
                valid_state = True
        elif state == 7:
            if self.process_state == 0 or self.process_state == 10:
                self.remove_ok_detection_popup()
                self.button.text = "Cancel"
                self.train_button.text = "New training"
                self.clear_image()
                if self.boundingBox_label is not None:
                    self.remove_widget(self.boundingBox_label)
                self.training_popup = BoxLayout(pos_hint={"x":.125, "y":.125}, size_hint=(.75, .75), orientation="vertical")
                self.training_popup.id = "training_popup"
                self.training_popup.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)
                self.training_popup.label = Label(text="Select iteration", color=(255, 255, 255, 1), pos=(self.training_popup.pos[0], self.training_popup.pos[1]+self.training_popup.size[1]-dp(40)), size=(self.training_popup.size[0], dp(40)))

                self.training_popup.scrollview_iterations = ScrollView()
                with self.training_popup.canvas.before:
                    Color(233, 236, 239, 1)
                    Rectangle(pos=(self.training_popup.pos[0], self.training_popup.pos[1]), size=(self.training_popup.size[0], self.training_popup.size[1]))
                self.training_popup.scrollview_iterations.gridLayout = GridLayout(cols=1, size_hint_y=None)
                self.training_popup.scrollview_iterations.gridLayout.bind(minimum_height=self.training_popup.scrollview_iterations.gridLayout.setter('height'))

                self.add_widget(self.training_popup)
                self.training_popup.add_widget(self.training_popup.scrollview_iterations)
                self.training_popup.scrollview_iterations.add_widget(self.training_popup.scrollview_iterations.gridLayout)

                for i in self.iterations.keys():
                    if i != "selected":
                        stackLayout = StackLayout(size_hint_y=None)
                        stackLayout.id = i
                        if self.iterations["selected"] == i:
                            stackLayout.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)

                        #print(boxLayout.id)
                        #boxLayout.bind(on_touch_down=self.iteration_select)
                        stackLayout.bind(on_touch_down=self.iteration_select)
                        self.training_popup.scrollview_iterations.gridLayout.add_widget(stackLayout)
                        label = Label(text=self.iterations[i]["name"], color=(255, 255, 255, 1), size_hint=(.3, None), height=dp(20))
                        stackLayout.add_widget(label)
                        label = Label(text=self.iterations[i]["lastModified"].strftime("%b %d %Y at %H:%M:%S"), color=(255, 255, 255, 1), size_hint=(.3, None), height=dp(20))
                        stackLayout.add_widget(label)
                        gridLayout = GridLayout(cols=1, size_hint_y=None, size_hint_x=.3)
                        gridLayout.bind(minimum_height=gridLayout.setter('height'))
                        stackLayout.bind(minimum_height=stackLayout.setter('height'))
                        stackLayout.add_widget(gridLayout)

                        for j in self.iterations[i]["tags"]:
                            label = Label(text=self.tags[j]["name"], color=(255, 255, 255, 1), size_hint_y=None, height=dp(20))
                            gridLayout.add_widget(label)


                #x = threading.Thread(target=self.train_thread, args=())
                #x.daemon = True
                #x.start()
                valid_state = True
            elif self.process_state == 8:
                self.change_state(0)
                self.change_state(7)
        elif state == 8:
            if self.process_state == 7:
                self.button.text = "Cancel"
                self.train_button.text = "Select iteration"
                self.training_popup.clear_widgets()
                self.training_popup.scrollview_tags = ScrollView(pos_hint={"x":0.05}, size_hint=(0.9, .9))#ScrollView()
                self.training_popup.scrollview_images = ScrollView(pos_hint={"x":0.05}, size_hint=(0.9, .9))#ScrollView()

                self.training_popup.next_image = Image(pos_hint={"x": .9}, size_hint=(.1, .1), source="arrow_right.png")
                self.training_popup.next_image.bind(on_touch_down=self.training_popup_next_page)
                self.training_popup.previous_image = Image(size_hint=(.1, .1), source="arrow_left.png")
                self.training_popup.previous_image.bind(on_touch_down=self.training_popup_previous_page)

                self.training_popup.scrollview_tags.gridLayout = GridLayout(cols=1, size_hint_y=None)
                self.training_popup.scrollview_tags.gridLayout.bind(minimum_height=self.training_popup.scrollview_tags.gridLayout.setter('height'))

                self.training_popup.scrollview_images.gridLayout = GridLayout(cols=10, size_hint_y=None, spacing=(10, 10))
                self.training_popup.scrollview_images.gridLayout.bind(minimum_height=self.training_popup.scrollview_images.gridLayout.setter('height'))

                self.training_popup.add_widget(self.training_popup.scrollview_tags)
                self.training_popup.scrollview_tags.add_widget(self.training_popup.scrollview_tags.gridLayout)
                self.training_popup.scrollview_images.add_widget(self.training_popup.scrollview_images.gridLayout)
                self.training_popup.add_widget(self.training_popup.next_image)

                for i in self.tags.keys():
                    boxLayout = BoxLayout(size_hint_y=None, size=(1, dp(100)))
                    #print(boxLayout.id)
                    #boxLayout.bind(on_touch_down=self.iteration_select)
                    self.training_popup.scrollview_tags.gridLayout.add_widget(boxLayout)
                    label = Label(text=self.tags[i]["name"], color=(255, 255, 255, 1))
                    boxLayout.add_widget(label)
                    checkbox = CheckBox(active=True)
                    checkbox.id = i
                    checkbox.bind(active=self.tag_select)
                    boxLayout.add_widget(checkbox)

                valid_state = True

            elif self.process_state == 9 or self.process_state == 11:
                self.train_button.text = "Select iteration"
                self.training_popup.remove_widget(self.training_popup.scrollview_images)
                self.training_popup.remove_widget(self.training_popup.previous_image)
                self.training_popup.add_widget(self.training_popup.scrollview_tags)
                self.training_popup.add_widget(self.training_popup.next_image)
                valid_state = True
        elif state == 9:
            if self.process_state == 8:
                self.train_button.text = "Train"
                self.training_popup.scrollview_images.gridLayout.clear_widgets()
                self.training_popup.remove_widget(self.training_popup.scrollview_tags)
                self.training_popup.remove_widget(self.training_popup.next_image)
                self.training_popup.add_widget(self.training_popup.scrollview_images)
                self.training_popup.add_widget(self.training_popup.previous_image)
                x = threading.Thread(target=self.image_load_thread)
                x.daemon = True
                x.start()
                valid_state = True
        elif state == 10:
            if self.process_state == 2 or self.process_state == 7 or self.process_state == 8 or self.process_state == 9:
                self.change_state(0)
                x = threading.Thread(target=self.image_load_thread)
                x.daemon = True
                x.start()
                valid_state = True
        elif state == 11:
            if self.process_state == 9:
                for i in self.images:
                    nparr = np.zeros((i["height"], i["width"], 3), dtype=np.uint8)
                    nparr[:,:] = [41, 44, 52]
                    image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgb')
                    image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                    image = Image(texture=image_texture, size_hint_y = None, allow_stretch=True)
                    image.bind(on_touch_down=self.preview_image_popup)
                    image.id = i
                    if self.training_popup is not None:
                        if self.training_popup.scrollview_images is not None:
                            self.training_popup.scrollview_images.gridLayout.add_widget(image)
                    if self.process_state == 9:
                        valid_state = True
            elif self.process_state == 12 or self.process_state == 13:
                self.image_preview.clear_widgets()
                self.remove_widget(self.image_preview)
                valid_state = True
        elif state == 12:
            if self.process_state == 11:
                valid_state = True
            elif self.process_state == 13:
                Animation(size_hint_x=.0, duration=.5).start(self.image_preview.meta)
                self.image_preview.meta_toggle_button.source = "arrow_right2.jpg"
                valid_state = True
        elif state == 13:
            if self.process_state == 12:
                Animation(size_hint_x=.25, duration=.5).start(self.image_preview.meta)
                self.image_preview.meta_toggle_button.source = "arrow_left2.jpg"
                valid_state = True
        elif state == 14:
            if self.process_state == 9 or self.process_state == 11:
                self.change_state(0)
                x = threading.Thread(target=self.train_thread)
                x.daemon = True
                x.start()
                valid_state = True
        elif state == 15:
            if self.process_state == 10:
                self.remove_ok_detection_popup()
                self.remove_widget(self.boundingBox_label)
                img = cv2.imread(self.probabilities["class"]+".jpeg")
                ratio = self.camera.resolution[0]/self.camera.resolution[1]
                width = ratio*self.camera.size[1]
                camera_blank_space = (self.camera.size[0]-width)/2
                startx, starty, stopx, stopy = self.probabilities["startx"], self.probabilities["starty"], self.probabilities["stopx"], self.probabilities["stopy"]

                original_coords = (int(startx/self.camera.resolution[0]*width+camera_blank_space), int(starty/self.camera.resolution[1]*self.camera.size[1]+self.camera.pos[1]), int(stopx/self.camera.resolution[0]*width+camera_blank_space), int(stopy/self.camera.resolution[1]*self.camera.size[1]+self.camera.pos[1]))

                nparr = np.frombuffer(self.camera.texture.pixels, dtype=np.uint8).reshape(self.camera.texture.size[1], self.camera.texture.size[0], 4)
                nparr = nparr[::-1].copy()

                nparr_crop = nparr[starty+self.boundingBox.borderWidth:stopy-self.boundingBox.borderWidth, startx+self.boundingBox.borderWidth:stopx-self.boundingBox.borderWidth]

                image_texture = Texture.create(size=(nparr_crop.shape[1], nparr_crop.shape[0]), colorfmt="rgba")
                image_texture.blit_buffer(nparr_crop.tobytes(), colorfmt="rgba", bufferfmt='ubyte')

                self.detected_object_image = Image(pos_hint={"x":original_coords[0]/self.width, "y":original_coords[1]/self.height}, size_hint=((original_coords[2]-original_coords[0])/self.width, (original_coords[3]-original_coords[1])/self.height), texture=image_texture, allow_stretch=True, keep_ratio=False)

                ksize = (30, 30)
                nparr = cv2.blur(nparr, ksize, cv2.BORDER_DEFAULT)

                image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
                image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')

                self.boundingBox.texture = image_texture

                self.boundingBox.canvas.opacity = 0
                Animation(opacity=1, duration=1).start(self.boundingBox.canvas)

                image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(img[::-1].tobytes(), colorfmt='bgr', bufferfmt='ubyte')

                self.add_widget(self.detected_object_image)

                Animation(pos_hint={"x":.5, "y":.5}, size_hint=(.5, .4), keep_ratio=True, duration=2).start(self.detected_object_image)

                self.detected_object_image.texture = image_texture

                valid_state = True
        if valid_state == True:
            self.process_state = state

    @mainthread
    def create_ok_detection_popup(self):
        if self.process_state == 10:
            self.ask_ok_detection_popup = FloatLayout(pos_hint={"x":(self.width-dp(300))/(2*self.width), "y": 0.11}, size_hint=(None, None), height=dp(150), width=dp(300))
            self.ask_ok_detection_popup.id = "ask_ok_detection_popup"
            self.ask_ok_detection_popup.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)
            self.add_widget(self.ask_ok_detection_popup)
            label = Label(text="Application detected " + self.probabilities["class"] + ".", color=(0, 0, 0, 1), size_hint=(1, .5), pos_hint={"x": 0, "y": .5})
            self.ask_ok_detection_popup.add_widget(label)
            button = Button(text="Show specifications", pos_hint={"x": 0.125, "y":.1}, size_hint=(.75, .25), background_color=(0, 0, 0, 0), background_normal="", on_touch_down=self.specs_button_pressed, on_touch_up=self.specs_button_released)
            button.id = "specs_button"
            button.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)
            self.ask_ok_detection_popup.add_widget(button)


    def specs_button_pressed(self, obj, touch):
        obj.canvas.before.clear()
        with obj.canvas.before:
            Color(48/255, 84/255, 150/255, 1)
            #Line(width=1, rounded_rectangle=(obj.pos[0]-1, obj.pos[1]-1, obj.size[0]+1, obj.size[1]+1, 25))
            RoundedRectangle(pos=obj.pos, size=obj.size, radius=[25])


    def specs_button_released(self, obj, touch):
        obj.canvas.before.clear()
        with obj.canvas.before:
            Color(48/255, 84/255, 150/255, 0.75)
            #Line(width=1, rounded_rectangle=(obj.pos[0]-1, obj.pos[1]-1, obj.size[0]+1, obj.size[1]+1, 25))
            RoundedRectangle(pos=obj.pos, size=obj.size, radius=[25])
        if obj.collide_point(*touch.pos):
            self.change_state(15)

    @mainthread
    def remove_ok_detection_popup(self):
        if self.ask_ok_detection_popup is not None:
            self.ask_ok_detection_popup.clear_widgets()
            self.remove_widget(self.ask_ok_detection_popup)



    def on_widget_pos_size(self, obj, largs):
        obj.canvas.before.clear()
        if obj.id == "training_popup":
            with obj.canvas.before:
                Color(233, 236, 239, 1)
                #Line(width=10, rounded_rectangle=(self.training_popup.pos[0], self.training_popup.pos[1], self.training_popup.size[0], self.training_popup.size[1], 25))
                Rectangle(pos=obj.pos, size=obj.size)
        elif obj.id == "image_preview":
            with obj.canvas.before:
                Color(0, 0, 0, 0.9)
                #Line(width=10, rounded_rectangle=(self.training_popup.pos[0], self.training_popup.pos[1], self.training_popup.size[0], self.training_popup.size[1], 25))
                Rectangle(pos=obj.pos, size=obj.size)
        elif obj.id == "image_preview_meta":
            with obj.canvas.before:
                Color(233/255, 236/255, 239/255, 0.75)
                #Line(width=10, rounded_rectangle=(self.training_popup.pos[0], self.training_popup.pos[1], self.training_popup.size[0], self.training_popup.size[1], 25))
                Rectangle(pos=obj.pos, size=obj.size)
        elif obj.id == "ask_ok_detection_popup":
            with obj.canvas.before:
                Color(233/255, 236/255, 239/255, 0.75)
                #Line(width=1, rounded_rectangle=(obj.pos[0]-1, obj.pos[1]-1, obj.size[0]+1, obj.size[1]+1, 25))
                RoundedRectangle(pos=obj.pos, size=obj.size, radius=[(25, 25), (25, 25), (25, 25), (25, 25)])
        elif obj.id == "specs_button":
            with obj.canvas.before:
                Color(48/255, 84/255, 150/255, 0.75)
                #Line(width=1, rounded_rectangle=(obj.pos[0]-1, obj.pos[1]-1, obj.size[0]+1, obj.size[1]+1, 25))
                RoundedRectangle(pos=obj.pos, size=obj.size, radius=[25])
        else:
            with obj.canvas.before:
                Color(0, 255/255, 0, 0.5)
                #boxLayout.line = Line(points=[0, 10, 100, 0], width=10)
                Rectangle(pos=obj.pos, size=obj.size)

    def tag_select(self, obj, state):
        self.tags[obj.id]["selected"] = state

    def training_popup_previous_page(self, obj, touch):
        if (self.process_state == 9 or self.process_state == 11) and obj.collide_point(*touch.pos):
            self.change_state(8)

    def training_popup_next_page(self, obj, touch):
        if self.process_state == 8 and obj.collide_point(*touch.pos):
            self.change_state(9)

    def iteration_select(self, obj, touch):
        if obj.collide_point(*touch.pos):
            for i in self.training_popup.scrollview_iterations.gridLayout.children:
                i.canvas.before.clear()
            if self.iterations["selected"] == None:
                self.iterations["selected"] = obj.id
                with obj.canvas.before:
                    Color(0, 255, 0, 0.5)
                    #boxLayout.line = Line(points=[0, 10, 100, 0], width=10)
                    Rectangle(pos=obj.pos, size=obj.size)
            else:
                self.iterations["selected"] = None


    def canvas_update(self, obj, *args):
        obj.rect.pos = obj.pos
        obj.rect.size = obj.size

    def get_calculated_coords(self):
        startx = self.boundingBox_coords[0][0]
        starty = self.boundingBox_coords[0][1]

        stopx = self.boundingBox_coords[1][0]
        stopy = self.boundingBox_coords[1][1]

        if startx > stopx:
            aux = startx
            startx = stopx
            stopx = aux

        if starty > stopy:
            aux = starty
            starty = stopy
            stopy = aux

        return startx,starty,stopx,stopy

    def on_touch_up(self, touch):
        if self.process_state == 1:
            self.change_state(2)
        elif self.process_state == 3:

            ratio = self.camera.resolution[0]/self.camera.resolution[1]

            width = ratio*self.camera.size[1]

            camera_blank_space = (self.camera.size[0]-width)/2

            startx,starty,stopx,stopy = self.get_calculated_coords()

            original_coords = (int(startx/self.camera.resolution[0]*width)+camera_blank_space, int(stopy/self.camera.resolution[1]*self.camera.size[1])+self.camera.pos[1])
            self.boundingBox_input = TextInput(text="Place label here", size_hint=(None, None), pos=(original_coords[0], original_coords[1]), size=(dp(100), dp(50)), background_color=(0, 255, 0, 0.5))
            self.add_widget(self.boundingBox_input)
            self.change_state(4)
        return super(Main, self).on_touch_up(touch)

    def on_touch_move(self, touch):
        ratio = self.camera.resolution[0]/self.camera.resolution[1]

        width = ratio*self.camera.size[1]

        camera_blank_space = (self.camera.size[0]-width)/2
        if (self.process_state == 1 or self.process_state == 3) and touch.x >= camera_blank_space and touch.x <= self.camera.size[0]-camera_blank_space and touch.y >= self.camera.pos[1] and touch.y <= self.camera.size[1]+self.camera.pos[1]:
            original_coords = (int((touch.x-camera_blank_space)/width*self.camera.resolution[0]), int((touch.y-self.camera.pos[1])/self.camera.size[1]*self.camera.resolution[1]))
            self.boundingBox_coords[1] = original_coords
            #self.clear_image()
            self.create_boundingBox()
            self.change_state(3)
        return super(Main, self).on_touch_move(touch)

    def on_touch_down(self, touch):
        ratio = self.camera.resolution[0]/self.camera.resolution[1]

        width = ratio*self.camera.size[1]

        camera_blank_space = (self.camera.size[0]-width)/2
        if touch.x >= camera_blank_space and touch.x <= self.camera.size[0]-camera_blank_space and touch.y >= self.camera.pos[1] and touch.y <= self.camera.size[1]+self.camera.pos[1]:
            if self.process_state == 2 or self.process_state == 1 or (self.process_state == 4 and (touch.x < self.boundingBox_input.pos[0] or touch.x > self.boundingBox_input.pos[0]+self.boundingBox_input.size[0] or touch.y < self.boundingBox_input.pos[1] or touch.y > self.boundingBox_input.pos[1]+self.boundingBox_input.size[1])):
                # Formula that calculates actual coordinates of resized image
                """
                newX = (currentX/currentWidth)*newWidth
                newY = (currentY/currentHeight)*newHeight

                currentX = (newX/newWidth)*currentWidth
                currentY = (newY/newHeight)*currentHeight
                """

                original_coords = (int((touch.x-camera_blank_space)/width*self.camera.resolution[0]), int((touch.y-self.camera.pos[1])/self.camera.size[1]*self.camera.resolution[1]))
                self.clear_image()
                if self.process_state == 2:
                    self.boundingBox_coords[0] = (original_coords[0], original_coords[1])
                    self.change_state(1)
                else:
                    self.change_state(2)
        return super(Main, self).on_touch_down(touch)

    def train_thread(self):

        tags = {"selectedTags": []}

        for i in self.tags.keys():
            if self.tags[i]["selected"] == True:
                tags["selectedTags"].append(i)
                #tags["selectedTags"].append({"tagId":i})

        #print(json.dumps(tags))

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/train?forceTrain=true'

        response = requests.post(url, data=json.dumps(tags), headers = {"Content-Type": "application/json", "Training-Key": training_key})

        iteration = json.loads(response.content)

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/iterations/'+iteration["id"]

        response = requests.get(url, headers = {"Training-Key": training_key})

        while json.loads(response.content)["status"] != "Completed":
            response = requests.get(url, headers = {"Training-Key": training_key})

        self.iterations[iteration["id"]] = {"name": iteration["name"], "lastModified": datetime.strptime(iteration["lastModified"], '%Y-%m-%dT%H:%M:%S.%fZ'), "tags": tags["selectedTags"]}

        self.change_state(0)

    def detect_objects_thread(self):
        while 1:
            if self.process_state == 10:
                #Clock.schedule_once(partial(self.get_camera_parameters,pixels,size))
                nparr = np.frombuffer(self.camera_thread_params[0], np.uint8).reshape((self.camera_thread_params[1][1], self.camera_thread_params[1][0], 4))
                retval, buffer = cv2.imencode('.png', nparr)
                url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/quicktest/image?iterationId='+self.iterations["selected"]+'&store=false'

                response = requests.post(url, data = bytes(buffer), headers = {"Content-Type": "application/octet-stream", "Training-Key": training_key})

                predictions = json.loads(response.content)

                image_id = predictions["id"]

                self.probabilities["class"] = ""
                self.probabilities["probability"] = .0
                self.probabilities["left"] = .0
                self.probabilities["top"] = .0
                self.probabilities["height"] = 0
                self.probabilities["startx"] = 0
                self.probabilities["starty"] = 0
                self.probabilities["stopx"] = 0
                self.probabilities["stopy"] = 0

                for p in predictions["predictions"]:
                    if p["probability"] > self.probabilities["probability"]:
                        self.probabilities["class"] = self.tags[p["tagId"]]["name"]
                        self.probabilities["probability"] = p["probability"]
                        self.probabilities["left"] = p["boundingBox"]["left"]
                        self.probabilities["top"] = p["boundingBox"]["top"]
                        self.probabilities["width"] = p["boundingBox"]["width"]
                        self.probabilities["height"] = p["boundingBox"]["height"]
                        self.probabilities["startx"] = int(self.probabilities["left"]*self.camera.resolution[0])
                        self.probabilities["stopx"] = int(self.probabilities["startx"] + self.probabilities["width"]*self.camera.resolution[0])
                        self.probabilities["stopy"] = int(self.camera.resolution[1] - self.probabilities["top"]*self.camera.resolution[1])
                        self.probabilities["starty"] = int(self.probabilities["stopy"] - self.probabilities["height"]*self.camera.resolution[1])

                if self.probabilities["probability"] > .0:
                    self.create_boundingBox(self.probabilities["startx"], self.probabilities["starty"], self.probabilities["stopx"], self.probabilities["stopy"])
                    self.remove_ok_detection_popup()
                    self.create_ok_detection_popup()

                url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/images?imageIds='+image_id

                #use the 'headers' parameter to set the HTTP headers:
                response = requests.delete(url, headers = {"Training-Key": training_key})
            time.sleep(1)

    @mainthread
    def preview_image_upload_texture(self, obj, nparr):
        image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(nparr[::-1].tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        obj.texture=image_texture

    def preview_image_popup_close(self, obj, touch):
        if (self.process_state == 12 or self.process_state == 13) and obj.collide_point(*touch.pos):
            self.change_state(11)

    def preview_image_popup_meta(self, obj, touch):
        if obj.collide_point(*touch.pos):
            if self.process_state == 12:
                self.change_state(13)
            elif self.process_state == 13:
                self.change_state(12)

    def preview_image_popup(self, obj, touch):
        if self.process_state == 11 and obj.collide_point(*touch.pos): # touch.is_double_touch
            self.image_preview = FloatLayout()
            #self.image_preview.InstructionGroup = InstructionGroup()
            self.image_preview.meta = BoxLayout(pos_hint={"x": 0.125}, size_hint=(.0, 1))
            self.image_preview.meta.id = "image_preview_meta"
            self.image_preview.meta.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)
            self.image_preview.meta.scrollView = ScrollView(size_hint_y=.9, pos_hint={"x": .1})
            #self.image_preview.meta = ScrollView()
            self.image_preview.meta.scrollView.stackLayout = StackLayout(size_hint_y=None)
            self.image_preview.meta.scrollView.stackLayout.bind(minimum_height=self.image_preview.meta.scrollView.stackLayout.setter('height'))
            boxLayout = BoxLayout(size_hint_y=None, height=dp(20), orientation="vertical")
            label = Label(text="Used tags for image", color=(255, 255, 255))
            self.image_preview.meta.scrollView.stackLayout.add_widget(boxLayout)
            boxLayout.add_widget(label)

            self.image_preview.image = Image(pos_hint={"x": 0.125}, size_hint=(0.75, 1), allow_stretch = True)
            texture = obj.texture
            i = 0
            for i in range(len(obj.id["tags"])):
                boxLayout = BoxLayout(size_hint_y=None, height=dp(20), orientation="vertical")
                label = Label(text=obj.id["tags"][i]["tagName"], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255, 1), size_hint_x=.5, pos_hint={"x":.125})
                self.image_preview.meta.scrollView.stackLayout.add_widget(boxLayout)
                boxLayout.add_widget(label)
                for j in obj.id["regions"]:
                    if j["tagId"] == obj.id["tags"][i]["tagId"]:
                        nparr = np.frombuffer(texture.pixels, dtype=np.uint8).reshape(texture.size[1], texture.size[0], 4)
                        nparr = nparr[::-1].copy()
                        startx = int(texture.size[0]*j["left"])-self.boundingBox.borderWidth
                        starty = int(texture.size[1]*j["top"])-self.boundingBox.borderWidth
                        stopx = int(startx+texture.size[0]*j["width"])+self.boundingBox.borderWidth
                        stopy = int(starty+texture.size[1]*j["height"])+self.boundingBox.borderWidth
                        #print(colors[i]+(255,))
                        nparr[starty:stopy,startx:startx+self.boundingBox.borderWidth] = colors[i]+(255,)
                        nparr[starty:stopy,stopx-self.boundingBox.borderWidth:stopx] = colors[i]+(255,)
                        nparr[starty:starty+self.boundingBox.borderWidth,startx:stopx] = colors[i]+(255,)
                        nparr[stopy-self.boundingBox.borderWidth:stopy,startx:stopx] = colors[i]+(255,)
                        texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
                        texture.blit_buffer(nparr[::-1].tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.image_preview.image.texture = texture
            self.image_preview.id = "image_preview"
            self.image_preview.bind(pos=self.on_widget_pos_size, size=self.on_widget_pos_size)
            self.image_preview.close_button = Image(pos_hint={"x":.9, "y":.9}, size_hint=(.1, .1), source="close_button.png")
            self.image_preview.close_button.bind(on_touch_up=self.preview_image_popup_close)

            self.image_preview.meta_toggle_button = Image(pos_hint={"x": 0.125}, size_hint=(0.1, 0.1), source="arrow_right2.jpg")
            self.image_preview.meta_toggle_button.bind(on_touch_up=self.preview_image_popup_meta)

            self.add_widget(self.image_preview)
            self.image_preview.add_widget(self.image_preview.close_button)
            self.image_preview.add_widget(self.image_preview.image)
            self.image_preview.add_widget(self.image_preview.meta)
            self.image_preview.meta.add_widget(self.image_preview.meta.scrollView)
            self.image_preview.meta.scrollView.add_widget(self.image_preview.meta.scrollView.stackLayout)
            self.image_preview.add_widget(self.image_preview.meta_toggle_button)
            self.change_state(12)


    @mainthread
    def add_image_preview_pattern(self):
        self.change_state(11)

    def image_load_thread(self):

        tags = ""

        for i in self.tags.keys():
            if self.tags[i]["selected"] == True:
                if tags == "":
                    tags = i
                else:
                    tags += ","+i

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/images?taggingStatus=Tagged&tagIds='+tags

        response = requests.get(url, headers = {"Training-Key": training_key})

        self.images = json.loads(response.content)

        self.add_image_preview_pattern()

        while self.process_state != 11:
            if self.process_state == 9:
                continue
            else:
                break

        i = 1
        length = 0
        if self.process_state == 11 or self.process_state == 12 or self.process_state == 13:
            length = len(self.training_popup.scrollview_images.gridLayout.children)+1

        while i in range(length):
            if self.process_state == 11 or self.process_state == 12 or self.process_state == 13:
                image = self.training_popup.scrollview_images.gridLayout.children[-i]
                binimage = requests.get(image.id["originalImageUri"]).content
                nparr = np.frombuffer(binimage, np.uint8)
                nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                nparr = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
                self.preview_image_upload_texture(image, nparr)
                i = i+1
            else:
                break

    def image_upload_thread(self, pixels, size):
        nparr = np.frombuffer(pixels, np.uint8).reshape((size[1], size[0], 4))

        if platform == "android":
            nparr = nparr[::-1]

        nparr = cv2.cvtColor(nparr, cv2.COLOR_RGBA2BGRA)
        retval, buffer = cv2.imencode('.png', nparr)

        tag_id = ""

        for i in self.tags.keys():
            if self.boundingBox_input_text == self.tags[i]["name"]:
                tag_id = i
                break

        if tag_id == "":
            url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/tags?name='+self.boundingBox_input_text
            response = requests.post(url, headers = {"Content-Type": "application/octet-stream", "Training-Key": training_key})
            tag_id = json.loads(response.content)
            self.tags[tag_id["id"]] = {"name": tag_id["name"], "selected": True}
            tag_id = tag_id["id"]


        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/images?tagIds='+tag_id

        response = requests.post(url, data = bytes(buffer), headers = {"Content-Type": "application/octet-stream", "Training-Key": training_key})

        image_detalis = json.loads(response.content)

        image_id = image_detalis["images"][0]["image"]["id"]

        url = ENDPOINT+'customvision/'+version+'/training/projects/'+project_id+'/images/regions'

        startx,starty,stopx,stopy = self.get_calculated_coords()

        region_json = {"regions":[
            {
                "imageId": image_id,
                "tagId": tag_id,
                "left": float((startx+self.boundingBox.borderWidth)/self.camera.resolution[0]),
                "top": float((self.camera.resolution[1]-(stopy-self.boundingBox.borderWidth))/self.camera.resolution[1]),
                "width": float(((stopx-self.boundingBox.borderWidth)-(startx+self.boundingBox.borderWidth))/self.camera.resolution[0]),
                "height": float(((stopy-self.boundingBox.borderWidth)-(starty+self.boundingBox.borderWidth))/self.camera.resolution[1])
            }
        ]}

        x = requests.post(url, data = json.dumps(region_json), headers = {"Content-Type": "application/json", "Training-Key": training_key})

        self.change_state(0)

    def train_button_pressed(self, obj):
        if self.process_state == 0 or self.process_state == 10:
            self.change_state(7)
        elif self.process_state == 7:
            self.change_state(8)
        elif self.process_state == 8:
            self.change_state(7)
        elif self.process_state == 9 or self.process_state == 11:
            self.change_state(14)

    def button_pressed(self, obj):
        if self.process_state == 0 or self.process_state == 10:
            self.change_state(2)
        elif self.process_state == 2 or self.process_state == 7 or self.process_state == 8 or self.process_state == 9 or self.process_state == 11:
            if self.iterations["selected"] == None:
                self.change_state(0)
            else:
                self.change_state(10)
        elif self.process_state == 4:
            self.change_state(5)

    def on_window_resize(self, window, width, height):

        """

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

        self.boundingBox.size = self.camera.size
        """
        """
        nparr = np.zeros((480, 640, 4), dtype=np.uint8)
        nparr[:,:,:] = 255
        image_texture = Texture.create(size=(nparr.shape[1], nparr.shape[0]), colorfmt='rgba')
        image_texture.blit_buffer(nparr.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.boundingBox.texture = image_texture
        """

        self.camera.size = (640/480*height*8/10, height*8/10)
        self.camera.pos = ((width-self.camera.size[0])/2, height/10)

        self.boundingBox.size = self.camera.size
        self.boundingBox.pos = self.camera.pos

        self.button.size = (width, height/10)
        self.button.pos = (0, height-height/10)

        self.train_button.size = (width, height/10)
        self.train_button.pos = (0, 0)

        if self.training_popup is not None:
            self.training_popup.pos=((width-3*width/4)/2, (height-3*height/4)/2)
            self.training_popup.size=(width*3/4, height*3/4)
            with self.training_popup.canvas.before:
                Color(233, 236, 239, 1)
                Line(width=10, rounded_rectangle=(self.training_popup.pos[0], self.training_popup.pos[1], self.training_popup.size[0], self.training_popup.size[1], 25))
                Rectangle(pos=(self.training_popup.pos[0], self.training_popup.pos[1]), size=(self.training_popup.size[0], self.training_popup.size[1]))

        """
        for item in self.camera.canvas.before.children:
            if isinstance(item, Rectangle):
                item.size = self.camera.size
        """

class Which_uCApp(App):
    def build(self):
        return Main()

if __name__ == "__main__":
    Which_uCApp().run()
