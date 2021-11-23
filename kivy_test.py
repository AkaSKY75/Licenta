import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import socket
import ssl
import threading
import cv2

class OpenCVThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        vid = cv2.VideoCapture(0)
        while(True):

            ret, frame = vid.read()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()

        cv2.destroyAllWindows()

class MyGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)

        self.sites = ["www.robofun.ro", "www.ardushop.ro"]

        # OpenCV Initialize
        thread = OpenCVThread(1, "Thread", 1)
        thread.start()

        #   UI front setup
        self.cols = 1
        self.grid = GridLayout(**kwargs)
        self.add_widget(self.grid)
        self.grid.cols = 2
        self.grid.add_widget(Label(text="Name: "))
        self.input = TextInput(multiline=False)
        self.grid.add_widget(self.input)
        self.button = Button(text="Click me", font_size=40)
        self.button.bind(on_press=self.pressed)
        self.add_widget(self.button)

    def pressed(self, instance):
        #print(self.input.text)
        #   Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #   Wrap socket
        self.wrappedSocket = ssl.wrap_socket(self.sock, ssl_version=ssl.PROTOCOL_TLSv1)
        self.wrappedSocket.settimeout(5)
        self.wrappedSocket.connect(("www.ardushop.ro", 443))
        #self.wrappedSocket.sendall(b"GET /ro/ HTTP/1.1\r\nHost: ardushop.ro\r\n\r\n")
        self.wrappedSocket.sendall(b"POST /ro/search HTTP/1.1\r\nHost: www.ardushop.ro\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: 20\r\n\r\nsearch_query="+self.input.text.encode('ascii')+b"\r\n\r\n")
        response = b''
        bytes = self.wrappedSocket.recv(4096)
        while bytes != b'':
            response += bytes
            bytes = self.wrappedSocket.recv(4096)
        # try:
        #     while True:
        #         response += self.wrappedSocket.recv(4096)
        #         print("In while...")
        # except Exception as e:
        #     pass
        print(response)
        self.wrappedSocket.close()
        self.sock.close()

class MyApp(App):
    def build(self):
        #return Label(text="Licenta")

        return MyGrid()

# class UIThread(threading.Thread):
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#     def run(self):
#         MyApp().run()

if __name__ == "__main__":
    # thread = UIThread(1, "Thread", 1)
    # thread.start()
    MyApp().run()
