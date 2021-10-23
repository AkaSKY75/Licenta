import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

class MyGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
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
        self.input.text = ""

class MyApp(App):
    def build(self):
        #return Label(text="Licenta")
        return MyGrid()

if __name__ == "__main__":
    MyApp().run()
