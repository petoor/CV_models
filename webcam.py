import tkinter as tk
import cv2

from tkinter import ttk
from PIL import Image, ImageTk
from models import FaceDetection, Hands, BBoxFace

class Window(tk.Tk):
    def __init__(self, state="off"):
        super().__init__()
        self.title('Computer Vision Demo')
        self.state = state
        self.resizable(0, 0)
        self.geometry("640x572")
        self.label = ttk.Label(self)
        self.label.grid(row=0, column=0)
        self.label.after(2, self.show_frames)       

        # Create a Label to capture the Video frames
        tk.Button(self, text = "Off", height = 2, width=10, command = lambda: self.new_state("off")).place(x = 50,y = 485)
        tk.Button(self, text = "Face", height = 2, width=10, command = lambda: self.new_state("face")).place(x = 230,y = 485)
        tk.Button(self, text = "Hands", height = 2, width=10, command = lambda: self.new_state("hands")).place(x = 50,y = 535)
        tk.Button(self, text = "BBox Face", height = 2, width=10, command = lambda: self.new_state("bbox_face")).place(x = 230,y = 535)
        tk.Button(self, text = "Exit demo", height = 2, width=10, command = self.destroy).place(x = 425, y = 535)
 
        self.cap = cv2.VideoCapture(0)
        self.face_model = FaceDetection()
        self.bbox_face_model = BBoxFace()
        self.hands_model = Hands()
    
    def new_state(self, new_state):
        self.state = new_state            
    
    def get_frame(self):
        # Get the latest frame and convert into Image
        frame = self.cap.read()[1]
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        
               
        if self.state == "hands":
           frame = self.hands_model.detect(img=frame)      
        
        elif self.state == "face":
          frame = self.face_model.detect(img=frame)

        elif self.state == "bbox_face":
          frame = self.bbox_face_model.detect(img=frame)

        img = Image.fromarray(frame)       
        return img

    def show_frames(self):
       imgtk = ImageTk.PhotoImage(image = self.get_frame())
       self.label.imgtk = imgtk
       self.label.configure(image=imgtk)
       self.label.after(20, self.show_frames)

if __name__ == "__main__":
    win = Window()
    win.mainloop()
