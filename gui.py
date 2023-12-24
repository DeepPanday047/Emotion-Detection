# import tkinter as tk 
# from tkinter import filedialog
# from tkinter import *

# from tensorflow.keras.models import model_from_json
# from PIL import Image,ImageTk
# import numpy as np 

# import cv2 as cv 



# def FacialExpressionModel(json_file,weights_file):
#     with open (json_file,'r') as file:
#         loaded_model_json = file.read()
#         model = model_from_json(loaded_model_json)

#     model.load_weights(weights_file)
#     model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#     return model

# top = tk.Tk()
# top.geometry('800x600')
# top.title('Emotion Decetor')
# top.configure(background='#CDCDCD')

# label1 = Label(top,background='#CDCDCD',font=('arial',15,'bold'))
# sign_image = Label(top)

# facec =cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = FacialExpressionModel('model_a_kaggle.json',weights_file='model_weights_kaggle.h5')

# EMOTIONS_LIST = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# def Detect(file_path):
#     global Label_packed

#     image = cv.imread(file_path)
#     gray_img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#     faces = facec.detectMultiScale(gray_img,1.3,5)
#     try:
#         for (x,y,w,h) in faces:
#             fc = gray_img[y:y+h,x:x+w]
#             roi = cv.resize(fc,(48,48))
#             pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
#             print('Predicted Emotion is:' + pred)
#             label1.configure(foreground='#011638',text=pred)
#     except:
#         label1.configure(foreground='#011638',text='Unable to detect..')


# def show_Detect_button(file_path):
#     detect_b = Button(top,text='Detect Button',command=lambda: Detect(file_path),padx=10,pady=5)
#     detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
#     detect_b.place(relx=0.79,rely=0.46)


# def upload_image():
#     try:
#         file_path = filedialog.askopenfilename()
#         if file_path:
#             uploaded = Image.open(file_path)
#             uploaded.thumbnail((top.winfo_width() // 2.3, top.winfo_height() // 2.3))
#             im = ImageTk.PhotoImage(uploaded)

#             sign_image.configure(image=im)
#             sign_image.image = im
#             label1.configure(text='')
#             show_Detect_button(file_path)
#         else:
#             label1.configure(foreground='#011638', text='No file selected')

#     except Exception as e:
#         label1.configure(foreground='#011638', text=f'Error: {str(e)}')


# upload = Button(top,text='upload Image',command=upload_image,padx=10,pady=5)
# upload.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
# upload.pack(side='bottom',pady=50)
# sign_image.pack(side='bottom',expand='True')
# label1.pack(side='bottom',expand='True')
# heading = Label(top,text='Emotion Detector',pady=20,font=('arial',20,'bold'))
# heading.configure(background='#CDCDCD',foreground='#364156')
# heading.pack()
# top.mainloop()


# top.protocol("WM_DELETE_WINDOW", lambda: (top.destroy(), cv.destroyAllWindows()))


import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, 'r') as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('model_a_kaggle.json', weights_file='model_weights_kaggle.h5')

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def Detect(file_path):
    image = cv.imread(file_path)
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    
    try:
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                fc = gray_img[y:y+h, x:x+w]
                # Adjust the size of the ROI according to your model's input size
                roi = cv.resize(fc, (48, 48))
                pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
                print('Predicted Emotion is: ' + pred)
                label1.configure(foreground='#011638', text=f'Predicted Emotion: {pred}')
        else:
            label1.configure(foreground='#011638', text='No faces detected.')

    except Exception as e:
        label1.configure(foreground='#011638', text=f'Error: {str(e)}')

def show_Detect_button(file_path):
    detect_b = Button(top, text='Detect Button', command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path)
            uploaded.thumbnail((top.winfo_width() // 2.3, top.winfo_height() // 2.3))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            label1.configure(text='')
            show_Detect_button(file_path)
        else:
            label1.configure(foreground='#011638', text='No file selected')

    except Exception as e:
        label1.configure(foreground='#011638', text=f'Error: {str(e)}')

upload = Button(top, text='Upload Image', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

cv.waitKey(0)
cv.destroyAllWindows()
top.mainloop()


