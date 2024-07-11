import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load pre-trained model
def load_model(model_path):
    model = tf.keras.models.load_model("C:/Users/harsh/Desktop/Task_2_3/model.keras")
    return model

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Animal Classifier')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the model
model = load_model("model.h5")

# Class names
SPECIES_LIST = ["dog", "cat", "horse", "spider", "butterfly", "chicken", "sheep", "cow", "squirrel", "elephant"]
AGE_LIST = ["child", "adult"]
DIET_LIST = ["herbivore", "carnivore"]

# Define age and diet mappings for species
SPECIES_AGE_MAPPING = {
    "dog": "adult",
    "cat": "child",
    "horse": "adult",
    "spider": "child",
    "butterfly": "adult",
    "chicken": "child",
    "sheep": "adult",
    "cow": "adult",
    "squirrel": "child",
    "elephant": "adult"
}

SPECIES_DIET_MAPPING = {
    "dog": "carnivore",
    "cat": "carnivore",
    "horse": "herbivore",
    "spider": "carnivore",
    "butterfly": "herbivore",
    "chicken": "herbivore",
    "sheep": "herbivore",
    "cow": "herbivore",
    "squirrel": "herbivore",
    "elephant": "herbivore"
}

def predict_animal(file_path):
    global label1

    # Load and preprocess the image
    image = Image.open(file_path)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = SPECIES_LIST[predicted_class_idx]

    # Determine age and diet
    predicted_age = SPECIES_AGE_MAPPING[predicted_class]
    predicted_diet = SPECIES_DIET_MAPPING[predicted_class]

    # Update the label with the prediction
    label1.configure(foreground="#011638", text=f"Predicted Class: {predicted_class}\nAge: {predicted_age}\nDiet: {predicted_diet}")

def show_predict_button(file_path):
    predict_btn = Button(top, text="Predict Animal", command=lambda: predict_animal(file_path), padx=10, pady=5)
    predict_btn.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    predict_btn.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_predict_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Animal Classifier', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
