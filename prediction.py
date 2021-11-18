from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 227, 227
classification = ["Female","Male"]
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

#loading the model we saved
model = load_model('gender_model1.h5')
model1 = load_model('age_model3.h5')

#predicting images using model
img = image.load_img('sarah.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = np.argmax(model.predict(images), axis=-1)
classes1 = np.argmax(model1.predict(images), axis=-1)

#printing the results
print(classification[classes[0]])
print(ageList[classes1[0]])