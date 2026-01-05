import tensorflow as tf

model = tf.keras.models.load_model("spiral_mobilenet.h5")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = r"C:\Users\Mohana\OneDrive\Desktop\final pro\data2\spiral\testing"

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    shuffle=False
)
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
import numpy as np

y_true = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
from sklearn.metrics import classification_report

print(classification_report(
    y_true,
    y_pred,
    target_names=test_gen.class_indices.keys()
))
from tensorflow.keras.preprocessing import image

img_path = r"data2\spiral\testing\healthy\V03HE1.png"

img = image.load_img(img_path, target_size=(128,128))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

if pred[0][0] > 0.5:
    print("Predicted: Parkinson's")
else:
    print("Predicted: Healthy")
print(test_gen.class_indices)
