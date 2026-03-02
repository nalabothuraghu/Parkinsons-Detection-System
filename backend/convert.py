import tensorflow as tf
print('Loading Keras model...')
model = tf.keras.models.load_model('spiral_mobilenet.keras')
print('Converting to TFLite...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open('spiral_mobilenet.tflite', 'wb') as f:
    f.write(converter.convert())
print('✅ Conversion complete!')
