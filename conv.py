import tensorflow as tf
model = tf.keras.models.load_model('sign_model_v3.h5')
model.export('sign_model_v3_savedmodel')