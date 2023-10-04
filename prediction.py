import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Load the finance data from keras
mnist = tf.keras.datasets.mnist

# Split the data into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def model_training():
    # Create the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=3)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc)

    # Save the model
    model.save('epic_num_reader.model')

    # Load the model
    new_model = tf.keras.models.load_model('epic_num_reader.model')

    # Make predictions
    predictions = new_model.predict(x_test)


if __name__ == '__main__':
    print("Prediction.py is running")
    if input("Do you want to train the model? (y/n): ") == 'y':
        model_training()
    else:
        model_loaded = tf.keras.models.load_model('epic_num_reader.model')
        predictions = model_loaded.predict(x_test)
    
    # Print the predictions and save the image
    for i in range(5):
        print("Prediction: ", np.argmax(predictions[i]))
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        import os
        if not os.path.exists('static/images'):
            os.makedirs('static/images')
        plt.savefig('static/images/prediction{}.png'.format(i))
        plt.show()