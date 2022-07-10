import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

file_path = 'CNN_data/samples'
MODEL_DIRECTORY = 'CNN_model'

IMG_SIZE = 32

def read_in_data():
    #images = np.array([])
    #image_labels = np.array([],dtype=int)
    images, image_labels = [], [] # Faster to append to python list first and then convert to numpy array
    for i in range(1, 11):
        prefix = str(i)
        while len(prefix) != 3:
            prefix = "0" + prefix
        category_path = file_path + "/Sample" + prefix
        with os.scandir(category_path) as category:
            for img in category:
                new_img = cv2.imread(category_path + "/" + img.name)
                new_img = cv2.resize(new_img, (IMG_SIZE, IMG_SIZE))  # 128 x 128 originally
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

                new_img = new_img / 255
                #images = np.append(images, new_img)
                #image_labels = np.append(image_labels, i-1)
                images.append(new_img)
                image_labels.append(i - 1)

    #plt.imshow(images[0])
    #plt.show()
    #print(images[0].shape)

    images = np.array(images)
    image_labels = np.array(image_labels)

    # shuffle_idx = np.random.permutation(len(training_data))
    X_train, X_test, Y_train, Y_test = train_test_split(images, image_labels, random_state=1234, test_size=0.1)

    # can call train_test_split again on X_train and Y_train to get another set for validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, random_state=1234, test_size=0.2)
    print("Training Set Shape = ", X_train.shape) # (7315, 32, 32)

    print("Test Set Shape = ", X_test.shape) # (1016, 32, 32)
    print("Validation Set Shape = ", X_validation.shape)  # (1829, 32, 32)

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, 3)
    X_validation = np.expand_dims(X_validation, 3)

    return X_train, X_test, Y_train, Y_test, X_validation, Y_validation

def create_model():

    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.1, input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.2, 0.2)
        ]
    )

    model = keras.Sequential(
        [  # keras.Input((32, 32, 1)), need this line if you dont use input_shape = as first layer in data_augmentation

            data_augmentation,
            layers.Conv2D(32, (3, 3), 1, activation='relu'),
            layers.Conv2D(32, (3, 3), 1, activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), 1, activation='relu'),
            layers.Conv2D(64, (3, 3), 1, activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(), # Flatten 3D tensor into 1D
            layers.Dense(580, 'relu'),
            layers.Dropout(0.5), # helps against overfitting
            layers.Dense(10, 'softmax') # last layer, so use softmax to get the biggest value from output of previous layers

        ])
    return model

def main(save_model = True):

    X_train, X_test, Y_train, Y_test, X_validation, Y_validation = read_in_data()

    model = create_model()
    #model.summary()

    EPOCHS = 200
    BATCH_SIZE = 32
    steps = X_train.shape[0] // BATCH_SIZE

    # Once validation loss begins to increase, model is over-fitting on the training data.
    # So if theres no decrease in validation loss for a while, should stop and save the model with the best weights found
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=25,
            verbose=1, restore_best_weights = True
        )
    ]

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer =Adam(), metrics = ['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, Y_validation),
                        steps_per_epoch = steps, shuffle=True, callbacks = callbacks) # callbacks = callbacks

    # Plot graph of validation and training loss
    loss = history.history['loss']
    epochs_labels = [i for i in range(len(history.history['loss']))]
    validation_loss = history.history['val_loss']
    plt.plot(epochs_labels, loss, label = 'Training Loss')
    plt.plot(epochs_labels, validation_loss, label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    score = model.evaluate(X_test, Y_test)
    print('(Test Loss, Test Accuracy) = ', score[0], score[1])
    if save_model:
        model.save(MODEL_DIRECTORY)
        print("Model Saved.")
    else:
        print("Model not saved.")

if __name__ == "__main__":
    main(save_model=True)