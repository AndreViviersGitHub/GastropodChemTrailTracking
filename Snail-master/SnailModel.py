import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
import numpy as np
import pandas as pd
from keras.utils import plot_model
import os
import visualkeras
from PIL import ImageFont
def getgenerators():
    matplotlib.style.use('ggplot')
    IMAGE_SHAPE = (64, 64)
    TRAINING_DATA_DIR = 'E:\\Snail Images\\Training\\Training'
    VALID_DATA_DIR = 'E:\\Snail Images\\Validation\\Validation'

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = datagen.flow_from_directory(
        TRAINING_DATA_DIR,
        shuffle=True,
        target_size=IMAGE_SHAPE,
        class_mode="categorical",
    )
    print(train_generator.class_indices)
    valid_generator = datagen.flow_from_directory(
        VALID_DATA_DIR,
        shuffle=False,
        target_size=IMAGE_SHAPE,
        class_mode="categorical",
    )

    return train_generator, valid_generator
    # print(1)


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                               input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def trainModel():
    model = getmodel()
    train_generator, valid_generator = getgenerators()

    EPOCHS = 20
    BATCH_SIZE = 512

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.samples // BATCH_SIZE,
                        verbose=1
                        )
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    valid_loss = history.history['val_loss']
    valid_acc = history.history['val_accuracy']
    return train_acc,valid_acc, train_loss, valid_loss, model
def save_combined_plot(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the combined loss and accuracy plot to disk.
    """
    fig, ax1 = plt.subplots(figsize=(12, 9))

    # Accuracy plot (left y-axis)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='black')
    ax1.plot(train_acc, color='green', linestyle='-', label='Train Accuracy')
    ax1.plot(valid_acc, color='blue', linestyle='-', label='Validation Accuracy')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()

    # Loss plot (right y-axis)
    ax2.set_ylabel('Loss', color='black')
    ax2.plot(train_loss, color='orange', linestyle='-', label='Train Loss')
    ax2.plot(valid_loss, color='red', linestyle='-', label='Validation Loss')
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine both legends underneath one another
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Accuracy and Loss')
    plt.savefig('combined_plot.png')
    plt.show()



def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    return train_acc, valid_acc, train_loss, valid_loss

def predictfromimg(model,image, labels):
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    predictions = model.predict(input_arr)

    # print(predictions)
    predicted_class = np.argmax(predictions, axis=-1)
    prediction = labels[predicted_class[0]][1]
    #print("------------------")
    #print(prediction)
    #print("------------------")
    return prediction


def predict(model, labels, image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    predictions = model.predict(input_arr)

    # print(predictions)
    predicted_class = np.argmax(predictions, axis=-1)
    for k in predicted_class:
        print("CLASS!")
        print("------------------")
        print(labels[k][1])
        print("------------------")


def getlabels():
    df_sheet_index = pd.read_excel('E:\\Snail Images\\Label.xlsx').to_numpy()
    return df_sheet_index

def loadModel():
    loaded_model = tf.keras.models.load_model("snail-model.h5")
    return loaded_model
def getmodel():
    model = build_model(num_classes=5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


    font = ImageFont.truetype("arial.ttf", 12)
    #Creates an image visualizing the DCNN.
    #visualkeras.layered_view(model, to_file='output.png', legend=True, font=font).show() # display using your system viewer
    return model

if __name__ == '__main__':
    labels = getlabels()
    train_loss, train_acc, valid_loss, valid_acc, model = trainModel()
    save_combined_plot(train_loss, train_acc, valid_loss, valid_acc)
    model.save("snail-model.h5")
    #pipeImg = "snail-0f846acf-2446-4178-843f-5fb84ba413da.jpg"
    # snailImg = "snail-0a802ddb-c44b-4b5e-a2e6-4a6b86bdf035.jpg"
    #path = "E:\\Snail Images\\"
    #predict(model, labels, path + pipeImg)
    # predict(model, path + snailImg)
