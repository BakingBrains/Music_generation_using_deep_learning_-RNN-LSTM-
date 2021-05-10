from preprocessing import generate_training_sequence, SEQUENCE_LENGTH
import tensorflow.keras as keras

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "MusicGen.h5"
CHECKPOINT_PATH = "C:/Users/adeat/PycharmProjects/Music Generation/Log/MusicGen.h5"


def build_model(output_units, num_units, loss, learning_rate):
    # model
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)
    # compile
    model.compile(loss= loss, optimizer= keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
    model.summary()

    return model


def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99


def train(output_units = OUTPUT_UNITS, num_units =NUM_UNITS, loss = LOSS, learning_rate =LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequence(SEQUENCE_LENGTH)
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=False,
        monitor='accuracy',
        verbose=1,
        save_best_only=False)
    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[model_checkpoint_callback, lr_scheduler])
    # save the model
    # model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()