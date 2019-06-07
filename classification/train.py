from data_utils.data_loader import image_generator
from config import *
from models.cnn import *
from keras.callbacks import TensorBoard
import glob
import os


def find_latest_checkpoint(checkpoints_path):
    paths = glob.glob(checkpoints_path + ".*")
    maxep = -1
    r = None
    for path in paths:
        ep = int(path.split('.')[-1])
        if ep > maxep:
            maxep = ep
            r = path
    return r, maxep


def train_cnn(train_images,
              input_height=None,
              input_width=None,
              checkpoints_path=None,
              epochs=5,
              batch_size=2,
              validate=False,
              val_images=None,
              val_batch_size=2,
              auto_resume_checkpoint=False,
              load_weights=None,
              steps_per_epoch=512,
              optimizer_name='adadelta',
              callbacks=None):

    if (not input_height is None) and (not input_width is None):
        model = get_cnn(input_height=input_height, input_width=input_width)
    else:
        model = get_cnn()

    if validate:
        assert not (val_images is None)

    if not optimizer_name is None:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    latest_ep = -1
    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint, latest_ep = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    train_gen = image_generator(train_images, batch_size, input_height, input_width)

    if validate:
        val_gen = image_generator(val_images, val_batch_size, input_height, input_width)

    if not validate:
        print("Starting Epoch ", latest_ep + 1)
        model.fit_generator(train_gen,
                            steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks)
        if not checkpoints_path is None:
            model.save_weights(checkpoints_path + "." + str(latest_ep + 1 + epochs))
            print("saved ", checkpoints_path + ".models." + str(latest_ep + 1 + epochs))

    else:
        print("Starting Epoch ", latest_ep + 1)
        model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=200,
                            callbacks=callbacks,
                            epochs=epochs)
        if not checkpoints_path is None:
            model.save_weights(checkpoints_path + "." + str(latest_ep + 1 + epochs))
            print("saved ", checkpoints_path + ".models." + str(latest_ep + 1 + epochs))


if __name__ == "__main__":
    tensorboard = TensorBoard(log_dir="./logs")
    train_cnn(train_images=os.path.join(DATA_PATH, "classification/train"),
              input_height=INPUT_HEIGHT,
              input_width=INPUT_WIDTH,
              checkpoints_path="checkpoints/cnn_alexnet",
              epochs=EPOCH,
              batch_size=BATCH_SIZE,
              validate=VALIDATA,
              val_images=os.path.join(DATA_PATH, "classification/test"),
              val_batch_size=VAL_BATCH_SIZE,
              auto_resume_checkpoint=AUTO_RESUME,
              load_weights=None,
              steps_per_epoch=STEPS_PER_EPOCH,
              optimizer_name=OPTIMIZER,
              callbacks=[tensorboard])
