from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as keras_backend
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

weights_filepath=R'weights/weights-improvement-{epoch:02d}-{val_acc:.4f}-LargerSeed.hdf5'
image_path = R'data'
training_data_dir = R'data/training_set'
validation_data_dir = R'data/test_set'
image_width = 350
image_height = 350
shear_range = 0.2
zoom_range = 0.2
batch_size = 32
steps_per_epoch = 2000
epochs = 50
number_of_validation_samples = 1000

if keras_backend.image_data_format() == 'channels first':
    input_shape = (3, image_height, image_width)
else:
    input_shape = (image_height, image_width, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=True)

training_dataset = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_data_generator = ImageDataGenerator(
    rescale=1./255)

validation_dataset = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='max')
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint, early_stopping]

model.fit_generator(
    training_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=validation_dataset,
    validation_steps=(number_of_validation_samples // batch_size))

