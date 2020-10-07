import tensorflow as tf
import matplotlib.pyplot as plt

from data_prep import *


# define the model
def get_compiled_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(1)
                                 ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# LOAD Data
df = pd.read_csv("mushrooms.csv")

# Encode categorical features
# df = label_encode_data(df)
df = one_hot_encode_data(df)    # Using One-hot-encoding because all categories are equally different

# DATA SPLITS
train, validate, test = train_test_split(df)

train_labels = train.pop('class')
train_dataset = tf.data.Dataset.from_tensor_slices((train.values, train_labels.values))
val_labels = validate.pop('class')
val_dataset = tf.data.Dataset.from_tensor_slices((validate.values, val_labels.values))
test_labels = test.pop('class')
test_dataset = tf.data.Dataset.from_tensor_slices((test.values, test_labels.values))

# Save the CSV files
train.to_csv("train.csv")
validate.to_csv("validation.csv")
test.to_csv("test.csv")

# Get the model
model = get_compiled_model()

# Train the model and validate
history = model.fit(train_dataset.batch(256), epochs=50, validation_data=val_dataset.batch(256))
# Save the model
model.save(('./model'))

# Plot the training over epochs
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='training')
ax.plot(history.history['val_accuracy'], label='validation')
ax.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# Test and Evaluate
result = model.test_on_batch(test.values, test_labels.values, return_dict=True)
print('The test accuracy is {}'.format(result["accuracy"]))
