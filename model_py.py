#Model.py
import nni
import tensorflow as tf
params = {
    'dense_units': 128,
    'activation_type': 'relu',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(params['dense_units'], activation=params['activation_type']),
    tf.keras.layers.Dropout(params['dropout_rate']),
    tf.keras.layers.Dense(10)
])

adam = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=adam, loss=loss_fn, metrics=['accuracy'])

callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)

model.fit(x_train, y_train, epochs=20, verbose=2, callbacks=[callback])
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

nni.report_final_result(accuracy)
