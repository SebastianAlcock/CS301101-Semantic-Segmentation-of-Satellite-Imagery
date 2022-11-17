import nni
import tensorflow as tf
from keras.models import load_model

params = {
    'dense_units': 128,
    'activation_type': 'relu',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

model = load_model("models/satellite_standard_unet_100epochs.hdf5",
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef':jacard_coef})

adam = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=adam, loss=total_loss, metrics=['accuracy', jacard_coef])

callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)

model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(X_test, y_test), callbacks=[callback])
accuracy = model.evaluate(X_test, y_test, verbose=1)

nni.report_final_result(accuracy)
