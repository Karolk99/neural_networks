from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np
from matplotlib import pyplot
import astor_data as astor


class Collection:  # features nie działa jeszcze musi być 1
    def __init__(self, row_type, file, features=1, output_len=1, time_steps=5):
        self.time_steps = time_steps
        self.features = features
        self.output_len = output_len
        training_values = self.normalization(astor.create_list_from_file(file, row_type))
        self.training_values = training_values
        self.inputs, self.outputs = self.create_inputs_outputs(training_values)
        self.predict_val = self.create_predict_values(training_values[-self.time_steps - self.output_len: -output_len])  #self.create_predict_values(training_values[-time_steps:])
        #astor.write_to_file(self.inputs, self.outputs, len(training_values) - self.input_len - self.input_len + 1)

    def create_predict_values(self, values):
        inputs_array = np.zeros((1, len(values), self.features))
        for i in range(self.time_steps):
            inputs_array[0][i][0] = values[i]

        return inputs_array

    @staticmethod
    def normalization(training_values):
        return (training_values - np.mean(training_values)) / np.std(training_values)

    def create_inputs_outputs(self, values):
        batch_no = len(values) - (self.output_len + self.time_steps - 1)  # zmienić na batch_no
        inputs_array = np.zeros((batch_no, self.time_steps, self.features))
        outputs_array = np.zeros((batch_no, self.output_len, self.features))

        for i in range(batch_no):
            for j in range(self.time_steps):
                inputs_array[i][j][0] = values[i + j]
            for j in range(self.output_len):
                outputs_array[i][j][0] = values[i + j + self.time_steps]

        return inputs_array, outputs_array

    @staticmethod
    def to_float(values):
        for i in range(len(values)):
            values[i] = float(values[i].replace(',', ''))

        return values

    def __str__(self):
        print("INPUTS:")
        print(self.inputs)
        print("////////////////////////")
        print("OUTPUTS:")
        print(self.outputs)
        print("////////////////////////")


class DeepLearning:
    def __init__(self, file, row_type, batch_no=32, features=1, output_len=1, epochs=100, validation_split=0.2,
                 time_steps=5, neurons=50):
        self.batch_no = batch_no
        self.collection = Collection(file=file, row_type=row_type, features=features, time_steps=time_steps,
                                     output_len=output_len)
        self.neurons = neurons
        self.model = None
        self.history = None
        self.creating_model(time_steps, features, output_len)
        self.learning(epochs, validation_split)

    def creating_model(self, time_steps, features, output_len):

        self.model = Sequential([
            Bidirectional(
                GRU(self.neurons, activation='relu', input_shape=(time_steps, features), return_sequences=True),
            ),
            # Bidirectional(LSTM(10, activation='relu', return_sequences=False)),
            Bidirectional(
                LSTM(int(self.neurons/2), activation='relu', return_sequences=False)
            ),
            #LSTM(int(self.neurons/2), activation='relu', return_sequences=False),
            # TimeDistributed(Dense(output_len))
            Dense(units=1 * output_len, kernel_initializer=initializers.zeros),
            Reshape([output_len, 1])
        ])

        self.model.compile(optimizer='adam', loss='mse')

    def learning(self, epochs, validation_split):
        self.history = self.model.fit(self.collection.inputs, self.collection.outputs,
                                      validation_split=validation_split,
                                      epochs=epochs,
                                      batch_size=32,
                                      shuffle=False,
                                      verbose=0
                                      )

    def change_batch_number(self, new_number):
        self.batch_no = new_number

    def hist_plot(self):
        pyplot.plot(self.history.history['loss'])
        pyplot.plot(self.history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()

    def predict(self, inputs):
        outputs = self.model.predict(inputs)
        result = []
        for i in range(len(outputs[0])):
            result.append(outputs[0][i])

        return result

    #def predict_plot(self):
       # pyplot.plot(list(range(0, len(self.collection.training_values))), self.collection.training_values,
       #             list(range(len(self.collection.training_values), len(self.collection.training_values) + self.collection.output_len)), self.predict(self.collection.predict_val), 'r-')
       # pyplot.show()
    def predict_plot(self):
        pyplot.plot(list(range(0, len(self.collection.training_values))), self.collection.training_values,
                    list(range(len(self.collection.training_values) - self.collection.output_len,
                               len(self.collection.training_values))),
                    self.predict(self.collection.predict_val), 'r-')
        pyplot.show()
