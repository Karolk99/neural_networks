from deep_learning import *

BATCH_NO = 32
FEATURES = 1
OUTPUT_LEN = 20
EPOCHS = 200
VALIDATION_SPLIT = 0.2
TIME_STEPS = 10  # it means that in one learning process a web will process time_step number data and return
# output_len number data
NEURONS = 50
FILE = 'astor-dataset.csv'
ROW_TYPE = 'PredictionOfOrdersValue'

# model.save
# model.load
# live_input = [1,2,3,4,5,6,7,8,9,10] // normalizacje za pomoca std oraz mean
# model.predict(live_input) -> [1,2,3,4,5,6,7,8,9,10] -> [20, 30]

web = DeepLearning(batch_no=BATCH_NO, features=FEATURES, output_len=OUTPUT_LEN, epochs=EPOCHS,
                   validation_split=VALIDATION_SPLIT, time_steps=TIME_STEPS, file=FILE, row_type=ROW_TYPE,
                   neurons=NEURONS)

web.hist_plot()
web.predict_plot()
