import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tqdm import trange

# fix random seed for reproducibility
numpy.random.seed(7)


def load_dataset(data: str) -> (numpy.ndarray, MinMaxScaler):
    """
    The function loads dataset from given file name and uses MinMaxScaler to transform data
    :param data: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """
    # load the dataset
    dataframe = pd.read_csv(data, delimiter='\t', header=None)
    init_dataset = dataframe.values
    init_dataset = init_dataset.astype('float32')

    # plt.plot(init_dataset)
    # plt.show()

    # normalize the dataset
    init_scaler = MinMaxScaler(feature_range=(0, 1))
    init_dataset = init_scaler.fit_transform(init_dataset)
    return init_dataset, init_scaler


def split_dataset(dataset: numpy.ndarray, train_size, look_back) -> (numpy.ndarray, numpy.ndarray):
    """
    Splits dataset into training and test datasets. The last `look_back` rows in train dataset
    will be used as `look_back` for the test dataset.
    :param dataset: source dataset
    :param train_size: specifies the train data size
    :param look_back: number of previous time steps as int
    :return: tuple of training data and test dataset
    """
    if not train_size > look_back:
        raise ValueError('train_size must be lager than look_back')
    train, test = dataset[0:train_size, :], dataset[train_size - look_back:len(dataset), :]
    print('train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
    return train, test


def create_dataset(dataset: numpy.ndarray, look_back: int = 1) -> (numpy.ndarray, numpy.ndarray):
    """
    The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
    and the `look_back`, which is the number of previous time steps to use as input variables
    to predict the next time period — in this case defaulted to 1.
    :param dataset: numpy dataset
    :param look_back: number of previous time steps as int
    :return: tuple of input and output dataset
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


def make_forecast(model: keras.models.Sequential, look_back_buffer: numpy.ndarray, timesteps: int = 1,
                  batch_size: int = 1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model.predict(look_back_buffer, batch_size)
        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
    return forecast_predict


if __name__ == '__main__':
    datasource = 'charts/ecg/Ecg_0_100Hz.tsv'
    dataset, scaler = load_dataset(datasource)

    # split into train and test sets
    look_back = int(len(dataset) * 0.20)
    train_size = int(len(dataset) * 0.70)
    train, test = split_dataset(dataset, train_size, look_back)

    # create train and test datasets
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # create and fit Multilayer Perceptron model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64,
                                activation='relu',
                                batch_input_shape=(1, look_back, 1),
                                stateful=True,
                                return_sequences=False))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for _ in trange(20, desc='fitting model\t', mininterval=1.0):
        model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=0, shuffle=False)
        model.reset_states()

    # generate predictions for training
    train_predict = model.predict(train_x, 1)
    test_predict = model.predict(test_x, 1)

    # generate forecast predictions
    forecast_predict = make_forecast(model, test_x[-1::], timesteps=100, batch_size=1)

    # invert dataset and predictions
    dataset = scaler.inverse_transform(dataset)
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    forecast_predict = scaler.inverse_transform(forecast_predict)

    plt.plot(dataset)
    plt.plot([numpy.zeros(1) for _ in range(look_back)] +
             [x for x in train_predict])
    plt.plot([numpy.zeros(1) for _ in range(look_back)] +
             [numpy.zeros(1) for _ in train_predict] +
             [x for x in test_predict])
    plt.plot([numpy.zeros(1) for _ in range(look_back)] +
             [numpy.zeros(1) for _ in train_predict] +
             [numpy.zeros(1) for _ in test_predict] +
             [x for x in forecast_predict])
    plt.legend()
    plt.show()
