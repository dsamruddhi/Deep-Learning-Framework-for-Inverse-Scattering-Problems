import numpy as np


class Process:

    @staticmethod
    def check_data_sanctity(input, output):
        assert not np.isnan(input).any()
        assert not np.isnan(output).any()

    @staticmethod
    def split_data(input, output, test_size):
        test_data_len = int(len(input) * test_size)
        train_data_len = len(input) - test_data_len
        input = np.asarray(input)
        output = np.asarray(output)

        train_input, train_output = input[:train_data_len, :, :, :], output[:train_data_len, :, :]
        test_input, test_output = input[train_data_len:, :, :, :], output[train_data_len:, :, :]
        return train_input, train_output, test_input, test_output
