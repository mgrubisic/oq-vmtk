import re
import pickle


def import_from_pkl(path):
    """
    Function to import data stored in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file

    ------
    Output
    ------
    Pickle file
    """

    # import file
    with open(path, 'rb') as file:
        return pickle.load(file)


def export_to_pkl(path, var):
    """
    Function to store data in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file
    :param var:          variable                Variable to store
    ------
    Output
    ------
    Pickle file
    """

    # store file
    with open(path, 'wb') as file:
        return pickle.dump(var, file)


def sorted_alphanumeric(data):
    """
    Function to sort data alphanumerically
    -----
    Input
    -----
    :param data:             list                Data to be sorted
    ------
    Output
    ------
    Sorted data of the same type as "data"
    """

    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
