import pickle


class Loader:
    def __init__(self):
        pass

    @staticmethod
    def load(path: str) -> list:
        with open(path, 'rt') as f:
            return f.readlines()

    @staticmethod
    def save(data: list, path: str):
        with open(path, 'wt') as f:
            f.writelines(data)

    @staticmethod
    def save_pickle(data: dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(obj=data, file=f)


