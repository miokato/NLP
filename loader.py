import pickle


class Loader:
    def __init__(self):
        pass

    def load(self, path: str) -> list:
        with open(path, 'rt') as f:
            return f.readlines()

    def save(self, data: list, path: str):
        with open(path, 'wt') as f:
            f.writelines(data)

    def save_pickle(self, data: dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(obj=data, file=f)


