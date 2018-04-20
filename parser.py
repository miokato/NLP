import settings
from loader import Loader


def separate_input_output():
    loader = Loader()
    data = loader.load(settings.RAW_DATA)
    inputs, outputs = [], []
    for idx, line in enumerate(data):
        if idx % 2 == 0:
            inputs.append(line)
        else:
            outputs.append(line)

    loader.save(inputs, settings.INPUTS)
    loader.save(outputs, settings.OUTPUTS)


if __name__ == '__main__':
    separate_input_output()

