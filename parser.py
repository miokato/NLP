import settings
from loader import Loader


def separate_input_output():
    data = Loader.load(settings.RAW_DATA)
    inputs, outputs = [], []
    for idx, line in enumerate(data):
        if idx % 2 == 0:
            inputs.append(line)
        else:
            outputs.append(line)

    Loader.save(inputs, settings.INPUTS)
    Loader.save(outputs, settings.OUTPUTS)


def padding_input_output():
    input_maxlen = _max_len(settings.INPUTS)
    output_maxlen = _max_len(settings.OUTPUTS)

    _padding(settings.INPUTS, input_maxlen)
    _padding(settings.OUTPUTS, output_maxlen)


def _max_len(path):
    data = Loader.load(path)
    maxlen = 0
    for line in data:
        if len(line) > maxlen:
            maxlen = len(line)
    return maxlen


def _padding(path, maxlen):
    result = []
    data = Loader.load(path)
    c = ' '
    for line in data:
        pad_num = maxlen - len(line)
        if pad_num > 0:
            line = line.strip('\n')
            line = _add_stop_word(line, settings.STOP_WORD)
            pad = c * pad_num
            line += pad
            line += '\n'
        result.append(line)
    Loader.save(result, path)


def _add_stop_word(line, stopword):
    line += stopword
    return line


def rm_space_input_output():
    """空白削除"""
    inputs = Loader.load(settings.INPUTS)
    outputs = Loader.load(settings.OUTPUTS)
    inputs = [line.replace(' ', '') for line in inputs]
    outputs = [line.replace(' ', '') for line in outputs]
    Loader.save(inputs, settings.INPUTS)
    Loader.save(outputs, settings.OUTPUTS)


if __name__ == '__main__':
    separate_input_output()
    padding_input_output()

