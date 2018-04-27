from preprocess.loader import Loader
import settings


class VocabBuilder:
    def build(self, input, vocab):
        for line in input:
            line = line.strip('\n')
            for char in line:
                vocab.add(char)


def build_vocabulary():
    """ 文字ベースの一覧表を作成 """
    builder = VocabBuilder()

    vocabulary = set()

    input_data = Loader.load(settings.INPUTS)
    output_data = Loader.load(settings.OUTPUTS)

    builder.build(input_data, vocabulary)
    builder.build(output_data, vocabulary)

    vocabulary.add(settings.STOP_WORD)
    vocabulary = sorted(list(vocabulary))

    # 必要のない文字は削除する

    char_id = dict((c, i) for (i, c) in enumerate(vocabulary))
    id_char = dict((i, c) for (i, c) in enumerate(vocabulary))

    Loader.save_pickle(char_id, settings.CHAR_ID_PKL)
    Loader.save_pickle(id_char, settings.ID_CHAR_PKL)

    print('Done!')


if __name__ == '__main__':
    output1 = 'data/char_id.pkl'
    output2 = 'data/id_char.pkl'

    # vocabularyファイルの作成
    #build_vocabulary()
