from loader import Loader
import settings


class VocabBuilder:
    def build(self, input, vocab):
        for line in input:
            line = line.strip('\n')
            for char in line:
                vocab.add(char)


def build_vocabulary():
    """ 文字ベースの一覧表を作成 """
    loader = Loader()
    builder = VocabBuilder()

    vocabulary = set()

    input_data = loader.load(settings.INPUTS)
    output_data = loader.load(settings.OUTPUTS)

    builder.build(input_data, vocabulary)
    builder.build(output_data, vocabulary)

    vocabulary = sorted(list(vocabulary))

    # 必要のない文字は削除する

    char_id = dict((c, i) for (i, c) in enumerate(vocabulary))
    id_char = dict((i, c) for (i, c) in enumerate(vocabulary))

    loader.save_pickle(char_id, settings.CHAR_ID_PKL)
    loader.save_pickle(id_char, settings.ID_CHAR_PKL)

    print('Done!')


if __name__ == '__main__':
    output1 = 'data/char_id.pkl'
    output2 = 'data/id_char.pkl'

    # vocabularyファイルの作成
    #build_vocabulary()
