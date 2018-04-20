from unittest import TestCase
from preprocess import get_char_and_ctable, CharacterTable, get_char_id


class CTableTestCase(TestCase):

    def setUp(self):
        chars, self.ctable = get_char_and_ctable()


    def test_encode(self):
        q = 'こんにちは'
        q_list = [c for c in q]
        ids = [self.ctable.char_id[c] for c in q_list]
        onehot = self.ctable.encode(q, 5)
        self.assertEqual(len(onehot), 39)


