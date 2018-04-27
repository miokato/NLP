from unittest import TestCase
from preprocess.preprocess import get_char_and_ctable


class CTableTestCase(TestCase):

    def setUp(self):
        chars, self.ctable = get_char_and_ctable()


    def test_encode(self):
        q = 'こんにちは'
        q_list = [c for c in q]
        ids = [self.ctable.char_id[c] for c in q_list]
        onehot = self.ctable.encode(q, 5)
        self.assertEqual(len(onehot), 39)


