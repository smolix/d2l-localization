import unittest

from l10n.protect import CheckOptions, check_block, tokens
from l10n.pseudo import pseudo

EN = ('We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which '
      'updates $\\mathbf{w}$ after every minibatch of 256 examples; see '
      '[the code](https://d2l.ai/x.html) and `torch.optim`.')


class TestTokens(unittest.TestCase):
    def test_classes(self):
        toks = [(t.cls, t.text) for t in tokens(EN)]
        self.assertIn(('ref', ':numref:`sec_sgd`'), toks)
        self.assertIn(('math', '$\\mathbf{w}$'), toks)
        self.assertIn(('target', '](https://d2l.ai/x.html)'), toks)
        self.assertIn(('code', '`torch.optim`'), toks)

    def test_escaped_dollar_is_text(self):
        self.assertEqual([t.cls for t in tokens('costs \\$5 and \\$10 per hour')], [])

    def test_alt_text_is_translatable(self):
        a = tokens('<img src="a.png" alt="A cover">')[0].text
        b = tokens('<img src="a.png" alt="表紙">')[0].text
        self.assertEqual(a, b)


class TestCheck(unittest.TestCase):
    def check(self, tr, en=EN, kind='paragraph', **opts):
        return check_block(kind, en, tr, CheckOptions(**opts))

    def test_good_japanese(self):
        tr = ('損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、256 個の例からなる'
              'ミニバッチごとに $\\mathbf{w}$ を更新します。[コード](https://d2l.ai/x.html) と '
              '`torch.optim` を参照してください。')
        self.assertEqual(self.check(tr), ([], []))

    def test_changed_math_and_missing_ref(self):
        tr = 'SGD で $\\mathbf{v}$ を更新します（256）。[コード](https://d2l.ai/x.html) `torch.optim`'
        errors, _ = self.check(tr)
        self.assertTrue(any(e.startswith('math tokens differ') for e in errors))
        self.assertTrue(any(e.startswith('ref tokens differ') for e in errors))

    def test_order_is_warning_by_default(self):
        en = 'Compare $a$ with $b$.'
        tr = '$b$ と $a$ を比べます。'
        self.assertEqual(self.check(tr, en)[0], [])
        self.assertIn('math tokens reordered', self.check(tr, en)[1])
        self.assertIn('math tokens reordered', self.check(tr, en, order='error')[0])

    def test_cjk_adjacent_role(self):
        en = 'See :numref:`sec_sgd` for details.'
        errors, _ = self.check(':numref:`sec_sgd`で詳しく説明します。', en)
        self.assertTrue(any('CJK' in e for e in errors))
        self.assertEqual(self.check(':numref:`sec_sgd` で詳しく説明します。', en)[0], [])
        self.assertEqual(self.check('詳細（:numref:`sec_sgd`）を参照。', en)[0], [])
        self.assertEqual(self.check(':numref:`sec_sgd`で詳しく。', en, cjk_adjacent='off')[0], [])
        errors, _ = self.check('値$x$を使います。', 'Use the value $x$.')
        self.assertTrue(any('CJK' in e for e in errors))

    def test_numbers_and_ranges(self):
        en = 'About 2× faster on 40–55 GPUs.'
        self.assertEqual(self.check('約 2 倍高速です（40〜55 GPU）。', en), ([], []))
        _, w = self.check('高速です。', en)
        self.assertTrue(any(x.startswith('numbers not found') for x in w))

    def test_forbidden(self):
        en = 'Let $x$ be 2.'
        self.assertIn('bidi control character', self.check('‏$x$ を 2 とします。', en)[0])
        errors, _ = self.check('$x$2 とします。', en)
        self.assertTrue(any('closing $' in e for e in errors))

    def test_structure(self):
        self.assertTrue(self.check('## 損失関数', '# The Loss', 'heading')[0])
        self.assertEqual(self.check('## 損失関数 {.unnumbered}', '## The Loss {.unnumbered}',
                                    'heading'), ([], []))
        self.assertTrue(self.check('| a | b |', '| x | y | z |', 'table_row')[0])
        self.assertTrue(self.check('*強調が閉じない', '*emphasis* here')[0])
        en = '::: {.slide title="The Affine Model" only="pytorch"}'
        self.assertEqual(self.check('::: {.slide title="アフィンモデル" only="pytorch"}', en,
                                    'slide_title'), ([], []))
        self.assertTrue(self.check('::: {.slide title="アフィンモデル"}', en, 'slide_title')[0])

    def test_pseudo_passes(self):
        for kind, en in [('paragraph', EN), ('heading', '## Exercises {.unnumbered}'),
                         ('table_row', '| $|x|$ | `a|b` | 3 |'),
                         ('kicker', '[The Model $f$]{.kicker}'),
                         ('caption', '![A [tiny] plot of $f$.](../img/f.svg){width=50%}'),
                         ('list_item', '1. **Bold** start\n   and more $x$.')]:
            tr = pseudo(kind, en)
            self.assertIn('⟦', tr)
            self.assertEqual(check_block(kind, en, tr), ([], []), (kind, tr))


if __name__ == '__main__':
    unittest.main()
