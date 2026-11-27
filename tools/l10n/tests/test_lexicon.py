import unittest

from l10n.units import decisions_brief, fold_lexicon, lexicon_surfaces, log_backlog, new_log

LOG = """# Decisions log

Intro.

## Lexicon
<!-- l10n:lexicon -->
- loss surface / loss landscape → مشهد الخسارة — not سطح
- bias (statistics) → التحيز
- kernel (convolution) → نواة
- learning rate → معدل التعلم
<!-- l10n:lexicon-end -->

## chapter_x/y.md#u01 (agent, date)

- new → جديد
"""


class LexiconTest(unittest.TestCase):
    def test_surfaces(self):
        self.assertEqual(lexicon_surfaces('- a b / c (sense) → x — n'), ['a b', 'c'])
        self.assertEqual(lexicon_surfaces('no arrow'), [])

    def test_filter(self):
        out = decisions_brief(LOG, 'We plot loss landscapes and kernels; `learning_rate`, $bias$.')
        self.assertIn('مشهد الخسارة', out)            # plural surface form matches
        self.assertIn('نواة', out)
        self.assertNotIn('معدل التعلم', out)          # only inside inline code
        self.assertNotIn('التحيز', out)               # only inside math
        self.assertIn('2 of 4 lexicon entries', out)
        self.assertIn('Intro.', out)
        self.assertIn('- new → جديد', out)            # entries after the lexicon stay verbatim

    def test_no_markers(self):
        self.assertEqual(decisions_brief('## a\n- x → y\n', 'x'), '## a\n- x → y\n')

    def test_fold(self):
        out, c = fold_lexicon(LOG, ['- Learning rate → معدل تعلم — changed',
                                    '- adam step → خطوة Adam',
                                    '- bias (statistics) → (delete)',
                                    'not a lexicon line'])
        self.assertEqual(c, {'added': 1, 'changed': 1, 'deleted': 1, 'unchanged': 0})
        lex = out.split('<!-- l10n:lexicon -->\n')[1].split('<!-- l10n:lexicon-end -->')[0]
        self.assertEqual([l.split(' →')[0] for l in lex.strip().split('\n')],
                         ['- adam step', '- kernel (convolution)', '- Learning rate',
                          '- loss surface / loss landscape'])
        self.assertTrue(out.endswith('<!-- l10n:lexicon-end -->\n'))   # unit sections dropped
        self.assertIn('Intro.', out)
        kept, _ = fold_lexicon(LOG, [], keep_sections=True)
        self.assertIn('- new → جديد', kept)

    def test_new_log_and_backlog(self):
        log = new_log('Français', 'fr')
        self.assertEqual(log_backlog(log), (1, True))           # the newline after the end marker
        out, c = fold_lexicon(log + '\n## u01 (a, d)\n\n- x → y\n', ['- loss → perte'])
        self.assertIn('- loss → perte\n<!-- l10n:lexicon-end -->', out)
        self.assertEqual(log_backlog(out)[0], 1)
        self.assertEqual(log_backlog('## a\n- x → y\n'), (15, False))


if __name__ == '__main__':
    unittest.main()
