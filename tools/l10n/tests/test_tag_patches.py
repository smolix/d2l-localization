"""fix_tag_tools: edits that older English tags need for a translated build
(part-PDF class check, fonts.emphasis), and `l10n strings` seeing the
setting the patch adds."""

import os
import tempfile
import unittest
from pathlib import Path

from l10n import build, config

from . import HAVE_BOOK

TAG = 'en-2026.11.27'
FILES = ('i18n/en.yml', 'tools/i18n.py', 'tools/gen_pdf.py',
         'tools/slides_to_beamer.py', 'tools/build_part_pdfs.py')


def have_tag():
    return HAVE_BOOK and config.git('tag', '--list', TAG).strip() == TAG


@unittest.skipUnless(have_tag(), f'd2l-neu with tag {TAG} not found')
class TestTagPatches(unittest.TestCase):
    def tree(self, tmp):
        wt = Path(tmp)
        for rel in FILES:
            text = config.show(TAG, rel)
            self.assertIsNotNone(text, rel)
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text, encoding='utf-8')
        return wt

    def test_patches_apply_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = self.tree(tmp)
            build.fix_tag_tools(wt)
            build.fix_tag_tools(wt)  # a second build of the same worktree
            en = (wt / 'i18n/en.yml').read_text(encoding='utf-8')
            self.assertEqual(en.count('fonts.emphasis:'), 1)
            i18n = (wt / 'tools/i18n.py').read_text(encoding='utf-8')
            self.assertEqual(i18n.count('def latex_locale_preamble'), 1)
            for rel in ('tools/gen_pdf.py', 'tools/slides_to_beamer.py'):
                self.assertIn('latex_locale_preamble()',
                              (wt / rel).read_text(encoding='utf-8'), rel)
            parts = (wt / 'tools/build_part_pdfs.py').read_text(encoding='utf-8')
            self.assertIn(build.DOCCLASS_NEW, parts)
            self.assertNotIn(build.DOCCLASS_OLD, parts)

    def test_strings_see_the_patched_setting(self):
        en = config.show(TAG, 'i18n/en.yml')
        self.assertNotIn('fonts.emphasis:', en)
        patched = build.patched_en_yml(en)
        self.assertEqual(patched.count('fonts.emphasis: italic'), 1)
        self.assertEqual(build.patched_en_yml(patched), patched)
        # the same lines the build patch adds
        diff = build.EMPHASIS_PATCH.read_text(encoding='utf-8')
        for line in build.EMPHASIS_LINES.splitlines():
            self.assertIn('+' + line, diff)


if __name__ == '__main__':
    unittest.main()
