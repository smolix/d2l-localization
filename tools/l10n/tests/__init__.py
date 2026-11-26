"""Tests: cd tools && python3 -m unittest discover -s l10n/tests -t ."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from l10n import config

HAVE_BOOK = (config.D2L_NEU / '.git').exists()
SLOW = os.environ.get('L10N_SLOW_TESTS', '1') != '0'


class TempRepo(unittest.TestCase):
    """A scratch d2l-localization root with its own languages.yml."""

    languages = (
        'languages:\n'
        '  - {code: en, locale: en, name: English, bucket: d2l-en, hosts: [d2l.ai, en.d2l.ai], '
        'dir: ltr, pinned: null, forum: english}\n'
        '  - {code: ja, locale: ja, name: 日本語, bucket: d2l-ja, hosts: [ja.d2l.ai], dir: ltr, '
        'pinned: null, forum: english}\n'
        '  - {code: zh, locale: zh-Hans, name: 中文, bucket: d2l-zh, hosts: [zh.d2l.ai], dir: ltr, '
        'pinned: en-2026.12.01, forum: own, forum_category: 16}\n'
        'old_editions:\n'
        '  - {code: en, name: English (old), url: https://en-v2.d2l.ai, bucket: d2l-en-v2}\n'
        'test_languages:\n'
        '  - {code: xx, locale: en, name: Pseudo, bucket: null, hosts: [xx.d2l.ai], dir: ltr, '
        'pinned: null, forum: english}\n'
    )

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='l10n-test-'))
        (self.tmp / 'languages.yml').write_text(self.languages, encoding='utf-8')
        self._saved = (config.REPO, config.LANGUAGES_YML)
        config.REPO = self.tmp
        config.LANGUAGES_YML = self.tmp / 'languages.yml'

    def tearDown(self):
        config.REPO, config.LANGUAGES_YML = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)
