import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from l10n import compact
from l10n.units import decisions_brief

LOG = """# Decisions log: Deutsch (de)

## a.md#u01 (agent, 2026-11-26)

- „loss“ → „Verlust“; „kernel“ → „Kern“

## a.md#u02 (agent, 2026-11-26)

- „kernels“ → „Faltungskern“
"""


class CompactTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / 'de'
            root.mkdir()
            (root / 'decisions.md').write_text(LOG, encoding='utf-8')
            (root / 'style.md').write_text('# Style\n\n## 1. Register\n\n## 2. Sources\n', encoding='utf-8')
            lang = SimpleNamespace(code='de', name='Deutsch', root=root)
            d = Path(t) / 'work'
            meta = compact.split(LOG, d, size=10)
            self.assertEqual(len(meta), 2)
            (d / 'ext00.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in [
                {'en': 'loss', 'tr': 'Verlust'}, {'en': 'kernel', 'sense': 'convolution', 'tr': 'Kern'}]) + '\n',
                encoding='utf-8')
            (d / 'ext01.jsonl').write_text(json.dumps(
                {'en': 'kernels', 'sense': 'convolution', 'tr': 'Faltungskern'}, ensure_ascii=False) + '\n',
                encoding='utf-8')
            c = compact.merge(d, 'de')
            self.assertEqual((c['entries'], c['keys'], c['conflicts']), (3, 2, 1))
            with self.assertRaises(Exception):
                compact.assemble(d, lang, '2026-11-27')       # conflict not resolved yet
            (d / 'resolved.jsonl').write_text(json.dumps(
                {'key': 'kernel', 'en': 'kernel', 'sense': 'convolution', 'de': 'Faltungskern',
                 'note': 'not Kern'}, ensure_ascii=False) + '\n', encoding='utf-8')
            (d / 'style-section.md').write_text('## X. Conventions\n\n### X.1 Headings\n\n- rule\n',
                                                encoding='utf-8')
            r = compact.assemble(d, lang, '2026-11-27')
            self.assertEqual((r['lexicon_lines'], r['style_section']), (2, 3))
            log = (root / 'decisions.md').read_text(encoding='utf-8')
            self.assertIn('- kernel (convolution) → Faltungskern — not Kern', log)
            self.assertIn('- loss → Verlust', log)
            self.assertIn('style.md §3', log)
            style = (root / 'style.md').read_text(encoding='utf-8')
            self.assertIn('## 3. Conventions', style)
            self.assertIn('### 3.1 Headings', style)
            self.assertIn('Faltungskern', decisions_brief(log, 'The kernels of a layer.'))
            self.assertNotIn('Verlust', decisions_brief(log, 'The kernels of a layer.'))
            with self.assertRaises(Exception):
                compact.split(log, d)                       # already compacted


if __name__ == '__main__':
    unittest.main()
