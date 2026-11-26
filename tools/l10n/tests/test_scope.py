"""Translation scope (scope.py): membership, references, placeholders, the
edits to the tag's tools, the notebook-zip filter, the UI strings."""

import io
import re
import unittest
import zipfile
from pathlib import Path

import yaml

from l10n import cli, config, scope, tfile
from l10n.answers import check_language
from l10n.build import strings
from l10n.pseudo import pseudo
from l10n.units import english_blocks

from . import HAVE_BOOK, TempRepo

TAG = 'en-2026.11.26'
DEFAULT = ['chapter_preface', 'chapter_basics', 'chapter_advanced',
           'chapter_mathematics-for-deep-learning', 'chapter_tools-for-deep-learning']


def have_tag():
    return HAVE_BOOK and config.git('tag', '--list', TAG).strip() == TAG


@unittest.skipUnless(have_tag(), f'd2l-neu with tag {TAG} not found')
class TestScopeAtTag(unittest.TestCase):
    def setUp(self):
        self.sc = scope.resolve('xx', DEFAULT, TAG)

    def test_membership_from_parts(self):
        sc = self.sc
        self.assertFalse(sc.full)
        for f in ('index.md', 'chapter_preface/index.md', 'chapter_introduction/index.md',
                  'chapter_notation/index.md', 'chapter_installation/index.md',
                  'chapter_preliminaries/ndarray.md', 'chapter_diffusion-models/index.md',
                  'chapter_mdl-linear-algebra/index.md',
                  'chapter_appendix-tools-for-deep-learning/hardware.md',
                  'chapter_references/zreferences.md'):
            self.assertTrue(sc.includes(f), f)
        for f in ('chapter_lm-pretrained/index.md', 'chapter_lm-agents/tool-calling.md',
                  'chapter_lm-data/crawl-lib.md', 'chapter_computer-vision/fine-tuning.md',
                  'chapter_gaussian-processes/gp-intro.md', 'chapter_language-models/index.md',
                  'chapter_attic/index.md'):
            self.assertFalse(sc.includes(f), f)
        self.assertEqual(len([c for c in sc.out_chapters if c.dir.startswith('chapter_lm-')]), 14)
        self.assertEqual({p.dir for p in sc.out_parts},
                         {'chapter_language-models', 'chapter_image-models', 'chapter_attic'})
        self.assertIn('chapter_lm-agents/index.md', sc.placeholders)
        self.assertIn('chapter_language-models/index.md', sc.placeholders)
        self.assertIn('chapter_lm-agents/tool-calling.md', sc.unbuilt)
        self.assertNotIn('chapter_lm-agents/index.md', sc.unbuilt)

    def test_spec_forms(self):
        self.assertTrue(scope.resolve('xx', 'all', TAG).full)
        self.assertTrue(scope.resolve('xx', None, TAG).full)
        one = scope.resolve('xx', ['chapter_lm-agents'], TAG)
        self.assertTrue(one.includes('chapter_lm-agents/tool-calling.md'))
        self.assertTrue(one.includes('chapter_language-models/index.md'))   # its part page
        self.assertFalse(one.includes('chapter_lm-data/index.md'))
        self.assertTrue(one.includes('index.md'))
        with self.assertRaises(config.L10nError):
            scope.resolve('xx', ['chapter_nonexistent'], TAG)

    def test_select_files(self):
        files = config.select_files(TAG, None, self.sc)
        self.assertTrue(all(self.sc.includes(f) for f in files))
        self.assertIn('chapter_mdl-dynamics/mdl-sdes.md', files)
        self.assertEqual(config.select_files(TAG, ['chapter_preface'], self.sc),
                         ['chapter_preface/index.md'])
        with self.assertRaises(config.L10nError) as e:
            config.select_files(TAG, ['chapter_lm-agents'], self.sc)
        self.assertIn('outside the translation scope', str(e.exception))

    def test_filter_quarto(self):
        text = config.show(TAG, '_quarto.yml')
        new = scope.filter_quarto(text, self.sc)
        cfg = yaml.safe_load(new)
        parts = [i for i in cfg['book']['chapters'] if isinstance(i, dict)]
        self.assertEqual(len(parts), len(yaml.safe_load(text)['book']['chapters']) - 2)
        lm = next(p for p in parts if p['part'] == 'chapter_language-models/index.qmd')
        self.assertEqual(len(lm['chapters']), 14)
        self.assertTrue(all(c.endswith('/index.qmd') for c in lm['chapters']))
        self.assertIn('chapter_mdl-linear-algebra/mdl-svd-low-rank.qmd', new)
        self.assertNotIn('language-models-pytorch.pdf', new)
        self.assertIn('basics-pytorch.pdf', new)

    def test_patches_apply_to_tag(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            for rel in ('tools/d2l_preprocess.py', 'tools/gen_pdf.py',
                        'tools/check_book_artifacts.py', 'make/render.mk'):
                (wt / rel).parent.mkdir(parents=True, exist_ok=True)
                (wt / rel).write_text(config.show(TAG, rel), encoding='utf-8')
            self.assertEqual(scope.tag_pdf_parts(wt), ['basics', 'advanced', 'language-models',
                                                       'mathematics-for-deep-learning'])
            scope.apply_patches(wt, scope.patches(self.sc, ['basics', 'advanced']))
            ns = {}
            src = (wt / 'tools/d2l_preprocess.py').read_text()
            block = src[src.index('# >>> l10n scope'):src.index('# <<< l10n scope')]
            exec('PDF_EXCLUDED_FILES = frozenset()\n' + block, ns)
            self.assertIn('chapter_lm-agents/index.md', ns['PDF_EXCLUDED_FILES'])
            self.assertIn('chapter_attic/index.md', ns['PDF_EXCLUDED_FILES'])
            self.assertNotIn('chapter_lm-agents/index.md', ns['L10N_UNBUILT'])
            with self.assertRaises(config.L10nError):      # anchors already replaced
                scope.apply_patches(wt, scope.patches(self.sc, []))

    def test_placeholder_titles(self):
        numbering = {'chapter_lm-agents/index.md': [27], 'chapter_lm-agents/tool-calling.md': [27, 1]}

        class L:
            book = Path('/nonexistent')
        text = scope.placeholder(L, self.sc, 'chapter_lm-agents/index.md', numbering,
                                 scope.STRINGS, 'https://d2l.ai')
        self.assertTrue(text.startswith('# Agents and Tool Use\n:label:`chap_lm_agents`\n'))
        self.assertIn('- [27 Agents and Tool Use](https://d2l.ai/chapter_lm-agents/index.html)'
                      '{hreflang="en"}', text)
        self.assertIn('[27.1 Tool Calling](https://d2l.ai/chapter_lm-agents/tool-calling.html)', text)
        self.assertIn(scope.STRINGS['l10n.chapter_not_translated'], text)
        part = scope.placeholder(L, self.sc, 'chapter_attic/index.md', {}, scope.STRINGS,
                                 'https://d2l.ai')
        self.assertIn(scope.STRINGS['l10n.part_not_translated'], part)
        self.assertIn('chapter_gaussian-processes/index.html', part)
        self.assertNotIn('gp-intro', part)

    def test_label_table(self):
        helper = {'numbering': {'chapter_lm-encoders/index.md': [19],
                                'chapter_lm-pretrained/scoring.md': [18, 4]},
                  'labels': {'fig_x': {'file': 'chapter_lm-encoders/bert.md', 'id': 'fig-x',
                                       'kind': 'fig', 'number': '19.2.3', 'title': ''}}}
        t = scope.label_table(self.sc, {'chap_lm_encoders', 'sec_lm_scoring', 'sec_sgd',
                                        'fig_x'}, helper)
        self.assertEqual(t['chap_lm_encoders']['kind'], 'chapter')
        self.assertEqual(t['chap_lm_encoders']['number'], '19')
        self.assertEqual(t['sec_lm_scoring'], {
            'file': 'chapter_lm-pretrained/scoring.md', 'id': 'sec_lm_scoring', 'h1': True,
            'title': t['sec_lm_scoring']['title'], 'kind': 'section', 'number': '18.4'})
        self.assertNotIn('sec_sgd', t)                 # in scope: not an English link
        self.assertNotIn('fig_x', t)                   # not defined in an out-of-scope page

    def test_in_scope_references_found(self):
        """The in-scope pages that point into the skipped parts (DESIGN 13)."""
        labels = scope.out_labels(self.sc)
        hits = {f for f in config.book_files(TAG) if self.sc.includes(f)
                and scope.references(config.show(TAG, f)) & set(labels)}
        for f in ('chapter_preface/index.md', 'chapter_introduction/index.md',
                  'chapter_installation/index.md',
                  'chapter_mdl-information-theory/mdl-mutual-information.md',
                  'chapter_computational-performance/further-reading.md'):
            self.assertIn(f, hits)
        self.assertGreaterEqual(len(hits), 12)


@unittest.skipUnless(have_tag(), f'd2l-neu with tag {TAG} not found')
class TestTagFixes(unittest.TestCase):
    def test_pdf_output_file(self):
        import tempfile
        from l10n.build import fix_tag_tools
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / 'tools').mkdir()
            (wt / 'tools/gen_pdf.py').write_text(config.show(TAG, 'tools/gen_pdf.py'))
            fix_tag_tools(wt)
            text = (wt / 'tools/gen_pdf.py').read_text()
            self.assertEqual(text.count('output-file: Dive-into-Deep-Learning'), 1)
            fix_tag_tools(wt)                                   # already there: no-op
            self.assertEqual((wt / 'tools/gen_pdf.py').read_text(), text)


class TestReferences(unittest.TestCase):
    TABLE = {
        'chap_lm_encoders': {'file': 'chapter_lm-encoders/index.md', 'id': 'chap_lm_encoders',
                             'h1': True, 'kind': 'chapter', 'number': '19', 'title': 'Encoders'},
        'eq_kl': {'file': 'chapter_lm-x/y.md', 'id': 'eq-kl', 'h1': False, 'kind': 'eq',
                  'number': '18.3.5', 'title': ''},
        'sec_sub': {'file': 'chapter_lm-x/y.md', 'id': 'sec-sub', 'h1': False,
                    'kind': 'section', 'number': None, 'title': 'A [sub] part'},
    }
    TERMS = {'Chapter': 'Kapitel', 'Section': 'Abschnitt', 'Equation': 'Gleichung'}

    def test_rewrite(self):
        text = ('Siehe :numref:`chap_lm_encoders` und :eqref:`eq_kl`, :ref:`sec_sub`, '
                ':numref:`sec_local`.\n\n```python\n# :numref:`chap_lm_encoders`\n```\n'
                '``:numref:`chap_lm_encoders` ``\n')
        out, n = scope.rewrite_refs(text, self.TABLE, self.TERMS, '(englisch)', 'https://d2l.ai')
        self.assertEqual(n, 3)
        self.assertIn('[Kapitel 19 (englisch)](https://d2l.ai/chapter_lm-encoders/index.html)'
                      '{.d2l-english-ref hreflang="en"}', out)
        self.assertIn('[Gleichung 18.3.5 (englisch)](https://d2l.ai/chapter_lm-x/y.html#eq-kl)', out)
        self.assertIn('[A \\[sub\\] part (englisch)](https://d2l.ai/chapter_lm-x/y.html#sec-sub)', out)
        self.assertIn(':numref:`sec_local`', out)
        self.assertIn('# :numref:`chap_lm_encoders`', out)          # code fence untouched
        self.assertIn('``:numref:`chap_lm_encoders` ``', out)        # inline code untouched
        # fix_crossref_numbers.py must not see "Section 18" (space/&nbsp; + digits)
        self.assertNotRegex(out, r'(Kapitel|Abschnitt|Gleichung)[ \xa0]\d')

    def test_h1(self):
        self.assertEqual(scope.h1('```\n# no\n```\n# Title {#x}\n\n:label:`chap_t`\n'),
                         ('Title', 'chap_t'))
        self.assertEqual(scope.h1('# T\nText\n'), ('T', None))


class TestZip(unittest.TestCase):
    def test_filter_zip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / 'd2l-pytorch.zip'
            with zipfile.ZipFile(z, 'w') as f:
                for name, data in [('d2l-pytorch/README.md', 'r'),
                                   ('d2l-pytorch/chapter_a/x.ipynb', '"../img/a.svg"'),
                                   ('d2l-pytorch/chapter_lm-b/y.ipynb', '"../img/b.svg"'),
                                   ('d2l-pytorch/img/a.svg', '<svg/>'),
                                   ('d2l-pytorch/img/b.svg', '<svg/>')]:
                    f.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
            self.assertTrue(scope.filter_zip(z, ['chapter_lm-b']))
            with zipfile.ZipFile(z) as f:
                self.assertEqual(f.namelist(), ['d2l-pytorch/README.md',
                                                'd2l-pytorch/chapter_a/x.ipynb',
                                                'd2l-pytorch/img/a.svg'])
                self.assertEqual(f.infolist()[0].date_time, (1980, 1, 1, 0, 0, 0))
            self.assertFalse(scope.filter_zip(z, ['chapter_lm-b']))


class TestPostprocess(unittest.TestCase):
    def test_markers_noindex_links(self):
        import tempfile
        page = ('<html><head>\n<meta charset="utf-8">\n</head><body><nav id="quarto-sidebar">'
                '<a href="../chapter_lm-a/index.html" class="sidebar-item-text sidebar-link">\n'
                ' <span class="menu-text"><span class="chapter-number">18</span>&nbsp; '
                '<span class="chapter-title">A</span></span></a>'
                '<a href="../chapter_basics/index.html" class="sidebar-item-text sidebar-link">\n'
                ' <span class="menu-text">Basics</span></a></nav>'
                '<a href="../chapter_lm-a/sec.html#x">see</a></body></html>')
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / '_book'
            for rel in ('chapter_x/p.html', 'chapter_lm-a/index.html', 'chapter_lm-a/sec.html'):
                (book / rel).parent.mkdir(parents=True, exist_ok=True)
                (book / rel).write_text(page, encoding='utf-8')
            prep = scope.Prepared([], ['chapter_lm-a/index.html'], ['chapter_lm-a/sec.html'],
                                  ['chapter_lm-a'], 0, [], '(Englisch)')
            stats = scope.postprocess(None, Path(tmp), prep)
            self.assertEqual(stats['deleted'], 1)
            self.assertFalse((book / 'chapter_lm-a/sec.html').exists())
            p = (book / 'chapter_x/p.html').read_text()
            self.assertIn('<span class="chapter-title">A</span> <span class="d2l-english-marker"',
                          p)
            self.assertEqual(p.count('d2l-english-marker'), 1)
            self.assertIn('href="https://d2l.ai/chapter_lm-a/sec.html#x" hreflang="en"', p)
            self.assertNotIn(scope.NOINDEX, p)
            self.assertIn(scope.NOINDEX, (book / 'chapter_lm-a/index.html').read_text())
            again = scope.postprocess(None, Path(tmp), prep)            # idempotent
            self.assertEqual((again['markers'], again['noindex'], again['links']), (0, 0, 0))


@unittest.skipUnless(have_tag(), f'd2l-neu with tag {TAG} not found')
class TestScopedTools(TempRepo):
    languages = TempRepo.languages.replace('languages:\n', 'scope: [chapter_preface]\nlanguages:\n', 1)

    def test_complete_check_and_status(self):
        config.set_language_field('xx', 'pinned', TAG)
        lang = config.get_language('xx')
        self.assertEqual(lang.scope, ['chapter_preface'])
        rep = check_language(lang, complete=True)
        bad = {e.split(':')[0] for e in rep.errors}
        self.assertTrue(bad)
        self.assertTrue(all(f in ('index.md', 'chapter_preface/index.md',
                                  'chapter_installation/index.md', 'chapter_notation/index.md',
                                  'chapter_introduction/index.md', 'chapter_references/zreferences.md')
                            for f in bad), bad)
        cli.main(['pseudo', 'xx'])
        files = sorted(str(p.relative_to(lang.book)) for p in lang.book.rglob('*.md'))
        self.assertIn('chapter_preface/index.md', files)
        self.assertNotIn('chapter_preliminaries/ndarray.md', files)
        self.assertEqual(check_language(lang, complete=True).errors, [])
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            cli.main(['status', 'xx'])
        self.assertIn('not translated: chapter_basics', buf.getvalue())

    def test_translated_placeholder_title(self):
        config.set_language_field('xx', 'pinned', TAG)
        lang = config.get_language('xx')
        rel = 'chapter_lm-agents/index.md'
        b = next(b for b in english_blocks(TAG, rel) if b.kind == 'heading')
        tfile.save(lang.book, rel, {b.anchor: tfile.Entry(pseudo(b.kind, b.content))})
        self.assertEqual(scope.translated_h1(lang, TAG, rel), '⟦Agents and Tool Use⟧')
        sc = scope.for_language(lang)
        text = scope.placeholder(lang, sc, rel, {rel: [27]}, scope.STRINGS, 'https://d2l.ai')
        self.assertTrue(text.startswith('# ⟦Agents and Tool Use⟧\n'))

    def test_strings_extra_keys(self):
        en = self.tmp / 'en.yml'
        en.write_text('code: en\nlang: en\ndir: ltr\nnav.slides: Slides\n', encoding='utf-8')
        lang = config.get_language('ja')
        path, _, _, todo = strings(lang, en_file=en)
        data = yaml.safe_load(path.read_text())
        self.assertEqual(data['l10n.english_marker'], '(English)')
        self.assertIn('l10n.english_marker', todo)
        path.write_text(path.read_text().replace(
            'l10n.english_marker: "(English)"  # TODO translate', 'l10n.english_marker: "（英語）"'))
        path, added, removed, todo = strings(lang, en_file=en)
        self.assertEqual(yaml.safe_load(path.read_text())['l10n.english_marker'], '（英語）')
        self.assertNotIn('l10n.english_marker', todo + removed + added)
        self.assertEqual(scope.ui_strings(lang)['l10n.english_marker'], '（英語）')
        wt = self.tmp / 'wt'
        wt.mkdir()
        loc = scope.build_locale_file(lang, wt)
        self.assertEqual(loc.parent, wt / '.l10n')
        flat = yaml.safe_load(loc.read_text())
        self.assertNotIn('l10n.english_marker', flat)
        self.assertEqual(flat['nav.slides'], 'Slides')


if __name__ == '__main__':
    unittest.main()
