"""Section and chapter reference wording (xref.py): the templates, the HTML,
search.json and LaTeX rewrites, the links to English, the strings.yml keys
and the build_one_pdf.sh hook."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from l10n import build, config, scope, xref
from l10n.build import strings

from . import HAVE_BOOK, TempRepo

TAG = 'en-2026.11.27'

CFG = {'templates': {'sec': '{n} 節', 'ch': '第 {n} 章'},
       'html_words': {'sec': 'セクション', 'ch': 'チャプター'},
       'tex_words': {'sec': '§', 'ch': 'チャプター'}}


def xr(href, text):
    return f'<a href="{href}" class="quarto-xref"><span>{text}</span></a>'


class TestTemplates(unittest.TestCase):
    def test_templates(self):
        self.assertEqual(xref.templates({}), {})
        self.assertEqual(xref.templates({'xref.sec': '{n} 節', 'nav.pdf': 'PDF'}),
                         {'sec': '{n} 節'})
        for bad in ('節', '{n} {n} 節', '{n}', ' {n} ', 3):
            with self.assertRaises(ValueError, msg=repr(bad)):
                xref.templates({'xref.ch': bad})

    def test_fill(self):
        self.assertEqual(xref.fill('{n} 節', '2.1.3'), '2.1.3\xa0節')
        self.assertEqual(xref.fill('第 {n} 章', '3', '&nbsp;'), '第&nbsp;3&nbsp;章')
        self.assertEqual(xref.fill('{n}절', '2.1.3'), '2.1.3절')
        self.assertEqual(xref.fill('第 {n} 章', '\\ref{sec-chap-a b}', '~'),
                         '第~\\ref{sec-chap-a b}~章')    # the number keeps its spaces


class TestHtml(unittest.TestCase):
    PAGE = ('<p>' + xr('#sec-a', 'セクション 2.1.3') + '、'
            + xr('../c/index.html', 'チャプター 3') + '、'
            + xr('x.html#sec-b', 'セクション&nbsp;12.4') + '、'
            + xr('#fig-a', '図&nbsp;2.1.1') + '、'
            + '<a href="#eq-a" class="quarto-xref">式&nbsp;2.1.4</a>、'
            + xr('../chapter_installation/index.html', 'インストール') + '、'
            + xr('../c/index.html', '9&nbsp; 最適化') + '</p>'
            + '<button aria-label="セクションを切り替え"></button>'
            + '<p>本文のセクション 3 つ</p>')

    def test_rewrite(self):
        out, n = xref.rewrite_html(self.PAGE, CFG)
        self.assertEqual(n, 3)
        self.assertIn(xr('#sec-a', '2.1.3&nbsp;節'), out)
        self.assertIn(xr('../c/index.html', '第&nbsp;3&nbsp;章'), out)
        self.assertIn(xr('x.html#sec-b', '12.4&nbsp;節'), out)
        for keep in (xr('#fig-a', '図&nbsp;2.1.1'), '式&nbsp;2.1.4', 'インストール',
                     '9&nbsp; 最適化', 'セクションを切り替え', '本文のセクション 3 つ'):
            self.assertIn(keep, out)
        self.assertEqual(xref.left_html(out, CFG), 0)
        self.assertEqual(xref.rewrite_html(out, CFG), (out, 0))     # idempotent

    def test_one_kind_only(self):
        cfg = dict(CFG, templates={'sec': '{n} 節'})
        out, n = xref.rewrite_html(self.PAGE, cfg)
        self.assertEqual(n, 2)
        self.assertIn('チャプター 3', out)

    def test_shared_word(self):
        cfg = dict(CFG, html_words={'sec': '§', 'ch': '§'})
        out, _ = xref.rewrite_html(xr('a', '§ 3') + xr('b', '§ 3.2'), cfg)
        self.assertEqual(out, xr('a', '第&nbsp;3&nbsp;章') + xr('b', '3.2&nbsp;節'))

    def test_no_templates(self):
        cfg = dict(CFG, templates={})
        self.assertEqual(xref.rewrite_html(self.PAGE, cfg), (self.PAGE, 0))

    def test_search(self):
        data = [{'text': '（セクション 2.1）と チャプター\xa03 と 図 2.1', 'title': '2.1 線形回帰',
                 'href': 'a.html'}, {'section': 'セクション 4.5.6'}]
        self.assertEqual(xref.rewrite_search(data, CFG), 3)
        self.assertEqual(data[0]['text'], '（2.1\xa0節）と 第\xa03\xa0章 と 図 2.1')
        self.assertEqual(data[0]['title'], '2.1 線形回帰')
        self.assertEqual(data[1]['section'], '4.5.6\xa0節')
        self.assertEqual(xref.rewrite_search(data, CFG), 0)

    def test_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp)
            (book / 'c').mkdir()
            (book / 'site_libs').mkdir()
            (book / 'c' / 'p.html').write_text(self.PAGE, encoding='utf-8')
            (book / 'site_libs' / 'q.html').write_text(self.PAGE, encoding='utf-8')
            (book / 'search.json').write_text(json.dumps([{'text': 'セクション 2.1'}]),
                                              encoding='utf-8')
            st = xref.rewrite_book(book, CFG)
            self.assertEqual(st, {'files': 1, 'refs': 3, 'search': 1, 'left': 0})
            self.assertEqual((book / 'site_libs' / 'q.html').read_text(encoding='utf-8'),
                             self.PAGE)
            self.assertEqual(json.loads((book / 'search.json').read_text(encoding='utf-8')),
                             [{'text': '2.1\xa0節'}])
            self.assertEqual(xref.rewrite_book(book, CFG)['refs'], 0)


class TestTex(unittest.TestCase):
    TEX = ('（§~\\ref{sec-linear-regression}）と チャプター~\\ref{sec-chap-preliminaries}、'
           '図~\\ref{fig-a}、\\hyperref[sec-chap-installation]{インストール}、'
           '\\href{https://d2l.ai/x.html}{チャプター\\,19 （英語版）}、'
           '§~\\ref{eq-a}')

    def test_rewrite(self):
        out, n = xref.rewrite_tex(self.TEX, CFG)
        self.assertEqual(n, 2)
        self.assertIn('（\\ref{sec-linear-regression}~節）', out)
        self.assertIn('第~\\ref{sec-chap-preliminaries}~章', out)
        for keep in ('図~\\ref{fig-a}', '\\hyperref[sec-chap-installation]{インストール}',
                     '{チャプター\\,19 （英語版）}', '§~\\ref{eq-a}'):
            self.assertIn(keep, out)
        self.assertEqual(xref.rewrite_tex(out, CFG), (out, 0))

    def test_label_decides_kind(self):
        cfg = dict(CFG, tex_words={'sec': '§', 'ch': '§'})
        out, n = xref.rewrite_tex('§~\\ref{sec-chap-a} §~\\ref{sec-b}', cfg)
        self.assertEqual(out, '第~\\ref{sec-chap-a}~章 \\ref{sec-b}~節')

    def test_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'xref.json').write_text(json.dumps(CFG, ensure_ascii=False), encoding='utf-8')
            tex = d / 'book.tex'
            tex.write_text(self.TEX, encoding='utf-8')
            r = subprocess.run(['python3', str(Path(xref.__file__)), 'tex', str(tex),
                                str(d / 'xref.json')], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('2 section/chapter', r.stdout)
            self.assertIn('第~\\ref{sec-chap-preliminaries}~章', tex.read_text(encoding='utf-8'))


class TestEnglishLinks(unittest.TestCase):
    TERMS = {'Chapter': 'チャプター', 'Section': 'セクション', 'Figure': '図',
             'Equation': '式'}

    def test_ref_text(self):
        tp = CFG['templates']
        ch = {'kind': 'chapter', 'number': '19', 'id': 'x'}
        sec = {'kind': 'section', 'number': '18.4.2', 'id': 'y'}
        fig = {'kind': 'fig', 'number': '18.4.1', 'id': 'z'}
        self.assertEqual(scope.ref_text(ch, self.TERMS, tp), '第\xa019\xa0章')
        self.assertEqual(scope.ref_text(sec, self.TERMS, tp), '18.4.2\xa0節')
        self.assertEqual(scope.ref_text(fig, self.TERMS, tp), '図\u202f18.4.1')
        self.assertEqual(scope.ref_text(ch, self.TERMS), 'チャプター\u202f19')   # unset
        link = scope.ref_link(dict(ch, file='chapter_lm-x/index.md', h1=True), self.TERMS,
                              '（英語版）', 'https://d2l.ai', tp)
        self.assertTrue(link.startswith('[第\xa019\xa0章 （英語版）](https://d2l.ai/'
                                        'chapter_lm-x/index.html)'), link)


class TestStringsKeys(TempRepo):
    def test_kept_and_stripped(self):
        en = self.tmp / 'en.yml'
        en.write_text('code: en\nlang: en\ndir: ltr\nnav.slides: Slides\n', encoding='utf-8')
        lang = config.get_language('ja')
        path, _, _, _ = strings(lang, en_file=en)
        text = path.read_text(encoding='utf-8')
        self.assertIn('xref.sec: "{n} 节"', text)              # the zh/ko documentation
        self.assertEqual(scope.xref_templates(lang), {})
        path.write_text(text + 'xref.sec: "{n} 節"\nxref.ch: "第 {n} 章"\n', encoding='utf-8')
        path, added, removed, _ = strings(lang, en_file=en)
        self.assertNotIn('xref.sec', removed)
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual((data['xref.sec'], data['xref.ch']), ('{n} 節', '第 {n} 章'))
        self.assertEqual(strings(lang, en_file=en)[0].read_text(encoding='utf-8'),
                         path.read_text(encoding='utf-8'))      # stable
        self.assertEqual(scope.xref_templates(lang), CFG['templates'])
        wt = self.tmp / 'wt'
        wt.mkdir()
        flat = yaml.safe_load(scope.build_locale_file(lang, wt).read_text(encoding='utf-8'))
        self.assertFalse([k for k in flat if k.startswith(('xref.', 'l10n.'))])
        self.assertEqual(flat['nav.slides'], 'Slides')

    def test_bad_template(self):
        lang = config.get_language('ja')
        lang.root.mkdir(parents=True, exist_ok=True)
        (lang.root / 'strings.yml').write_text('xref.sec: "節"\n', encoding='utf-8')
        with self.assertRaises(config.L10nError):
            scope.xref_templates(lang)


def have_tag():
    return HAVE_BOOK and config.git('tag', '--list', TAG).strip() == TAG


@unittest.skipUnless(have_tag(), f'd2l-neu with tag {TAG} not found')
class TestPdfHook(unittest.TestCase):
    def test_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            sh = wt / 'tools' / 'build_one_pdf.sh'
            sh.parent.mkdir()
            sh.write_text(config.show(TAG, 'tools/build_one_pdf.sh'), encoding='utf-8')
            build.patch_xref_hook(wt)
            build.patch_xref_hook(wt)            # a second build of the same worktree
            text = sh.read_text(encoding='utf-8')
            self.assertEqual(text.count(build.XREF_TEX_HOOK), 1)
            self.assertIn(build.XREF_TEX_ANCHOR + build.XREF_TEX_HOOK, text)
            # the hook line as build_one_pdf.sh runs it (cwd: the worktree)
            tex = wt / '_pdf' / 'pytorch' / 'Dive-into-Deep-Learning.tex'
            tex.parent.mkdir(parents=True)
            tex.write_text(TestTex.TEX, encoding='utf-8')
            run = ['bash', '-c', 'fw=pytorch; ' + build.XREF_TEX_HOOK + 'echo done']
            r = subprocess.run(run, cwd=wt, capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, 'done'))
            self.assertEqual(tex.read_text(encoding='utf-8'), TestTex.TEX)   # no settings
            (wt / '.l10n').mkdir()
            (wt / '.l10n' / 'xref.json').write_text(json.dumps(CFG), encoding='utf-8')
            shutil.copy2(xref.__file__, wt / '.l10n' / 'xref.py')
            r = subprocess.run(run, cwd=wt, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('\\ref{sec-linear-regression}~節', tex.read_text(encoding='utf-8'))
            (wt / '.l10n' / 'xref.json').write_text('{', encoding='utf-8')
            r = subprocess.run(run, cwd=wt, capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)                  # a broken hook stops the PDF


if __name__ == '__main__':
    unittest.main()
