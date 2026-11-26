import unittest

from l10n import config, tfile
from l10n.segment import reconstruct, segment, text_hash, translatable

from . import HAVE_BOOK

DOC = '''```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Linear Regression
:label:`sec_linreg`

Suppose we have $n$ examples, see :numref:`sec_other`
and §{sec_x}. The model is
$$\\hat{y} = \\mathbf{w}^\\top \\mathbf{x} + b,$$
:eqlabel:`eq_y`

$$
L = \\frac{1}{2} \\sum_i (y_i - \\hat{y}_i)^2
$$
:eqlabel:`eq_loss`

```{.python .input #linreg-imports}
%%tab pytorch
import torch
```

:begin_tab:`pytorch`
In PyTorch, `torch.randn` draws samples.
:end_tab:

![The shapes of $\\mathbf{X}$.](../img/shapes.svg)
:width:`400px`
:label:`fig_shapes`

| Model | Accuracy |
|---|---|
| MLP | 92.3% |

<!-- execution: locked; reason=expensive -->

Same text.

Same text.

## Exercises

1. Show that $x > 0$.
   Then compute

   $$y = x^2$$

   for $x = 2$.
2. Second exercise.

   ```python
   print(1)
   ```

   After the code.

[Discussions](https://d2l.discourse.group/t/1)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.1]{.kicker}

Linear Regression<br>
:::
:::

::: {.slide title="The Affine Model" only="pytorch"}
[The Model Class]{.kicker}

::: {.cols .vc}
::: {.col}
- **Weights:** one per feature.
- **Bias:** an offset.
:::
:::

@linreg-imports
@fig:affine-plane
:::
'''


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.blocks = segment(DOC)

    def kinds(self, kind):
        return [b for b in self.blocks if b.kind == kind]

    def test_partition(self):
        self.assertEqual(''.join(b.text for b in self.blocks), DOC)

    def test_identity_reconstruct(self):
        tr = {b.anchor: b.content for b in translatable(self.blocks)}
        self.assertEqual(reconstruct(self.blocks, tr), DOC)
        stored = tfile.parse(tfile.render({a: tfile.Entry(t) for a, t in tr.items()}))
        self.assertEqual(reconstruct(self.blocks, {a: e.text for a, e in stored.items()}), DOC)

    def test_kinds(self):
        by = {b.content.split('\n')[0]: b for b in self.blocks}
        self.assertEqual(by['# Linear Regression'].kind, 'heading')
        self.assertEqual(by[':label:`sec_linreg`'].kind, 'meta')
        self.assertTrue(by['# Linear Regression'].translatable)
        # display math glued to the paragraph stays in it; standalone math is its own block
        para = by['Suppose we have $n$ examples, see :numref:`sec_other`']
        self.assertEqual(para.kind, 'paragraph')
        self.assertIn('$$\\hat{y}', para.content)
        self.assertEqual(by['$$'].kind, 'math')
        self.assertFalse(by['$$'].translatable)
        self.assertEqual(by['```{.python .input #linreg-imports}'].cell_id, 'linreg-imports')
        self.assertEqual(by['In PyTorch, `torch.randn` draws samples.'].kind, 'tab_note')
        self.assertEqual(by['![The shapes of $\\mathbf{X}$.](../img/shapes.svg)'].kind, 'caption')
        self.assertEqual(by['| MLP | 92.3% |'].kind, 'table_row')
        self.assertEqual(by['|---|---|'].kind, 'table_sep')
        self.assertEqual(by['<!-- execution: locked; reason=expensive -->'].kind, 'comment')
        self.assertEqual(by['1. Show that $x > 0$.'].kind, 'exercise')
        self.assertIn('$$y = x^2$$', by['1. Show that $x > 0$.'].content)
        # a fence inside a list item splits it: code is never translatable
        self.assertEqual(by['2. Second exercise.'].content, '2. Second exercise.')
        self.assertEqual(by['   ```python'].kind, 'code')
        self.assertEqual(by['   After the code.'].kind, 'exercise')
        self.assertEqual(by['[Dive into Deep Learning · §2.1]{.kicker}'].kind, 'verbatim')
        self.assertEqual(by['[The Model Class]{.kicker}'].kind, 'kicker')
        st = by['::: {.slide title="The Affine Model" only="pytorch"}']
        self.assertEqual(st.kind, 'slide_title')
        self.assertTrue(st.translatable)
        self.assertEqual(by['- **Weights:** one per feature.'].kind, 'slide_text')
        self.assertEqual(by['@linreg-imports'].kind, 'placeholder')
        self.assertEqual(by['@linreg-imports'].refs, ['linreg-imports'])
        self.assertEqual(by['@fig:affine-plane'].kind, 'placeholder')
        self.assertTrue(by['@fig:affine-plane'].in_slides)
        for b in translatable(self.blocks):
            self.assertNotIn('```', b.content)
            self.assertFalse(b.content.startswith(':::') and b.kind not in ('slide_title',))

    def test_anchors(self):
        same = [b for b in self.blocks if b.content == 'Same text.']
        self.assertEqual([b.anchor for b in same], [text_hash('Same text.'),
                                                   text_hash('Same text.') + '-2'])
        self.assertEqual(text_hash('a  b\nc'), text_hash('a b c'))
        anchors = [b.anchor for b in translatable(self.blocks)]
        self.assertEqual(len(anchors), len(set(anchors)))

    def test_raw_html_fence(self):
        doc = ('---\npagetitle: "Dive into Deep Learning"\ntoc: false\n---\n\n```{=html}\n<style>\n'
               '.a { color: red; }\n</style>\n<section>\n  <h2>Start here</h2>\n  <p>&nbsp;</p>\n'
               '<!-- @gen-begin x -->\n<p>Generated</p>\n<!-- @gen-end -->\n</section>\n```\n')
        blocks = segment(doc)
        self.assertEqual(''.join(b.text for b in blocks), doc)
        tr = [b.content for b in translatable(blocks)]
        self.assertEqual(tr, ['pagetitle: "Dive into Deep Learning"', '  <h2>Start here</h2>'])


@unittest.skipUnless(HAVE_BOOK, 'd2l-neu checkout not found')
class TestBookRoundTrip(unittest.TestCase):
    def test_every_file_byte_identical(self):
        rev = 'HEAD'
        files = config.book_files(rev)
        self.assertGreater(len(files), 100)
        for f in files:
            text = config.show(rev, f)
            blocks = segment(text)
            tr = {b.anchor: b.content for b in translatable(blocks)}
            stored = tfile.parse(tfile.render({a: tfile.Entry(t) for a, t in tr.items()}))
            self.assertEqual(reconstruct(blocks, {a: e.text for a, e in stored.items()}), text, f)
            for b in translatable(blocks):
                self.assertNotRegex(b.content, r'(?m)^\s*(```|~~~)', f'{f}:{b.line + 1}')
                self.assertNotRegex(b.content, r'(?m)^:(begin|end)_tab:', f'{f}:{b.line + 1}')


if __name__ == '__main__':
    unittest.main()
