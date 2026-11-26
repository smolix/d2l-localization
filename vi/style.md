# Style guide: Vietnamese (vi)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Vietnamese, or ✗ wrong / ✓ right). Examples marked F, O, S are real
sentences from the translation test `evals/vi-conv-layer/` (Fable, Opus,
Sonnet). Points that need a native reviewer's ruling are marked "(to confirm)"
and collected in §11.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, order and spell*.
The glossary's house terms come from the human community edition of D2L in
Vietnamese (2019–2021, `glossary.md`), not from the machine-translated text now
on vi.d2l.ai (`glossary/review/vi.md`); where this guide says "the earlier
edition" it means the human one.

## 1. Register

**Decision: authorial *chúng ta* (short form *ta*), the reader as *bạn*, the
authors alone as *chúng tôi*.** Reason: inclusive *chúng ta* is the "we work
through this together" voice of TRANSLATING.md §8 and of the earlier edition;
*ta* is the voice of Vietnamese mathematics textbooks (*ta có*, *ta được*) and
keeps derivations short; *bạn* is the neutral address of Vietnamese technical
books. Never *các bạn*, *các em*, *anh/chị*, *quý vị* (classroom or
ceremonial), never *mình*, *tớ*, *tui*.

| English | Vietnamese | Example |
|---|---|---|
| we + action, opening a section, a paragraph or a statement of purpose | *chúng ta* | "In :numref:`sec_why-conv`, we derived the convolutional layer on paper" → Trong :numref:`sec_why-conv`, chúng ta đã suy ra tầng tích chập trên giấy (F) |
| we + a step of a computation or derivation, and every later "we" of the paragraph | *ta* | "we slide a $2 \times 2$ kernel across …" → ta trượt một hạt nhân $2 \times 2$ qua … (F) |
| "let us", "let's", "consider" | *Hãy* + verb, or *Xét* … ; ✗ Hãy để chúng ta … | "Consider locating vertical edges" → Hãy xét bài toán xác định các biên dọc (F) |
| "we will see", "we will use" | bare verb; *sẽ* only for something that really comes later | "we examine it in :numref:`sec_padding`" → ta sẽ xem xét nó trong :numref:`sec_padding` |
| "our model", "our edge detector" | *của chúng ta* once, then nothing or *này* | bộ phát hiện biên $1 \times 2$ của chúng ta ; then: bộ phát hiện biên này |
| "you" = the reader (advice, a question to the reader, quiz slides, exercises) | *bạn*, lower case | "How would you design a blur kernel?" → Bạn sẽ thiết kế một hạt nhân làm mờ như thế nào? (F) |
| "you can", "one can" = anyone | *ta có thể*, or no subject: *có thể* + verb | "You can verify this numerically." → Có thể kiểm tra điều này bằng số. |
| "we recommend", "we thank", "in our experience" (the authors only) | *chúng tôi* | "We recommend AdamW." → Chúng tôi khuyên dùng AdamW. |
| "we" = people, the field | *con người*, *chúng ta*, *giới nghiên cứu* | "We humans attend selectively." → Con người chúng ta chú ý có chọn lọc. |

✗ A paragraph in which every sentence has *chúng ta* (S wrote it 28 times on
the test page, F 7 times with *ta* 24 times): it reads as translated. ✓ *chúng
ta* once, then *ta*, then sentences without a subject.

**Sentence shape** (each point is checkable):

- **No *một* for every "a/an", no *các/những* for every plural.** Vietnamese
  marks number only where it matters. ✗ Một tầng tích chập áp dụng một phép
  tương quan chéo với một hạt nhân. ✓ Tầng tích chập áp dụng phép tương quan
  chéo với một hạt nhân được học. Keep *một* where it means "one" or
  introduces something new; keep *các* where the plural carries the meaning
  (*mọi*, *từng*, *nhiều* are often better).
- **Verbs, not *sự/việc* + verb.** ✗ Việc xếp chồng các tầng dẫn đến sự mở
  rộng của vùng tiếp nhận. ✓ Xếp chồng các tầng làm vùng tiếp nhận rộng ra.
  No light verbs: ✗ thực hiện tương quan chéo giữa … (S), ✗ tiến hành huấn
  luyện ✓ tính tương quan chéo giữa … (F), huấn luyện.
- **Active voice.** *được* only where the agent does not matter (mô hình được
  huấn luyện trên ImageNet); *bị* only for something adverse (bị tràn số, bị
  quá khớp). Never *được … bởi*. ✗ mảnh đầu vào bị che phủ (F) ✓ mảnh đầu vào
  nằm dưới cửa sổ (O).
- **At most two *của* in a row.** ✗ đạo hàm theo phương ngang của độ sáng của
  ảnh ✓ đạo hàm độ sáng của ảnh theo chiều ngang.
- **Repeat the noun instead of *nó/chúng*** when the pronoun is more than one
  clause away from its noun or two candidates exist.
- **No *sẽ* for every "will", no *đã* for every past tense.** Context carries
  tense. ✗ Giờ đây chúng ta sẽ xem công thức đó … (S) ✓ Bây giờ ta xem công
  thức ấy … (F).
- **One main clause and at most two subordinate clauses** per sentence. Split
  English sentences at "which", "where", "so that" (TRANSLATING.md §3 allows
  it within a block).
- **No English left where Vietnamese has a word.** ✗ panel (c) (S), ✗ profile
  theo cột (S), ✗ RF hiệu dụng (S) ✓ phần (c), biên dạng theo cột, vùng tiếp
  nhận hiệu dụng. The loans of §4.2 are the only English common nouns.

## 2. Spelling and Unicode

Spelling errors are invisible to the eye and break the glossary check, which
compares strings literally (`tools/l10n/glossary.py`).

| Item | Rule | Example |
|---|---|---|
| Normalization | **NFC** (precomposed letters, *dựng sẵn*). Never combining marks (NFD, *tổ hợp*), never legacy encodings (TCVN3, VNI, Windows-1258). All 1,191 glossary terms are NFC; an NFD *ấ* is a different string and looks the same. | ✓ `ấ` = U+1EA5 ✗ `a` + U+0302 + U+0301 |
| Tone mark on *oa, oe, uy* | **Old style (*kiểu cũ*): hóa, hòa, họa, thỏa, tòa, khỏe, lũy, thủy, tùy, ủy, quý.** Reason: the glossary uses it in all of its about 45 affected terms (tối ưu hóa, chuẩn hóa, mã hóa, khóa, tích lũy, tọa độ, hòa trộn), it is the usual default of Vietnamese input methods and the spelling of most of the press, and *chuẩn hoá* would not match *chuẩn hóa* in the check. With a final consonant or a third vowel there is one spelling only: hoàn, hoạch, toán, thuyết, thuật, hoài, khuỷu. | ✗ tối ưu hoá, chuẩn hoá, toạ độ, luỹ thừa ✓ tối ưu hóa, chuẩn hóa, tọa độ, lũy thừa |
| *i* or *y* | As the glossary: *y* in the Sino-Vietnamese syllables kỳ, lý, kỹ, ký, tỷ, mỹ, quy (kỳ vọng, kỳ dị, lý thuyết, định lý, kỹ thuật, ký hiệu, tỷ lệ, quy tắc, quy nạp); *i* elsewhere (vi phân, phi tuyến, chi phí, di chuyển). | ✗ kì vọng, lí thuyết, qui tắc ✓ kỳ vọng, lý thuyết, quy tắc |
| đ | U+0111 / U+0110. | đạo hàm |
| Regional words | Standard (northern) written forms: nghìn (not ngàn), tỷ (not tỉ), vì sao / tại sao (not sao lại), này/ấy/đó (not nầy). | hàng nghìn bộ lọc |
| Invisible characters | None: no no-break or zero-width spaces, no soft hyphens (TRANSLATING.md §8). | |
| Loanwords | Original Latin spelling, lower case, no syllable hyphens and no respelling (§4.2). The assimilated forms are a closed list: nơ-ron, mô-đun, cosin, sin, logarit. | ✗ véc-tơ, vectơ, ten-xơ, gra-đi-en, tô-ken, nơron ✓ vector, tensor, gradient, token, nơ-ron |

## 3. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Quotation marks | Type straight `"…"` as the English source does; the build's smart typography prints “…”. Inside them `'…'`. Never «…», never ‘’ typed by hand. Inside a slide `title="…"`, which ASCII `"` delimits, type the curly “…”. | call cross-correlation "convolution" → gọi phép tương quan chéo là "tích chập" (F) |
| Dash | The English em dash `—` becomes a spaced en dash ( – ), at most once per sentence; a comma, a colon or parentheses are usually better. Never an unspaced `—` (F: ảnh đã trải phẳng—đúng là …), never ` — ` (O), never a hyphen with spaces. | ✓ … nhân với ảnh đã trải phẳng, đúng là dạng 2D của … (S) |
| Pairs and ranges | En dash without spaces joins two equal parts and the ends of a range. The hyphen is for assimilated syllables and names only (nơ-ron, Fashion-MNIST). | cặp đầu vào–đầu ra, bộ mã hóa–giải mã, dạng ma trận–vector, 40–55 |
| Comma | **No comma before *và*, *hoặc*, *hay* in a list**; a comma before *và* only where it joins two clauses with their own subjects. A comma after a fronted adverbial or clause (Trong mục này, …; Nếu …, thì …). | "Adam, RMSProp, and SGD" → Adam, RMSProp và SGD ; ✗ chuyển thành một kênh, và giảm mẫu (S) ✓ chuyển nó về một kênh và giảm mẫu (F) |
| Colon | Lower case after a colon, also in headings and slide titles; a capital only after a bold lead-in that the English follows with a sentence. No space before `:`, `;`, `?`, `!`. | ✓ Thực thi nhanh: tích chập bằng phép nhân ma trận (F) ✗ Thực thi nhanh: Tích chập … (S) ; **Đánh đổi:** Tốc độ cao nhất … |
| Semicolon | Allowed between two main clauses; a full stop is usually better. | |
| `&`, "vs." | *và*; *so với* (a contrast) or *và* (a pair). Vietnamese prose has no ampersand. | "Summary & Key Takeaways" → Tóm tắt và các ý chính ; "Cross-Correlation vs. Mathematical Convolution" → Tương quan chéo so với tích chập toán học |
| Abbreviations | Write out: "e.g." → ví dụ, chẳng hạn; "i.e." → tức là; "cf." → xem; "etc." → v.v. (only at the end of a list in parentheses, captions, slides); hand-written "et al." → và cộng sự (to confirm, §11). | "i.e., edges in the direction …" → tức là các biên theo hướng … |
| Capitals | Sentence case everywhere: the first word and proper names only. No capital on a term in running text, no title case in headings. This departs from the earlier edition, which as far as the drafter recalls capitalised content words in headings (to confirm, §14). | ✗ Mạng Nơ-ron Tích chập, ✗ Học Sâu ✓ mạng nơ-ron tích chập, học sâu |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s). A classifier added in Vietnamese (§5) and the English gloss stay outside the markers. Add none. | "the *two-dimensional cross-correlation*" → phép *tương quan chéo hai chiều* (F) |
| Spaces | Ordinary spaces in `24 GiB`, `10 000`; the build makes them no-break (§12). No space before `%`. | 24 GiB, 45% |

## 4. Terminology

### 4.1 How to choose

**Use the Vietnamese (mostly Sino-Vietnamese) term wherever Vietnamese
mathematics, statistics or computing has one; keep the English word, in Latin
spelling, where Vietnamese ML writing does; never coin a Sino-Vietnamese
compound and never respell an English word by sound.** The glossary decides
each case. For a term without an entry: apply the rule, prefer the term of a
Vietnamese university textbook over a blog's, and record it in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established Vietnamese term | Use it. | ma trận, đạo hàm, xác suất, kỳ vọng, phương sai, tích chập, trọng số, hàm mất mát, tốc độ học, quá khớp, lan truyền ngược, học tăng cường, tiên nghiệm, hậu nghiệm, trị riêng, chuyển vị |
| House terms of the earlier edition | Use them, even where other Vietnamese texts differ. Reason: they are consistent, were argued in public, and readers know them. | tầng (layer; ✗ lớp, which is "class"), hạ gradient (✗ giảm độ dốc, xuống đồi), điều chuẩn (regularization; ✗ chính quy hóa), hệ số điều chỉnh (the bias $b$; ✗ độ lệch), mạng nơ-ron hồi tiếp (RNN), gọt gradient, sải bước (stride), đệm (padding), gộp (pooling), hạt nhân (kernel), vùng tiếp nhận, khung chứa, chú ý (attention) |
| No Vietnamese term in use | English, lower case, roman, no plural *-s* (§4.2). | batch, minibatch, epoch, dropout, token, embedding, logit, prompt, perplexity, precision, recall, framework, notebook, actor, critic |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | As in English; never translated in part. | Transformer, ResNet, BERT, Adam, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, CNN, LSTM, GPU, GEMM, im2col, Q-learning |

### 4.2 English loans in a Vietnamese sentence

- **Lower case, roman, unchanged**: no italics, no quotation marks, no
  plural or tense ending. ✗ các tokens, ✗ *dropout*, ✗ "batch" ✓ các token,
  dropout, batch.
- **Give a loan a Vietnamese head noun** where it names a thing of a known
  kind; the head carries the grammar: hàm softmax, tầng dropout, vector
  embedding, kích thước batch, trục batch, số epoch, độ đo recall. The bare
  loan is fine once the kind is clear (sau 10 epoch, mỗi minibatch).
- **Verbs are Vietnamese**: ✗ fine-tune mô hình, ✗ train, ✗ sample từ $p$,
  ✗ update ✓ tinh chỉnh mô hình, huấn luyện, lấy mẫu từ $p$, cập nhật. A loan
  noun takes a Vietnamese verb: áp dụng dropout, tách token, chia batch.
- **Acronyms as the English writes them.** Where the English writes CNN, MLP,
  SGD, write CNN, MLP, SGD; do not expand them. ✓ trong một CNN sâu (F)
  ✗ trong một mạng nơ-ron tích chập sâu (O, for "a deep CNN"): the expansion
  makes slides too long, and the check accepts the acronym.

### 4.3 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text: plain parentheses, lower case as in the
glossary, roman, outside the emphasis markers, with no label such as "tiếng
Anh:". Check the English page in CONTEXT: if an earlier unit of the page
introduced the term, do not repeat it.

- **Gloss** a term the page defines or emphasises (`*…*` in the English) and
  every house term of §4.1 on its first use: *tương quan chéo hai chiều*
  (two-dimensional cross-correlation), tầng tích chập (convolutional layer),
  hệ số điều chỉnh (bias), *đệm* (padding), sải bước (stride), điều chuẩn
  (regularization). Reason: Vietnamese readers meet these concepts under their
  English names, and several Vietnamese renderings compete (tầng / lớp, điều
  chuẩn / chính quy hóa), so the English is what identifies the term.
- **Do not gloss** loans (token, dropout), names, or school vocabulary (ma
  trận, đạo hàm, xác suất, trung bình, điểm ảnh, cửa sổ). ✗ điểm ảnh (pixel),
  ✗ *cửa sổ* (window) (both S).
- With an acronym: mạng nơ-ron tích chập (convolutional neural network, CNN);
  afterwards write what the English writes.
- Never in headings, captions, slides, exercises or table cells.

On the test page F glossed nothing, O ten terms and S thirteen; the rule
above is closest to O.

## 5. Word order, classifiers, math and code

Vietnamese puts the head noun first and every modifier after it. Most errors
in the test came from keeping the English order around a symbol or a name.

| Case | Rule | Example |
|---|---|---|
| Noun + symbol | Head noun, then the symbol. | "the matrix $\mathbf{X}$" → ma trận $\mathbf{X}$ ; "the learning rate $\eta$" → tốc độ học $\eta$ |
| Noun + name | Head noun, then the name. | "the Adam optimizer" → bộ tối ưu Adam ; "a ReLU activation" → hàm kích hoạt ReLU ; "GPU memory" → bộ nhớ GPU ; "a Transformer block" → khối Transformer ; "an `nn.Linear` layer" → tầng `nn.Linear` ; ✗ Adam bộ tối ưu, ✗ GPU bộ nhớ |
| Noun + size | Head noun, then the size. | "a $2 \times 2$ kernel" → hạt nhân $2 \times 2$ ; "a $3 \times 3$ convolutional layer" → tầng tích chập $3 \times 3$ |
| Name + size, or two symbols | **Never two protected tokens side by side.** Put *cỡ*, *kích thước*, *là*, *ký hiệu* between them. | ✗ tensor $3 \times 3$ `X` (O, S) ✓ tensor đầu vào `X` cỡ $3 \times 3$ (F) ; ✗ hạt nhân $1 \times 2$ $\mathbf{K} = …$ (all three) ✓ hạt nhân $1 \times 2$ là $\mathbf{K} = …$ |
| Count + noun | Numeral or count symbol, then the noun; no classifier for abstract nouns. | "three stacked $3 \times 3$ layers" → ba tầng $3 \times 3$ xếp chồng ; "$L$ layers" → $L$ tầng ; "5 parameters" → 5 tham số |
| Ordinal | Noun + *thứ* + number or symbol. Math that already contains the ordinal is copied and takes no *thứ*. | "the $i$-th layer" → tầng thứ $i$ ; "the $i^{\textrm{th}}$ layer" → tầng $i^{\textrm{th}}$ (not reported in NOTES) |
| "-dimensional", "-fold", "-step" | Noun + symbol + unit word, no hyphen. | "a $d$-dimensional vector" → vector $d$ chiều ; "$k$-fold cross-validation" → kiểm định chéo $k$ phần ; "2D", "4D" stay as written; "two-dimensional" → hai chiều |
| Possession | *của* or nothing; never an apostrophe. | "Adam's learning rate" → tốc độ học của Adam ; "the kernel's weights" → trọng số của hạt nhân |

**Order of protected tokens.** The check compares the order of tokens of one
class (math with math, code with code) and warns on a change. The head-first
order often swaps a name and its size:

"Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$" →
✓ Cho ma trận đầu vào $\mathbf{X}$ kích thước $n_\textrm{h} \times n_\textrm{w}$ (S; reordered, named in NOTES)
✓ Cho ma trận đầu vào $n_\textrm{h} \times n_\textrm{w}$ là $\mathbf{X}$ (F; order kept)
✗ Cho ma trận đầu vào $n_\textrm{h} \times n_\textrm{w}$ $\mathbf{X}$ (O)

Write the first form, which is how a Vietnamese textbook states it, and name
the block in NOTES; use the second only where it reads as well. A name in
code and a size in math are different classes, so `X` cỡ $3 \times 3$ needs no
note. A sentence that runs through a display equation keeps its shape: "… is
given by $$…$$ where $n$ is …" → … có dạng $$…$$ trong đó $n$ là ….

**Classifiers.** Add the classifier that Vietnamese needs when a term is used
as a countable thing; the glossary term stays intact inside it, so the check
still finds it.

| Classifier | For | Examples |
|---|---|---|
| phép | an operation | phép tích chập, phép tương quan chéo, phép nhân ma trận, phép chuyển vị |
| hàm | a function | hàm mất mát, hàm softmax, hàm kích hoạt |
| bộ | a device or component | bộ lọc, bộ mã hóa, bộ tối ưu, bộ phát hiện biên |
| tính | a property | tính cục bộ, tính đẳng biến tịnh tiến, tính phi tuyến |
| độ | a measured quantity | độ chính xác, độ lệch chuẩn, độ giãn, độ sâu |
| thuật toán, phương pháp, mô hình, tập | as the noun says | thuật toán Adam, mô hình BERT, tập dữ liệu ImageNet |

"A convolution becomes a single matrix multiplication" → một phép tích chập
trở thành một phép nhân ma trận duy nhất (F, O, S). Do not add a classifier
where the term is a field or an uncountable notion: tích chập toán học lật
hạt nhân; học sâu.

## 6. Numbers and units

**Decision: the decimal point everywhere, the space as thousands separator.**
Numerals are copied from the English (0.5, 92.3%), in prose as in math, code,
code output and figures. Reason: math and code are protected and keep the
point, and outputs and figure axes show it; the Vietnamese decimal comma in
prose would put two conventions into one sentence (✗ với $\eta = 0.03$ mất
mát giảm xuống dưới 0,01) and the check reports `0,01` as a missing `0.01`.
All three test translators chose the point. With
the point as decimal sign, neither a point nor a comma may group thousands:
`10.000` reads as ten in the book's convention and `10,000` as ten in the
school convention; the space is unambiguous in both.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "at learning rate $0.03$ … below 0.01" → với tốc độ học $0.03$ … xuống dưới 0.01 |
| Thousands | A space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 ảnh ; 2048 ; 60 000 |
| Number words | thousand → nghìn, million → triệu, **billion → tỷ**, trillion → nghìn tỷ. "thousands of" → hàng nghìn. | "3 billion parameters" → 3 tỷ tham số |
| Digits or words | As the English: spelled out where the English spells out, digits with units, symbols and comparisons. | "nine steps" → chín bước ; "5 parameters" → 5 tham số |
| Zero | *không* also means "not". Write *số không* for the noun and *bằng không* / *bằng 0* for the value, and use the digit where *không* could be read as a negation. | "a border of zeros" → một viền số không ; "yields all zeros" → cho kết quả toàn số không (F) ; ✗ kết quả không |
| Percent | No space before `%`, as in the source. In prose after a spelled-out number: phần trăm. | "45% fewer parameters" → ít hơn 45% tham số |
| Units | Symbols as in the source, with a space; time words written out become giây, phút, giờ, ngày. | 24 GiB, 5 ms ; "3 hours" → 3 giờ |
| Ranges | En dash without spaces, unit once; with *từ* write *đến*. | "40–55" → 40–55 ; ✗ từ 40–55 ✓ từ 40 đến 55 |
| Multipliers | In prose *gấp đôi*, *gấp N lần*, *nhanh hơn N lần*; keep `×` in slides, tables, captions and math. | "about 2× faster" → nhanh hơn khoảng 2 lần ; slide: nhanh hơn 2× |
| Approximation | khoảng, xấp xỉ, gần; "an order of magnitude" → một bậc độ lớn. | "roughly 90%" → khoảng 90% |
| Ordinals | thứ nhất / đầu tiên, thứ hai, … ; digits after *thứ* from 10 on. | "the third epoch" → epoch thứ ba |
| Years, decades | Always with *năm*; decades as *thập niên*. | "in 2017" → vào năm 2017 ; "the 2010s" → thập niên 2010 |
| Money | Amount, space, ISO code; do not convert. | "\$2.50" → 2.50 USD |

Quantities, signs and orders of magnitude stay exact. What the check does
(`tools/l10n/protect.py`): it compares the numerals of each English block with
the translation's, outside protected tokens, and warns when one is missing. A
numeral counts with its `.` or `,` (so `0,5` ≠ `0.5`, `10.000` ≠ `10,000`);
with `thousands: space` (§12) the English comma and the Vietnamese space are
removed first, so `10,000` → `10 000` passes and a kept `10,000` warns. `N×`
and ranges are compared by their digits only. A digit the translation adds
(`bằng 0` for "zero") is not reported.

## 7. Cross-references

The build prints the prefix (§12): Chương 3, Mục 5.1, Hình 4.2, Bảng 2.1,
Công thức (7.2.1) (to confirm, §11). Vietnamese nouns do not inflect, so a
reference fits any position; the only rules are to write no prefix or
classifier of your own and to choose a preposition that fits every kind.

| English | Vietnamese |
|---|---|
| "In :numref:`fig_correlation`, we slide …" | Trong :numref:`fig_correlation`, ta trượt … (F) |
| "as shown in :numref:`fig_x`, the loss falls" | Như :numref:`fig_x` cho thấy, mất mát giảm. |
| "see :numref:`sec_sgd`" | xem :numref:`sec_sgd` ; ✗ xem mục :numref:`sec_sgd`, ✗ trong hình :numref:`fig_x` |
| "We implement :eqref:`eq_corr2d` directly" | Ta lập trình trực tiếp :eqref:`eq_corr2d` (O) ; ✗ công thức :eqref:`…`, ✗ phương trình :eqref:`…` |
| "panel (c) in :numref:`fig_x`" | phần (c) của :numref:`fig_x` ; ✗ hình (c) trong :numref:`fig_x` (F, O: prints "hình (c) trong Hình 7.1") ✗ panel (c) (S) |
| ":citet:`Luo.Li.Urtasun.ea.2016` showed that …" | :citet:`Luo.Li.Urtasun.ea.2016` đã chỉ ra rằng … (the citation is the subject, as in English) |
| "(§{sec_channels})" on a slide | (§{sec_channels}), copied; no *mục* before it |

- "this section" → mục này; "this chapter" → chương này; "this book" → cuốn
  sách này; "the previous section" → mục trước. **Section is *mục*, not
  *phần***: *Phần* is the book's Part (Phần Cơ bản, Phần Nâng cao), and all
  three test translators wrote *trong phần này* for "in this section".
- A citation `:cite:` stands at the end of the clause it supports, before the
  comma or full stop, as in English.
- A hand-written reference to another work keeps that work's numbering with a
  capitalised Vietnamese word: "Chapter 6" → Chương 6.

## 8. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, Tensor Core.
- **People:** as in the English source, without title and without
  transliteration: Gauss, Markov, Bayes, Kolmogorov, Chebyshev. ✗ Gau-xơ,
  ✗ Mác-cốp. Vietnamese names are written with their diacritics when the
  source has them.
- **Terms named after people:** Vietnamese noun + the name, unchanged, no
  possessive and no adjective ending: phân phối Gauss, nhiễu Gauss, định lý
  Bayes, tối ưu hóa Bayes, chuỗi Markov, quá trình quyết định Markov, bất
  đẳng thức Jensen, nhân tử Lagrange, khai triển Taylor, khoảng cách Euclid,
  liên tục Lipschitz. "Gaussian" → Gauss, "Bayesian" → Bayes, "Markovian" →
  Markov, "Euclidean" → Euclid (✗ Gaussian, ✗ Euclide).
- **Titles of courses, assignments, papers and books:** in English as cited:
  *Phỏng theo Stanford CS231n, Assignment 2, notebook ConvolutionalNetworks*
  (F; ✗ sổ tay ConvolutionalNetworks, S).
- **The book:** *Dive into Deep Learning* stays in English.
- **Languages and places:** Vietnamese (tiếng Anh, Mỹ, Trung Quốc);
  programming languages stay (Python).
- **Structure words:** part Phần, chapter Chương, section Mục, subsection
  tiểu mục, appendix Phụ lục; Summary → Tóm tắt, Exercises → Bài tập,
  Discussions → Thảo luận; Theorem → Định lý, Proposition → Mệnh đề, Lemma →
  Bổ đề, Corollary → Hệ quả, Definition → Định nghĩa, Proof → Chứng minh,
  Remark → Nhận xét, Example → Ví dụ.

## 9. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | A noun phrase or a bare verb phrase, sentence case, no full stop, no English gloss, no leading *Các/Một/Sự/Việc* for "The/A/-ing". English "-ing + object" becomes verb + object. A question stays a question. | "Convolutions for Images" → Tích chập cho ảnh ; "Learning a Kernel" → Học một hạt nhân ; "Training the Model" → Huấn luyện mô hình ; "Implementation" → Lập trình ; "Why Depth Helps" → Vì sao độ sâu có ích |
| `caption` | Body register without *chúng ta*/*bạn*; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Two-dimensional cross-correlation." → Phép tương quan chéo hai chiều. |
| `slide_title`, `kicker` | As short as the English, sentence case, no full stop; *và* for `&`. | `title="Detecting Local Edges Across an Image"` → `title="Phát hiện biên cục bộ trên toàn ảnh"` ; `[Sliding-Window Inner Products]{.kicker}` → `[Tích trong trên cửa sổ trượt]{.kicker}` |
| `slide_text` | Short statements; no *chúng ta*; drop *một*/*các*; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and colon. In quiz slides "you" → *bạn*. | "**Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch." → **Một tầng ($L=1$):** Hạt nhân $k_1 \times k_1$ nhìn thấy một mảnh đầu vào $k_1 \times k_1$. (O) |
| `exercise` | An instruction starts with *Hãy* + verb, or with *Cho* …, *hãy* … ; a question is a direct question; "you" → *bạn*; "Why?" → Vì sao? The bold title stays bold; `[code]` and ● are copied. Reason: *Hãy* is the imperative of Vietnamese textbooks; the bare verb reads as a note. | "Design some kernels manually." → Hãy tự tay thiết kế một số hạt nhân. (F) ; "Derive a finite difference operator for the second derivative." → Hãy suy ra một toán tử sai phân hữu hạn cho đạo hàm bậc hai. |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → `nn.LazyLinear(1)` của PyTorch chỉ cần số chiều đầu ra. |
| `note` | Register of its surroundings. Labels: **Note:** → **Lưu ý:**, **Warning:** → **Cảnh báo:**, **Tip:** → **Mẹo:**, **Answers:** → **Đáp án:**, "Concept Check" → Kiểm tra khái niệm, "Adapted from" → Phỏng theo | |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → Thời gian huấn luyện |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 10. Known pitfalls

### 10.1 Words with more than one sense

| English | Sense → Vietnamese | Not |
|---|---|---|
| bias | parameter $b$ → hệ số điều chỉnh; of an estimator → độ chệch, *unbiased* → không chệch; inductive bias → thiên kiến quy nạp; social → thiên kiến | độ lệch, thiên vị for $b$ |
| layer, class | layer → tầng (tầng ẩn, tầng tích chập); class of a classifier → lớp; Python class → lớp | lớp for a layer |
| kernel | convolution → hạt nhân (kích thước hạt nhân); kernel methods and covariance functions → hàm hạt nhân; GPU or OS code → kernel | nhân, lõi; nhân tích chập |
| edge | in an image → biên (phát hiện biên, bộ phát hiện biên, bản đồ biên: F's choice, and to the drafter's knowledge the term of Vietnamese image-processing textbooks); the border of an image → viền, mép; of a graph, of a square → cạnh | cạnh for an image edge (O, S); dò cạnh |
| shape, size | of a tensor → kích thước (glossary); "reshape" → đổi kích thước; the form of a curve → dạng, hình (hình chuông); a count along one axis → số phần tử, độ dài | hình dạng for a tensor (to confirm, §11) |
| unit, node, neuron | of a network → nút (nút ẩn); neuron → nơ-ron; of measurement → đơn vị | đơn vị for a network unit (S) |
| dense | dense layer = fully connected layer → tầng kết nối đầy đủ; dense matrix or array → ma trận / mảng dày đặc | tầng dày đặc (F, S; to confirm, §11) |
| scalar | noun → số vô hướng; adjective → vô hướng: "a scalar bias" → hệ số điều chỉnh vô hướng (O, S; the glossary warning is a false match, name it in NOTES) | hệ số điều chỉnh là số vô hướng (F) |
| attention, note | the mechanism → chú ý (cơ chế chú ý, tự chú ý); "Note that", "pay attention" → lưu ý, never *chú ý rằng* | chú ý for both |
| chuẩn | norm → chuẩn (chuẩn $\ell_2$); normal distribution → phân phối chuẩn, standard normal → phân phối chuẩn tắc; normalization → chuẩn hóa; standard deviation → độ lệch chuẩn. So "standard" as an ordinary word is never *chuẩn*: thông dụng, tiêu chuẩn. | thuật ngữ chuẩn của học sâu (O, S) ✓ thuật ngữ thông dụng của học sâu (F) |
| regularization, normalization | điều chuẩn; chuẩn hóa | chính quy hóa; mixing them |
| translation, shift | of an image or function → tịnh tiến (tính đẳng biến tịnh tiến, bất biến tịnh tiến); a shift by $\Delta$ → phép dịch, độ dời; distribution shift → dịch chuyển phân phối; of language → dịch (dịch máy) | bất biến dịch thuật |
| implement, install | implement → lập trình (F, O); install → cài đặt | cài đặt for "implement" (S): it is the Installation chapter's word |
| loss, cost, objective, error, risk | mất mát (hàm mất mát); chi phí; hàm mục tiêu; rủi ro. Error: lỗi for training/test/generalization error (glossary), sai số for a numerical quantity (sai số bình phương, sai số tương đối, sai số làm tròn). | tổn thất; merging them |
| accuracy, precision, recall | độ chính xác; precision; recall. Floating-point precision → độ chính xác (độ chính xác kép). | độ chuẩn xác, độ phủ |
| sample, example | a draw, a data point → mẫu (lấy mẫu, mẫu ngẫu nhiên); "for example" → ví dụ, chẳng hạn | thí dụ; ví dụ for a data point |
| feature | đặc trưng (bản đồ đặc trưng, trích xuất đặc trưng); of a library or product → tính năng | tính năng for an input feature |
| return | RL → tổng phần thưởng tích lũy (glossary; to confirm, §11); of a function → trả về, giá trị trả về; "black returns to white" → trở lại | lợi nhuận |
| policy, state, action, reward, agent, environment | chính sách, trạng thái, hành động, phần thưởng, tác nhân, môi trường | đại lý (agent), củng cố (reinforcement): both from the machine-translated text |
| value | RL → giá trị (hàm giá trị); attention → giá trị; key → khóa; query → truy vấn | |
| gradient, derivative | gradient; đạo hàm, "with respect to $x$" → theo $x$ | độ dốc; đối với $x$ |
| expectation, expected | $\mathbb{E}[\cdot]$ → kỳ vọng; "as expected" → đúng như dự đoán | kỳ vọng for the ordinary word |
| likelihood | hàm hợp lý, hợp lý cực đại | khả năng, xác suất |
| generalization | khái quát hóa (lỗi khái quát, khoảng cách khái quát: glossary) | (to confirm, §11) |
| training, learning | train → huấn luyện; learn → học (tốc độ học, học có giám sát); "learned kernel" → hạt nhân được học; "learnable" → học được; pretraining → tiền huấn luyện; fine-tuning → tinh chỉnh | đào tạo, luyện |
| memory | hardware → bộ nhớ; LSTM → bộ nhớ ngắn hạn dài (glossary) | trí nhớ for RAM |
| performance | of a model → chất lượng, kết quả; of code → hiệu năng, tốc độ | hiệu suất for both |
| stack, flatten, pad, pool | xếp chồng; trải phẳng; đệm; gộp | |
| window, patch, entry | cửa sổ; mảnh (ma trận mảnh); phần tử | bản vá (patch) |
| order | second-order → bậc hai; order of magnitude → bậc độ lớn; sequence order → thứ tự | |
| converge | hội tụ về $x^*$ | hội tụ tới/đến alternating |

### 10.2 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "Note that" every time as *Lưu ý rằng* | Drop it, or *Lưu ý rằng* at most once per page | "Notice that the output shrinks slightly" → Đầu ra co lại một chút … |
| "Recall that" | Nhớ lại rằng … ; Như đã thấy ở … | |
| "It is … that", "There is/are" | The plain statement; Có … | "There are three reasons." → Có ba lý do. |
| Passive with *bởi* | Active | ✗ Hạt nhân được học bởi hạ gradient. ✓ Hạ gradient học được hạt nhân. |
| Gerund subject as *Việc …* | Verb-initial clause, or *Khi …, ta …* | "Sweeping the detector across … produces an edge map." ✗ Việc quét bộ phát hiện … tạo ra … ✓ Quét bộ phát hiện … qua mọi vị trí hợp lệ trên ảnh, ta thu được một bản đồ biên. (F) |
| Inanimate subject + "tells us", "allows us to" | Từ đó ta thấy, Nhờ đó ta có thể | ✗ Kết quả này nói với chúng ta rằng … |
| "we can" every time as *chúng ta có thể* | The verb itself, or *có thể* without subject | "we can unroll the kernel" → ta có thể trải hạt nhân ; "can be evaluated as" → tính được bằng |
| "respectively" | lần lượt | |
| "in terms of", "based on" | theo, về; dựa trên | ✗ trong các điều khoản của |
| "simply", "just" | Drop, or *chỉ* | |
| Idioms translated by image | The meaning | "brings us full circle to" ✗ đưa ta trở lại trọn vòng với (O) ✓ đưa ta trở lại đúng điểm xuất phát (F) ; "under the hood" → ở bên trong ; "from scratch" → từ đầu ; "out of the box" → dùng được ngay ; "rule of thumb" → quy tắc kinh nghiệm ; "trade-off" → sự đánh đổi ; "end to end" → từ đầu đến cuối (adverb), đầu-cuối (modifier: huấn luyện đầu-cuối) |
| Technical "fires", "sees", "spikes" | A plain verb where the image fails | "`Y` fires `+1` along …" → `Y` cho giá trị `+1` dọc theo … (F) |
| English order around a symbol or name | §5 | ✗ $3 \times 3$ tensor `X` |
| Title Case, plural *-s*, apostrophes | Vietnamese forms | ✗ Tầng Tích Chập ✗ các kernels ✗ Adam's |

Logical and mathematical phrases are fixed: because → vì, do; therefore → do
đó, vì vậy; however → tuy nhiên; whereas → trong khi … thì; if … then → nếu …
thì; only if → chỉ khi; if and only if → khi và chỉ khi; at most → nhiều
nhất, không quá; at least → ít nhất; "let $x$ be …" → Cho $x$ là …, Giả sử
$x$ là …; such that → sao cho; where (after a formula) → trong đó; with
respect to → theo (derivative), đối với; without loss of generality → không
mất tính tổng quát; for all → với mọi; there exists → tồn tại; up to sign →
sai khác về dấu; almost surely → hầu chắc chắn; i.i.d. → độc lập và cùng
phân phối (then i.i.d., as the English); bounded → bị chặn; smooth → trơn;
dot product → tích vô hướng; elementwise → theo từng phần tử.

## 11. For the author to confirm

1. **Thousands separator** (§6): a space (10 000), which needs `check:
   {thousands: space}`. Alternative: keep the English comma (10,000), which
   needs no setting but reads as a decimal to a reader schooled in Vietnam.
2. **Section = *Mục*, Part = *Phần*** (§7, §8), against *phần* for "section"
   in all three test translations.
3. **Printed prefixes** (§7, §12): Chương, Mục, Hình, Bảng, capitalised, and
   *Công thức* (not *Phương trình*) before an equation number, or no word at
   all before it. I do not know whether the build prints a word before
   `:eqref:` numbers in the body; the rule in §7 works either way.
4. **Image edge = *biên*** (§10.1), not *cạnh*. Proposed: biên.
5. **Dense layer = *tầng kết nối đầy đủ*** (§10.1), one term for "dense" and
   "fully connected" layer. Alternatives: tầng dày đặc (F, S), tầng kết nối
   dày đặc (O).
6. **Shape and size both *kích thước*** (§10.1; question 6 of
   `glossary/review/vi.md`). Proposed: keep the verified *kích thước*;
   *hình dạng* would separate the two but departs from the earlier edition.
7. **Generalization: *khái quát hóa*** (glossary) or *tổng quát hóa* (review
   question 3). Proposed: khái quát hóa, as the two verified compounds.
8. **Error: *lỗi* / *sai số*** split as in §10.1 (review question 7).
9. **RL return: *tổng phần thưởng tích lũy*** is four syllables too long for
   a term used hundreds of times in ch. 14–15. Proposed: *lợi tức* with
   (return) at first use; needs a reader who knows the Vietnamese RL
   literature (review question 5).
10. **English gloss** (§4.3): once per page, for defined terms and house
    terms only. The earlier edition's practice was not checked.
11. **"et al." → *và cộng sự*** in hand-written text (§3), and what the build
    prints for `:citet:` ("Luo et al. (2016)" or a Vietnamese form).
12. **Quotation marks** (§3): relies on the build turning straight quotes
    into “…” for `lang: vi`; not tested.
13. **Tone-mark style** (§2): old style (hóa, thủy). The Ministry of
    Education's school-textbook rules prefer the new style (hoá, thuỷ); the
    glossary, the press and input-method defaults use the old one. Changing
    it later means respelling about 45 glossary terms.
14. ***chúng tôi* for the authors alone** (§1), in the preface and wherever
    the English "we" excludes the reader.
15. **Loan list** (§4.1): framework, notebook, prompt, actor, critic kept in
    English; *lập trình* for "implement".

## 12. Settings

`languages.yml`, existing keys (`tools/l10n/config.py`, `protect.py`):

- `check: {thousands: space}`: accept `10 000` for the English `10,000` and
  warn on a kept comma (§6). The defaults stay for the rest (`numbers:
  warning`, `order: warning`, `cjk_adjacent: letters`, which never fires for
  Vietnamese).
- `typography: {nbsp: true}`: no-break spaces in grouped numbers (10 000) and
  between number and unit (24 GiB). Its abbreviation rule is written for
  German letters and does nothing in Vietnamese.
- `dir: ltr`, `forum: english`, as now.

`vi/strings.yml` (created by `l10n strings vi`), existing keys:

- `lang: "vi"`; `pdf.lang`: `"vi"` only if babel's Vietnamese module builds
  with XeLaTeX on the build host (not tested), else `null`.
- Quarto ships no `_language-vi.yml` (checked in `.venv-build`), so every
  Quarto string falls back to English unless overridden:
  `quarto.crossref-ch-prefix: "Chương"`, `quarto.crossref-sec-prefix: "Mục"`,
  `quarto.crossref-fig-prefix` and `-fig-title: "Hình"`,
  `quarto.crossref-tbl-prefix` and `-tbl-title: "Bảng"`,
  `quarto.crossref-eq-prefix: "Công thức"`,
  `quarto.section-title-references: "Tài liệu tham khảo"`; the remaining
  Quarto keys (callout titles, table of contents, search) need the same.
- `book.exercises_heading: "Bài tập"`, `book.discussions_link: "Thảo luận"`.
- Fonts: the defaults (Source Sans 3, Source Serif 4, JetBrains Mono,
  Inconsolata) are expected to cover Vietnamese; confirm that the PDF fonts
  in `static/fonts/` contain U+1EA0–U+1EF9 before the first PDF build.

Settings that would need new code:

- **A normalization check**: reject a translated block that is not NFC, and
  warn on new-style tone placement (*oà, oá, oả, oã, oạ, oè, oé, oẻ, oẽ, oẹ,
  uỳ, uý, uỷ, uỹ, uỵ* at the end of a syllable not after *q*). The glossary
  check compares strings literally, so today such a block only shows up as a
  false "term absent" warning.
- **NFC-normalizing both sides** in `glossary.py` `_present`, so that the
  glossary check does not depend on the first item.
- **A smart-quote test for `lang: vi`** in the build (question 12).

## 13. Proposed glossary corrections

None of these is applied; `glossary/terms/vi.jsonl` was not edited.

| Concept | Now | Proposed | Reason |
|---|---|---|---|
| kernel size | kích thước nhân tích chập (proposed) | kích thước hạt nhân | *kernel* is the verified *hạt nhân*; all three test translators overrode the entry. |
| scalar | số vô hướng (verified) | add the alternative *vô hướng* | Adjectival use ("scalar bias", "scalar loss") otherwise warns every time (§10.1). |
| off-policy, on-policy | ngoài chính sách (off-policy), theo chính sách (on-policy) | the term without the parenthesis; the gloss is the guide's job (§4.3) | A parenthesis inside a term is inserted at every occurrence. |
| soft actor-critic | actor-critic mềm (SAC) | actor-critic mềm | Same. |
| softmax, sigmoid | hàm softmax, hàm sigmoid (proposed) | softmax, sigmoid | The classifier is added by §4.2 where needed; "hồi quy softmax" and "chuẩn hóa softmax" are already entries without it. |
| return (RL) | tổng phần thưởng tích lũy (proposed) | lợi tức (to confirm, §11.9) | Length. |
| null space | không gian nghiệm không (proposed) | không gian hạt nhân (to confirm) | The review calls the current form odd; a specialist should choose. |
| Hessian, Jacobian | ma trận Hessian, ma trận Jacobi | one pattern for both: ma trận Hesse, ma trận Jacobi (to confirm) | §8: noun + the person's name. |
| standard normal distribution | no entry | phân phối chuẩn tắc | F and S chose it independently; O wrote phân phối Gauss chuẩn tắc. |
| edge (image), edge detection | no entry | biên, phát hiện biên | §10.1; the three translators split between biên and cạnh. |
| dense layer | no entry | tầng kết nối đầy đủ (to confirm, §11.5) | Three translators, three renderings. |
| patch, patch matrix, flatten, banded matrix, effective receptive field | no entries | mảnh, ma trận mảnh, trải phẳng, ma trận dải, vùng tiếp nhận hiệu dụng | All three translators agree. |
| implement | no entry | lập trình | §10.1; S wrote cài đặt. |

## 14. Sources

Read for this draft (2026-11-26), all in this repository:

- `glossary/terms/vi.jsonl` (1,191 entries; spelling counts in §2 are from a
  scan of the `preferred` fields), `glossary/review/vi.md` (provenance of the
  earlier edition's terms, its house choices, the open questions cited in
  §11), `glossary/notes/east-asian.md`.
- `evals/vi-conv-layer/`: RESULTS.md and the answers of Fable, Opus and
  Sonnet with their DECISIONS and NOTES; the quoted sentences and the counts
  (*chúng ta* / *ta*, glosses, dashes) are from these files.
- `tools/l10n/protect.py`, `glossary.py`, `config.py`, DESIGN.md §10,
  TRANSLATING.md; d2l-neu `tools/i18n.py` and the Quarto language directory
  of `.venv-build` (no Vietnamese file).

Not checked (stated from the drafter's knowledge; no web source was read):

- The earlier edition's `STYLE_GUIDE.md` and chapter text. Its terms are
  taken from what `glossary/review/vi.md` records (the house terms of §4.1);
  its heading capitalisation, its decimal sign, its glossing practice and its
  word for "implement" were not looked up.
- Vietnamese textbook usage for *biên* (image edge), *phân phối chuẩn tắc*,
  *mục* / *phần*, *và cộng sự*, and the old/new tone-placement rules.
- Vietnamese number norms (decimal comma and thousands point in general
  publishing; the space of the SI standard).
- Font coverage, babel Vietnamese under XeLaTeX, and pandoc's quote
  rendering for `lang: vi`.
