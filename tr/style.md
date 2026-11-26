# Style guide: Turkish (tr)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Turkish, or ✗ wrong / ✓ right). Examples marked with a translator's
name are real sentences from the translation test `evals/tr-conv-layer/`.
Points that still need a ruling are marked "(to confirm)" and collected in §12.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, spell and
inflect*. Spelling follows the TDK *Yazım Kılavuzu* and *Güncel Türkçe Sözlük*
except where this guide departs from them on purpose (numbers, §5).

## 1. Register

**Decision: authorial *biz*, the reader as *siz*, never *sen*.** Reason: *biz*
is the voice of Turkish university textbooks and of the earlier Turkish
edition (tr.d2l.ai), and it carries the book's "we work through this together"
tone (TRANSLATING.md §8); *sen* is the tone of blogs and video tutorials. The
person shows in the verb ending only: do not write the pronouns *biz*, *siz*
or *o* as a subject unless there is a contrast.

| English | Turkish | Example |
|---|---|---|
| we + action the text performs now | *-(I)yoruz* | "We implement … in `corr2d`." → … `corr2d` içinde gerçekleştiriyoruz. |
| we + a general procedure, a definition, a fact | aorist *-(I)r / -(A)r* | "We multiply … and sum the products." → … çarpar ve çarpımları toplarız. ; "The kernel defines a window." → Çekirdek bir pencere tanımlar. |
| "let us", "let's", "consider" | *-(y)AlIm* | "Let us look at the gradient." → Gradyana bakalım. ; "Consider locating vertical edges …" → … dikey kenarları bulmak istediğimizi düşünelim. (Fable) |
| "we will see", "we will return to" | *-(y)AcAğIz* | "… which we examine in :numref:`sec_padding`" → … onu :numref:`sec_padding` içinde inceleyeceğiz. |
| "we derived", "recall that we" | *-DIk*; *-mIştIk* for a reminder of an earlier section | "we derived the convolutional layer" → evrişim katmanını … türettik |
| "you" = the reader (advice, a question, exercises) | second person plural: *-(y)In*, *-sInIz*, *-(I)nIz* | "Notice that the output shrinks." → Çıktının … küçüldüğüne dikkat edin. ; "Suppose you want to estimate the price of a house." → Bir evin fiyatını tahmin etmek istediğinizi varsayalım. |
| "you can", "one can" = anyone | *-(y)Abiliriz*, or passive *-(y)Abilir* | "You can verify this numerically." → Bunu sayısal olarak doğrulayabiliriz. |
| "we recommend", "we believe" | *biz* form | "We recommend AdamW." → AdamW kullanmanızı öneririz. |
| "we" = people, the field | *biz insanlar*, *araştırmacılar*, or a passive | "We humans attend selectively." → Biz insanlar dikkatimizi seçici biçimde yöneltiriz. |
| "he or she", "they" (a person) | *o*, or the plural; Turkish has no grammatical gender, so nothing else is needed | "the practitioner … she" → uygulayıcı … |

**Sentence shape** (each point is checkable):

- **The finite verb ends the sentence.** No inverted sentences (*devrik
  cümle*) in body text. ✗ Türettik evrişim katmanını iki ilkeden.
- **No officialese endings.** No *-mAktAdIr* and no *-mIştIr* as the main verb
  of a body sentence. ✗ Bu katmanda bir çekirdek kullanılmaktadır. ✓ Bu katman
  bir çekirdek kullanır. The copula *-DIr* stays in definitions (… bir
  matristir).
- **Active with the real subject.** *tarafından* only where the agent must be
  named and cannot be the subject (✓ :citet:`…` tarafından önerilen model,
  §2). ✗ Kayıp, model tarafından en aza indirilir. ✓ Model kaybı en aza
  indirir.
- **Converbs, not *ve* chains.** Two verbs with one subject are joined with
  *-(y)Ip* or *-(y)ArAk*; at most two *ve* in a sentence. "We construct `X`
  and evaluate `corr2d(X, K)`." → `X` tensörünü oluşturup `corr2d(X, K)`
  ifadesini hesaplıyoruz.
- **At most two participle clauses (*-An*, *-DIğI*, *-AcAğI*) before one
  noun, and at most three nouns in a genitive chain.** Otherwise split the
  sentence; TRANSLATING.md §3 allows splitting within a block. ✗ ağın ilk
  katmanının çıktısının şekli ✓ ağın ilk katmanından çıkan tensörün şekli.
- **English "which", "where", "so that" start a new Turkish sentence**
  (Bu, …; Burada …; Böylece …) when the clause would otherwise be the third
  one before its noun.
- **Singular after a numeral or a quantifier, singular verb for a plural
  thing.** ✗ 3 katmanlar, birçok çekirdekler, ağırlıklar güncellenirler
  ✓ 3 katman, birçok çekirdek, ağırlıklar güncellenir.
- **No *bir* for every "a/an".** Write *bir* only where it means "one" or
  introduces something new. "Learning a Kernel" ✗ Bir Çekirdeği Öğrenme
  ✓ Çekirdek Öğrenme.
- **No *ki* clauses and no filler.** ✗ Görüyoruz ki model yakınsıyor.
  ✓ Modelin yakınsadığını görüyoruz. Drop literal "simply", "just",
  "basically" (✗ basitçe, sadece … -iz, temel olarak).

**Spelling.** TDK forms. The circumflex is written where TDK requires it for
meaning or pronunciation (kâğıt, hâl, hâlâ, kâr, zekâ, hikâye), not on the
adjective ending *-i* (merkezi, yerel, nihai; the glossary writes *merkezi
limit teoremi*). Written apart: the conjunctions *de/da* (never *te/ta*) and
*ki*, the question particle *mı/mi/mu/mü*, *her şey*, *ya da*, *art arda*, *üst
üste*, *fark etmek*, *elde etmek*, *var olmak*. Written together: *birkaç*,
*birçok*, *hiçbir*, *herhangi* (*herhangi bir*), *yalnızca*.

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Apostrophe | Type the ASCII `'` (U+0027); the build turns it into ’. Reason: one form to type and to search for; tested with the book's pandoc (`GPU'lar`, `$k$'nin`, `2'nin` all come out as ’). When it is used is §3. | ✓ `CNN'de` ✗ `CNN’de`, ✗ `CNN´de` |
| Quotation marks | In running text type ASCII `"…"`; the build makes “…” (TDK's form). Inside a quotation ‘…’, typed `'…'`. Never «…» or „…“. Book titles in italics as in the source. A suffix follows the closing quote without an apostrophe (TDK), but prefer a wording that needs none. | call cross-correlation "convolution" → çapraz korelasyona "evrişim" diyoruz |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it type “…” (U+201C, U+201D). | `title="“Derin” Ne Demek?"` |
| Dash | The English em dash `—` is not used. Write a comma, a colon or parentheses; a new sentence if the insertion is long. Never a hyphen with spaces. | "the bias—an offset—shifts the plane" → ek girdi (bir kayma) düzlemi öteler |
| Comma | Turkish rules: none before or after *ve*, *veya*, *ya da*, *hem … hem* (no "and" comma in a list); none after a converb (*-Ip*, *-ArAk*, *-IncA*, *-DIğIndAn*) unless the clause is long and the sentence is misread without it; none after *ancak*, *fakat*, *ama*, *çünkü*, *yani* at the start of a sentence. A comma after a long subject that could be read as a modifier of the next word, and after a fronted connective phrase (Bununla birlikte, Buna karşılık, Öte yandan, Örneğin). | "Adam, RMSProp, and SGD" → Adam, RMSProp ve SGD ; ✗ Ancak, model … ✓ Ancak model … ; Çekirdek, şekli $k_\textrm{h} \times k_\textrm{w}$ olan kayan bir *pencere* tanımlar. (Fable) |
| Colon | A capital after a colon when a full sentence follows, lower case before a list of words (TDK). | **Temel varsayım:** Hatalar bağımsızdır. ; üç büyüklük: ağırlık, ek girdi ve öğrenme oranı |
| Semicolon | Allowed between two main clauses; do not copy every English semicolon, a full stop is usually better. | |
| Question mark | Every direct question ends in `?`, also a heading or slide title that the Turkish makes a question (§8). | |
| Abbreviations | In body text write *örneğin*, *yani*, *yaklaşık*; *vb.* is allowed in body text. *bkz.*, *örn.*, *yak.*, *krş.* only inside parentheses, captions, slides and tables. "i.e." → yani; "e.g." → örneğin; "etc." → vb. (✗ vs., ✗ v.s.); "cf." → krş.; "et al." in visible text → vd.; "vs." → *ve*, *ile* or *karşısında*. | "Theoretical vs. Effective Receptive Fields" → Kuramsal ve Etkin Alıcı Alanlar |
| `&` | → *ve* | "Summary & Key Takeaways" → Özet ve Çıkarımlar |
| Spaces | No invisible characters (TRANSLATING.md §8). Type ordinary spaces in `10 000` and `24 GiB`; see §11 for the build setting that keeps them on one line. No space after `%` (§5). | ✓ `%92.3`, `24 GiB` |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word; add none. Where the English defines a term, phrase the sentence so the term stands in its dictionary form. The English gloss stays outside the markers. | ✓ `*çekirdeğin*` ✗ `*çekirdek*in`, ✗ `*çekirdeğ*in` ; … buna `*dolgu* (İng. padding)` denir |
| Headline case | §8. | |
| Line starts | Never begin a line or a block with a number and a full stop, or with `-`, `+`, `#`, `>`: Markdown reads a list item, a heading or a quotation. Opus hit this in the test with "6. Bölüm". | ✗ `6. Bölüm ve …` at a line start ✓ `Bölüm 6 ve …` |

### 2.1 References: no suffix, and no second noun

The Turkish build prints the word and the number: ``:numref:`fig_x` `` →
Şekil 6.2.1, ``:numref:`tab_x` `` → Tablo 6.1, ``:numref:`sec_x` `` and
``:numref:`chap_x` `` → Bölüm 6.2 and Bölüm 6, ``:eqref:`eq_x` `` →
Eşitlik (6.2.1); `§{sec_x}` prints §6.2. The translator sees neither the word
nor the number, and a Turkish case suffix depends on how the last digit is
read (Şekil 6.2.1'de, 6.2.3'te, 6.2.6'da).

**Decision: never attach a suffix or an apostrophe to a reference, and never
add the noun the build already prints.** Use, in this order of preference:

| Construction | Example |
|---|---|
| The reference as subject (no case needed) | "as shown in :numref:`fig_x`, the loss falls" → :numref:`fig_x` kaybın düştüğünü gösteriyor. ; :numref:`fig_conv_im2col` bunu … üzerinde gösterir. |
| In parentheses, after the statement; *bkz.* only for "see" | "… expect 4D tensors (:numref:`sec_channels`)" → … 4B tensörler beklediğinden (:numref:`sec_channels`) … ; "see :numref:`sec_sgd`" → (bkz. :numref:`sec_sgd`) |
| A postposition that takes the bare form: *içinde*, *içindeki*, *ile*, *için*, *üzerinde*, *boyunca*, *uyarınca* | "In :numref:`fig_correlation`, we slide …" → :numref:`fig_correlation` içinde … kaydırıyoruz. (Fable) ; "the derivation of :numref:`sec_why-conv`" → :numref:`sec_why-conv` içindeki türetim ; "The output reproduces :numref:`fig_correlation`" → Çıktı :numref:`fig_correlation` ile … örtüşür. |
| For an equation that is an object: *ile verilen* + a noun that is not *denklem* or *eşitlik* | "We implement :eqref:`eq_corr2d` in `corr2d`." → :eqref:`eq_corr2d` ile verilen işlemi `corr2d` içinde gerçekleştiriyoruz. |

- ✗ ``:numref:`sec_why-conv` bölümünde`` (four of the six test translations):
  prints "Bölüm 6.1 bölümünde". ✗ ``:numref:`fig_x` şeklinde``, ✗
  ``:numref:`tab_x` tablosunda``, ✗ ``:eqref:`eq_corr2d` denklemini`` (all six
  test translations): prints "Eşitlik (6.2.1) denklemini".
- ✗ ``:numref:`fig_x`'de``, ✗ ``:numref:`fig_x`’deki``, ✗ ``§{sec_blocks}'te``.
  ✓ ``§{sec_blocks} içindeki``.
- Do not write *Şekil*, *Tablo*, *Bölüm* or *Eşitlik* before a reference
  either.
- A reference written out by hand in the source ("Chapter 6" of another book)
  puts the word first, as the build does: Bölüm 6, Şekil 3 (✗ 6. Bölüm).
- ``:citet:`Vaswani.2017` `` prints the authors and the year. Make it the
  subject with a singular verb, which fits one author or several:
  ":citet:`Vaswani.2017` introduced …" → :citet:`Vaswani.2017` … önerdi. For a
  possessive use *tarafından*: "the model of :citet:`…`" → :citet:`…`
  tarafından önerilen model. No suffix on a citation.
- ``:cite:`…` `` stands at the end of the clause it supports, before the comma
  or full stop, as in English.

This is also the convention of the earlier Turkish edition (*… içinde*).

### 2.2 Order of protected tokens

The check compares the order of the protected tokens of one class with the
English, and Turkish is verb-final. In the test every translator had reorder
warnings.

- Keep the English order where a natural Turkish sentence allows it: front
  the topic, or split the sentence. "Thus a $3 \times 3$ input with a
  $2 \times 2$ kernel yields $2 \times 2$" → Buna göre $3 \times 3$ boyutlu
  bir girdi $2 \times 2$ boyutlu bir çekirdekle $2 \times 2$ … boyutlu bir
  çıktı verir. (Fable; the order is the English one.)
- Where Turkish needs another order, write correct Turkish and name the block
  in NOTES. *X yerine Y* always reverses "Y instead of X": "$27 c^2$ weights
  instead of $49 c^2$" → $49 c^2$ yerine $27 c^2$ ağırlık.
- A paragraph that leads into a display equation ends as a complete sentence
  with a word that points forward and a colon: "… so the output shape is" →
  … dolayısıyla çıktının şekli şöyledir: (Fable, Opus, Sonnet). The paragraph
  after the equation starts with *Burada* ("where $n$ is …" → Burada $n$ …).
- Protected math is copied as it is, even when it contains English: "the
  $i^{\textrm{th}}$ layer" → $i^{\textrm{th}}$ katman. Do not change it, add
  nothing to it (no full stop, no suffix), and do not report it in NOTES.

## 3. Suffixes after names, acronyms, numbers, symbols and code

Turkish puts case, possessive and plural endings on the word. An apostrophe
separates the ending from a proper name, an acronym, a numeral and a letter
(TDK); the ending follows the **pronunciation** of what precedes it: vowel
harmony from its last vowel, a buffer *y/n/s* after a vowel, *t* for *d* after
a voiceless consonant (*-te/-ta/-ten/-tan* after p, ç, t, k, f, h, s, ş). The
stem is never changed (✗ BERD'in).

**Decision: one table for what takes an apostrophe, and a carrier noun
wherever the pronunciation is not obvious.**

| What precedes | Rule | Examples |
|---|---|---|
| A Turkish common noun, also a loan in Turkish spelling | No apostrophe; normal sound changes. | çekirdeğin, gradyanı, tensörler, pikselde, logitler, momentumun, sigmoidin |
| An acronym read letter by letter | Apostrophe; the ending follows the Turkish name of the **last letter** (TDK: TDK'den, TRT'den). Every consonant name ends in *e* (be, ce, de, …, ke, le, me, ne, pe, re, se, te), so these take *-nin, -yi, -ye, -de, -den, -ler*; after A and I back vowels, after O and U *-nun, -yu, -ya, -da, -lar*. | CNN'nin, CNN'de, CNN'ler ; MLP'nin, MLP'ler ; SGD'yi, SGD'nin ; LSTM'ler ; RNN'de ; KL'nin ; GPU'nun, GPU'lar, GPU'da ; CPU'ya ✗ CNN'in, ✗ GPU'ler |
| An acronym or name read as a word | Apostrophe; the ending follows the word as Turkish engineers say it. | ReLU'nun, BERT'in, BERT'te, GAN'lar, LoRA'nın, CUDA'nın, Adam'ın, ResNet'in, ResNet'te, ImageNet'te, PyTorch'ta, PyTorch'un, Python'da, TensorFlow'da, Keras'ta, JAX'te, Gauss'un, Markov'un, Bayes'in |
| A name with a number | The ending follows the number. | GPT-2'nin, ResNet-18'in, CIFAR-10'da, RTX 4090'da |
| A numeral | Apostrophe; the ending follows the number read in Turkish (the last word read): 1 bir, 2 iki, 3 üç, 4 dört, 5 beş, 6 altı, 7 yedi, 8 sekiz, 9 dokuz, 10 on, 20 yirmi, 30 otuz, 40 kırk, 50 elli, 60 altmış, 70 yetmiş, 80 seksen, 90 doksan, 100 yüz, 1000 bin. A decimal is read digit by digit after *nokta*. | 2'nin, 3'ün, 4'ten, 6'ya, 9'un, 10'un, 100'den, 0.5'in, 0.01'in, 2017'de, 2010'larda, %45'i |
| A word left in English spelling that is not a name (§4.1: softmax, softplus, bootstrap, dropout in an API name) | Do not inflect it; give it a Turkish noun that takes the ending. | softmax fonksiyonunun, softmax işlemi, softplus fonksiyonu ✗ softmaxın, ✗ softmax'ın |
| Inline math: one plain or bold Latin letter other than q and w | A carrier noun is preferred. An apostrophe is allowed where the noun was just named; the ending follows the Turkish letter name (a, be, ce, de, e, fe, ge, he, ı, i, je, ke, le, me, ne, o, pe, re, se, te, u, ve, ye, ze; x is *iks*). | ✓ $\mathbf{K}$ çekirdeğiyle ; ✓ $\mathbf{v}$ vektörüne dik (Fable) ; ✓ $\mathbf{v}$'ye dik (Sonnet) ; $n$'nin, $k$'yi, $x$'in, $x$'e, $b$'den |
| Any other math: Greek letters, q, w, indices, hats, expressions, numbers in math | Never an apostrophe. A carrier noun (*değeri*, *vektörü*, *matrisi*, *katsayısı*, *ifadesi*, *boyutlu*), or a postposition that needs no ending (*ile*, *için*, *ve*, *kadar*, *gibi*, *arasında*). | ✓ $-\Delta$ ile $\Delta$ arasında (Fable) ✗ $-\Delta$'dan $\Delta$'ya (Sonnet) ; ✓ $\mathbf{w}$ ağırlık vektörünü ; ✓ $\eta$ öğrenme oranının ; ✓ $x_i$ değerini ✗ $\eta$'nın, ✗ $x_i$'yi |
| Inline code | Never an apostrophe. A carrier noun: *fonksiyonu*, *sınıfı*, *yöntemi*, *tensörü*, *değişkeni*, *argümanı*, *nesnesi*. | ✓ `corr2d` fonksiyonunu ; ✓ `X` tensörünü oluşturuyoruz (Fable) ✗ `X`'i (Sol) |
| A reference or citation | Never (§2.1). | |
| A word in quotation marks | Ending directly after the closing quote, no apostrophe (TDK); better a carrier noun. | "evrişim" adını |
| An abbreviation that ends in a full stop | Ending directly, no apostrophe (TDK: vb.leri); avoid. | |

- Where you are not sure how a name is said (AdamW, NumPy, BLEU, Newton,
  OpenAI), use a carrier noun and no ending on the name: AdamW optimize
  edicisinin, NumPy kütüphanesinde, BLEU puanı, Newton yöntemi.
- Symbol, code or name + Turkish noun is an ordinary noun compound: the noun
  takes the third-person ending, there is no hyphen and no apostrophe:
  $\ell_2$ normu, `nn.Linear` katmanı, ReLU aktivasyonu, softmax fonksiyonu,
  Adam optimize edicisi, GPU belleği, KV önbelleği, Transformer bloğu.
  ✗ ReLU-aktivasyon, ✗ GPU bellek.
- A measure after math takes no ending and no hyphen: $d$ boyutlu, $k$ kat,
  $L$ katmanlı, $n$ tane; "a width-$k$ window" → genişliği $k$ olan bir
  pencere; "a $3 \times 3$ kernel" → $3 \times 3$ boyutlu bir çekirdek
  (✗ $3 \times 3$'lük).
- An ordinal on a symbol is a full stop: "layer $i$", "the $i$-th layer" →
  $i$. katman (Opus) (✗ $i$'inci, ✗ $i$'nci). Not at a line start, and not
  after `$i^{\textrm{th}}$` (§2.2).
- A closing `$` must not be followed by a digit (the build rejects it).

**Dotted and dotless i.** Turkish has two pairs: i/İ and ı/I.

- Turkish words are capitalised the Turkish way: için → İçin, ilkleme →
  İlkleme, iz → İz, ısınma → Isınma, ışın → Işın; and lower-cased the same
  way (İlkleme → ilkleme, Isınma → ısınma).
- Latin names, acronyms and code keep their letters exactly: ImageNet,
  Inception, InfoNCE, IoU, `init`, i.i.d. ✗ İmageNet, ✗ İnception,
  ✗ ımagenet. Never change the case of a name or an acronym to fit a heading.
- A capital I in a name is still the name's I in an ending: ImageNet'in.
- `l10n check` lower-cases with Python rules and so misses a capitalised
  glossary term that begins with İ or I (İlkleme, İz, İzdüşüm, İnce Ayar,
  Isınma, Işın Arama; §11). If it warns that such a term is missing, the
  translation is right: name the false match in NOTES and change nothing.

## 4. Terminology: Turkish term or English loan

### 4.1 How to choose

**Use the Turkish term that Turkish mathematics, statistics and
computer-science teaching uses. Use a loan in Turkish spelling where that is
the taught term. Leave a word in English spelling only for names. Never coin
a Turkish word, and never leave an English common noun in the sentence.** The
glossary decides each case; it was reviewed on 2026-11-26
(`glossary/review/tr.md`) against the earlier Turkish edition's
`TERMINOLOGY.md`, which follows Alpaydın's *Yapay Öğrenme*. For a term without
an entry, apply the rule and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established Turkish term | Use it. | katman, ağırlık, evrişim, çekirdek, öznitelik, etiket, kayıp, türev, olasılık, beklenti, dağılım, öğrenme oranı, aşırı öğrenme, düzenlileştirme, genelleme, gömme, dikkat, yığın, dönem, ortaklama, dolgu, adım aralığı, geri yayılım, ince ayar, ön eğitim, pekiştirmeli öğrenme, etmen, ödül, durum, eylem |
| Taught term is a loan in Turkish spelling | Use it; do not replace it with a purist coinage. | gradyan (✗ eğim), matris (✗ dizey), vektör (✗ yöney), tensör, fonksiyon (✗ işlev), operatör (✗ işleç), parametre, optimizasyon (✗ eniyileme), varyans (✗ değişinti), regresyon (✗ bağlanım), entropi, aktivasyon, model, norm, piksel, maske, logit, momentum, sigmoid, stokastik, politika, epizot |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English; never translate a name in part. | Transformer, ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, GELU, SGD, MLP, CNN, RNN, LSTM, GPU, RMSNorm, Mamba, word2vec |
| Function names the glossary keeps in English spelling | Lower case, never inflected (§3). | softmax, softplus, log-sum-exp, one-hot (one-hot kodlama), all-reduce |
| Name or acronym + Turkish noun | Noun compound with the third-person ending (§3). | Adam optimize edicisi, ReLU aktivasyonu, Gauss dağılımı, KV önbelleği, softmax regresyonu |

One word family per head word, as the glossary now has it. These pairs were
mixed in the earlier edition; the second form is not used:

girdi, çıktı (✗ giriş, çıkış) · görüntü (✗ imge, resim) · sözcük (✗ kelime) ·
doğrulama (✗ geçerleme) · veri kümesi (✗ veri seti) · sinir ağı (✗ nöral ağ) ·
eğitim (✗ öğretim, antrenman) · yığın, mini yığın (✗ grup, toplu, minigrup) ·
ilkleme (✗ başlatma, ilklendirme) · normalleştirme (✗ normalizasyon) · kuramsal
(✗ teorik; but teorem, bilgi teorisi as in the glossary) · gizli = hidden,
gizil = latent · stokastik = stochastic, rastgele = random · örnek = example or
generated sample, örneklem = statistical sample.

General words the test page settled (all six translators differed; these are
now fixed):

| English | Turkish | Not |
|---|---|---|
| implement, implementation | gerçekleştirmek, gerçekleştirim (Opus, Sonnet); "from scratch" → sıfırdan | kodlamak (it is *encode*: kodlayıcı, konumsal kodlama); uygulamak (it is *apply*) |
| framework (PyTorch, JAX) | çerçeve, derin öğrenme çerçevesi | çatı, yazılım çatısı, kütüphane (it is *library*) |
| dense layer; fully connected layer | yoğun katman; tam bağlı katman | merging the two |
| to stack layers, a stack of layers | katmanları üst üste dizmek, üst üste dizilmiş katmanlar | yığın (it is *batch*) |
| a nonlinearity (a layer in a network) | doğrusal olmayan dönüşüm; the property is *doğrusal olmama* (glossary) | doğrusal olmamalar |
| window, patch, entry or element, index | pencere, yama, eleman, indis | parça; girdi for *entry* (it is *input*) |
| edge (image feature); border, boundary | kenar; sınır | kenar for both |
| spatial; resolution; offset; shift | uzamsal; çözünürlük; kayma; öteleme | |
| built-in; notebook; library | yerleşik; not defteri; kütüphane | |
| 2D, 3D, 4D | iki boyutlu … in prose; 2B, 3B, 4B in headings and slides | 2D in Turkish text (API names such as `Conv2d` stay) |
| machine learning; deep learning; artificial intelligence | makine öğrenmesi; derin öğrenme; yapay zekâ | makine öğrenimi, yapay öğrenme |

### 4.2 Noun compounds and their endings

Most Turkish terms are indefinite noun compounds: the second noun carries the
third-person ending (*kayıp fonksiyonu*, *evrişim katmanı*, *öğrenme oranı*,
*Gauss dağılımı*). The glossary gives the dictionary form; inflect it as
grammar requires.

| Case | Rule | Example |
|---|---|---|
| Plural | Before the ending. | kayıp fonksiyonları, evrişim katmanları ✗ kayıp fonksiyonular |
| A case ending | After the compound ending, with *n*. | kayıp fonksiyonunu, evrişim katmanında, öğrenme oranına |
| A possessor is added | One ending only. | modelin kayıp fonksiyonu ✗ modelin kayıp fonksiyonusu |
| The compound is the first part of a longer one | The inner ending stays when the inner compound is itself the term. | öğrenme oranı çizelgesi, sinir ağı katmanı, yığın boyutu seçimi ; but dikkat ağırlığı, evrişim çekirdeği |
| Adjective + noun | No compound ending; the adjective is not inflected. | evrişimli sinir ağı, gizli katman, yoğun katman, artık bağlantı, stokastik gradyan inişi |
| Prefix-like first parts | As in the glossary: closed *öz-* and *hiper-* (özdikkat, özdeğer, özvektör, hiperparametre), open *alt*, *ön*, *çok* (alt uzay, alt örnekleme, ön eğitim, çok katmanlı). | |

`l10n check` accepts inflected forms (it lets the last two letters of each
word change and any ending follow: *çekirdek* in *çekirdeği*, *şekil* in
*şekli*).

### 4.3 Verbs

A Turkish verb for every action; no English stem with *etmek* or *-lemek*
unless the table has it.

| English | Turkish | Not |
|---|---|---|
| train; pretrain; fine-tune | eğitmek; önceden eğitmek (önceden eğitilmiş model; noun *ön eğitim*); ince ayar yapmak (ince ayarlı model) | train etmek, fine-tune etmek |
| sample (from a distribution) | örneklemek, … dağılımından çekmek | sample almak |
| overfit, underfit | aşırı öğrenmek, eksik öğrenmek; model veriyi aşırı öğrenir | overfit olmak, ezberlemek as the term |
| initialize; update; normalize; regularize | ilklemek; güncellemek; normalleştirmek; düzenlileştirmek | initialize etmek, ilklendirmek |
| optimize, minimize, maximize | optimize etmek; en aza indirmek; en çoğa çıkarmak | minimize etmek in running text (glossary noun: *minimize edici*) |
| pad, pool, clip, mask | dolgu eklemek; ortaklama uygulamak; kırpmak; maskelemek | padlemek |
| tokenize, quantize, encode, decode | belirteçlemek; nicemlemek; kodlamak; kodunu çözmek | |
| flatten, reshape, transpose, concatenate, broadcast | düzleştirmek; yeniden şekillendirmek; devriğini almak; bitiştirmek; yayınlamak | |
| compute, evaluate, derive, verify, test | hesaplamak; değerlendirmek; türetmek; doğrulamak; sınamak | test etmek for "we test our detector" |
| converge to | *-e* yakınsamak | *-e* doğru yakınsamak |
| cache, shuffle, log, deploy, benchmark | önbelleğe almak; karıştırmak; günlüğe yazmak; devreye almak; kıyaslamak | cache'lemek, log'lamak |
| plot, debug | çizmek; hata ayıklamak | plot etmek |

No light-verb padding: ✗ hesaplama gerçekleştirmek, eğitim yapmak, güncelleme
işlemini uygulamak ✓ hesaplamak, eğitmek, güncellemek. "X has Y" is not *sahip
olmak*: ✗ Katman 10 birime sahiptir. ✓ Katmanın 10 birimi vardır.

### 4.4 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(İng. …)`, lower case as in the
glossary, after the inflected word and outside the emphasis markers. Check the
English page in CONTEXT: if an earlier unit of the page introduced the term,
do not repeat it.

- Gloss Turkish terms whose English a reader cannot guess, which in Turkish
  is most native terms; readers know the English words from work and from
  papers: *ek girdi* (İng. bias), *seyreltme* (İng. dropout), *belirteç* (İng.
  token), *ortaklama* (İng. pooling), *dolgu* (İng. padding), *adım aralığı*
  (İng. stride), *yığın* (İng. batch), *dönem* (İng. epoch), *öznitelik*
  (İng. feature), *gömme* (İng. embedding), *dikkat* (İng. attention),
  *kerte* (İng. rank), *ilkleme* (İng. initialization), *nicemleme* (İng.
  quantization), *istem* (İng. prompt), *etmen* (İng. agent).
- Do not gloss loans a reader recognises (gradyan, matris, tensör,
  optimizasyon, regresyon, entropi, aktivasyon, momentum), names, or school
  vocabulary (türev, olasılık, ortalama, vektör).
- With an acronym: *stokastik gradyan inişi* (İng. stochastic gradient
  descent, SGD), *evrişimli sinir ağı* (İng. convolutional neural network,
  CNN); afterwards write what the English writes (SGD, CNN). Do not expand an
  acronym the English leaves short (✗ "SGD recovers …" → Stokastik gradyan
  inişi …, Opus).
- Never in headings, captions, slides, exercises or table cells.

"… a border of zeros (*padding*)" → … buna *dolgu* (İng. padding) denir.
(Fable) Reason: Opus, Astra and Sol added no English at all in the test and Fable
added it as `(İng. …)`; a reader who meets *ek girdi* or *seyreltme* for
the first time needs the bridge once, and no more than once.

## 5. Numbers and units

**Decision: the decimal point everywhere, a space as thousands separator, the
percent sign before the number.** Numerals are copied from the English (0.5,
0.01), in prose as in math, code, code output and figures. Reason: inline and
display math are protected and keep the point, and outputs and figure axes
show it; a decimal comma in prose would put two conventions into one sentence
(✗ $\eta = 0.1$ ile %92,3 doğruluk) and would need a rule for every
identifier (GPT-3.5, Python 3.12). All six test translators kept the point.
This departs from TDK (comma for decimals, point for thousands). With the
point as decimal sign, a point or a comma must never group thousands: `10.000`
would read as ten, and the English `10,000` reads as ten to a Turkish reader.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "a dropout rate of 0.5" → 0.5 seyreltme oranı |
| Thousands | A space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 görüntü ; 2048 ; 60 000 |
| Number words | million → milyon, **billion → milyar**, trillion → trilyon. | "3 billion parameters" → 3 milyar parametre ✗ 3 bilyon |
| Digits or words | As the English: a number spelled out in English is spelled out in Turkish, a digit stays a digit (the check compares digits: Astra got a warning for *dört sütun* where the English had 4). Spelled-out numbers are separate words: on iki, yüz yirmi sekiz. | "two hidden layers" → iki gizli katman ; "10 epochs" → 10 dönem |
| Percent | The sign first, no space (TDK): `92.3%` → `%92.3`. In words *yüzde*. A range has the sign once. Terra kept `45%`; that is wrong in Turkish. | "45% fewer parameters" → %45 daha az parametre ; "45–50%" → %45–50 |
| Units | Symbols as in the source, with a space; time words written out become saniye, dakika, saat, gün. The noun after a number is singular. | 24 GiB, 5 ms ; "3 hours" → 3 saat |
| Ranges | En dash without spaces, as in the source; with words, *ile … arasında* or *-den -e kadar*. | "40–55" → 40–55 ; "from 40 to 55" → 40 ile 55 arasında |
| Multipliers | In prose *N kat*; keep `×` in slides, tables, captions and math. | "about 2× faster" → yaklaşık 2 kat hızlı ; slide: 2× hızlı |
| Approximation | yaklaşık, kabaca; *yak.* only in tables and slides; "an order of magnitude" → bir büyüklük mertebesi. | "roughly 90%" → yaklaşık %90 |
| Ordinals | Written out up to *onuncu*; otherwise digit + full stop (TDK allows 15. and 15'inci; this guide uses the full stop). Never at a line start (§2). | "the third epoch" → üçüncü dönem ; 12. katman |
| Suffixes on numerals | §3. | 0.01'in altına ; 2017'de |
| Decades, years | "the 2010s" → 2010'lar, 2010'lu yıllar; "in 2017" → 2017'de, 2017 yılında. | ResNet 2015'te yayımlandı. |
| Dimensions | `×` as in the source; in prose with *boyutlu*. | "a 6×8 image" → 6×8 boyutlu bir görüntü |
| Money | Amount, space, currency name, in the source's currency. | "\$2.50" → 2.50 ABD doları |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies. `l10n check` compares the digits of each number: `%92.3`
for `92.3%` passes; `92,3` and `10.000` produce a warning, and so does
`10 000` until the setting in §11 is made.

## 6. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2. Endings with an
  apostrophe (§3).
- **People:** as in the English source, without title: Gauss, Bayes, Markov,
  Kolmogorov, Chebyshev, Lagrange, Itô. Exception: Öklid (Öklid uzaklığı,
  Öklid normu), the form of Turkish school mathematics and of the glossary.
- **Terms named after people:** name + Turkish noun with the compound ending,
  the name unchanged and capitalised, no apostrophe and no hyphen: Gauss
  dağılımı, Bayes teoremi, Markov zinciri, Lagrange çarpanı, Taylor açılımı,
  Newton yöntemi, Jensen eşitsizliği, Lipschitz sürekliliği, Cholesky
  ayrışımı, Fourier dönüşümü, Monte Carlo tahmini. ✗ Bayes'in teoremi,
  ✗ Gauss'un dağılımı, ✗ gauss dağılımı. Two names keep the hyphen of the
  English: Euler-Maruyama, Kullback-Leibler ıraksaması.
- **Adjectives from names:** the name as first part, not an adjective ending:
  Gauss gürültüsü (✗ Gaussçu, ✗ Gaussian gürültü). Only *Bayesçi* exists as an
  adjective (Bayesçi çıkarım, glossary).
- **The book:** *Dive into Deep Learning* stays in English; "this book" → bu
  kitap.
- **Languages and places:** Turkish (İngilizce, ABD); programming languages
  stay (Python).
- **Structure words:** part → kısım; chapter → bölüm; section → bölüm (the
  build prints *Bölüm* for both; *alt bölüm* where one sentence contrasts the
  two); appendix → ek; Summary → Özet; Exercises → Alıştırmalar; Discussions
  → Tartışmalar; Theorem → Teorem; Proposition → Önerme; Lemma → Lemma;
  Corollary → Sonuç; Definition → Tanım; Proof → İspat; Remark → Not;
  Example → Örnek. "In this section" → Bu bölümde.

## 7. Cross-framework and tab notes

A `tab_note` is shown for one framework. Describe this framework only; never
compare frameworks. API names stay in code and take a carrier noun (§3).
"PyTorch's `nn.LazyLinear(1)` requires only the output dimension." →
PyTorch'un `nn.LazyLinear(1)` katmanı yalnızca çıktı boyutunu ister.

## 8. Headings, captions, slides, exercises, notes

**Title case in three kinds of block, sentence case everywhere else.**
`heading`, `slide_title` and `kicker` capitalise every word except *ve*,
*ile*, *ya*, *veya*, *yahut*, *ki*, *da*, *de* and the question particle
*mı/mi/mu/mü* (TDK's list; *için*, *gibi* and *kadar* are not in it and are
capitalised) (to confirm). Capitals follow §3 (İ/I); names, acronyms, code and
math keep their case; a word after math or code is capitalised like any other.
Captions, bold lead-ins, list labels, table cells and exercise titles are in
sentence case.

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase in title case, no full stop, no English gloss. English "-ing + object" becomes a noun where one exists (eğitim, seçim), otherwise the *-mA* noun with its object; not the *-mAk* infinitive and no *Bir*. A direct question takes `?`, also where the English has none. | "Convolutions for Images" → Görüntüler İçin Evrişimler ; "Training the Model" → Modelin Eğitimi ; "Learning a Kernel" → Çekirdek Öğrenme (Sol) ✗ Bir Çekirdeği Öğrenmek (Opus) ; "Edge Detection" → Kenar Algılama ; "Why Depth Helps" → Derinlik Neden İşe Yarar? |
| `caption` | Body register without *biz*/*siz*, sentence case; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Two-dimensional cross-correlation. The shaded portions are …" → İki boyutlu çapraz korelasyon. Gölgeli kısımlar, … elemanlarıdır: … (Fable) |
| `slide_title`, `kicker` | As a heading, as short as the English. | `title="Theoretical vs. Effective Receptive Fields"` → `title="Kuramsal ve Etkin Alıcı Alanlar"` ; `[Detecting Vertical Edges with K = [1, -1]]{.kicker}` → `[K = [1, -1] ile Dikey Kenarları Algılama]{.kicker}` (Sonnet, Terra) |
| `slide_text` | Short statements with the aorist, or noun phrases; no *-(I)yoruz* narration, no *siz* except in a task; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and their colon, in sentence case. | "**45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels." → **%45 daha az parametre:** $c$ kanalda $49 c^2$ yerine $3 \times 9 c^2 = 27 c^2$ ağırlık. (Fable) ✗ **%45 daha az parametre.** … kullanılır. Tek katmanda ise … vardır. (Sol: two added sentences, the colon lost) |
| `exercise` | Second person plural imperative *-(y)In*; direct questions. Reason: it is the imperative of the body register; *-(y)InIz* is exam language, the bare *-mAk* infinitive reads as a note, *sen* breaks the register. The bold title is in sentence case and keeps its full stop. Keep `[code]`, `●`, the numbering and the indentation. | "Derive a finite difference operator for the second derivative. Which structures in images respond most strongly to it?" → İkinci türev için bir sonlu fark operatörü türetin. Görüntülerdeki hangi yapılar buna en güçlü tepkiyi verir? (Fable) ✗ türetiniz, ✗ türet, ✗ türetmek |
| Source credits in exercises | "Adapted from X" → X … uyarlanmıştır, at the end. Course and book titles and notebook names stay in English; the visible text of a link is translated when it is a common noun. | *Stanford CS231n, [Ödev 2](…), ConvolutionalNetworks not defterinden uyarlanmıştır.* (Fable, Opus) |
| `tab_note` | §7. The forum link text "Discussions" → Tartışmalar. | `[Tartışmalar](https://d2l.discourse.group/t/65)` |
| `note` | Register of its surroundings. Labels: **Note:** → **Not:**, **Warning:** → **Uyarı:**, **Tip:** → **İpucu:**, **Answers:** → **Yanıtlar:** | |
| `table_row` | Noun phrases in sentence case; numbers and units as in §5. | "Training time" → Eğitim süresi |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by
the build; copy it unchanged. Slides name topics, not section numbers; keep a
`§{…}` token where the English has one (§2.1).

## 9. Known pitfalls

### 9.1 Words with more than one sense

| English | Sense → Turkish | Not |
|---|---|---|
| bias | parameter $b$ → ek girdi; of an estimator → yanlılık, *unbiased* → yansız; inductive bias → tümevarımsal yanlılık; social → ön yargı | sapma; yanlılık for $b$; meyil |
| kernel | convolution, kernel methods, covariance functions → çekirdek (çekirdek fonksiyonu); of a linear map → sıfır uzayı | kernel |
| return | RL → getiri; of a function → döndürmek, dönüş değeri | getiri for "the function returns" |
| score | $\nabla \log p$ → skor (skor fonksiyonu, skor eşleştirme); a metric or attention score → puan (F1 puanı, dikkat puanı) | skor for a metric |
| sample | a data point or a generated output → örnek; statistics → örneklem (örneklem ortalaması); to sample → örneklemek | numune |
| feature | → öznitelik (öznitelik haritası, öznitelik vektörü); of a library or product → özellik | özellik for an input feature |
| policy, state, action, reward, value, environment, agent | politika, durum, eylem, ödül, değer, ortam, etmen | siyaset, hâl, aksiyon, ajan, çevre |
| episode | → epizot | bölüm (it is the book's word for chapter and section) |
| loss, cost, objective, error, risk | Five words, five terms: kayıp, maliyet, amaç fonksiyonu, hata, risk. | yitim; merging them |
| accuracy, precision, recall | doğruluk, kesinlik, duyarlılık; floating-point *precision* → hassasiyet (karma hassasiyet, yarı hassasiyet) | kesinlik for float formats |
| generalization | ML → genelleme; *generalized* linear model → genelleştirilmiş doğrusal model | genelleştirme for the ML sense |
| normalization, regularization, standardization | normalleştirme, düzenlileştirme, standartlaştırma; normal distribution → normal dağılım | mixing them |
| training, learning | train → eğitmek, eğitim; learn → öğrenmek (öğrenme oranı, gözetimli öğrenme) | öğretim |
| convolution, convolutional | the operation → evrişim; a layer, block, kernel → evrişim katmanı, evrişim bloğu, evrişim çekirdeği; the network → evrişimli sinir ağı | konvolüsyon; evrişimsel |
| step size, stride | adım büyüklüğü (optimization); adım aralığı (convolution) | adım for both |
| batch, stack | yığın; stacking layers → üst üste dizmek | yığın for a stack |
| sequence, array, series | dizi (ML data, and in mathematics); array in code → dizi; where both meet, *tensör* or *liste* for the container | seri for a sequence |
| hidden, latent | gizli (gizli katman, gizli durum); gizil (gizil değişken, gizil uzay) | gizli for latent |
| shape, form | of a tensor → şekil; of a curve or expression → biçim ("bell shape" → çan biçimi) | şekil for both; near a figure reference write *biçim* |
| figure | a picture → şekil (printed by the build); a number → sayı, rakam | figür |
| translation | of an image or function → öteleme (öteleme değişmezliği, ötelemeye eşdeğişim); of language → çeviri | çeviri değişmezliği (the earlier edition's error) |
| inference | running a trained model, and statistics → çıkarım; "the layer infers its input size" → kendiliğinden belirler | sonuç çıkarma |
| expectation, mean, average | beklenti; ortalama; "on average" → ortalamada; "in expectation" → beklentide | beklenen for the noun |
| likelihood, probability | olabilirlik; olasılık | ihtimal for either |
| rank, trace, norm | kerte, iz, norm; ranking → sıralama | rütbe, rank |
| order | second-order method → ikinci mertebeden yöntem; sequence order → sıra; order of magnitude → büyüklük mertebesi | düzen |
| power | $x^2$ → kuvvet; statistics → güç; electrical → güç | |
| memory | hardware → bellek; LSTM, cognition → bellek (uzun kısa süreli bellek) | hafıza |
| domain | of a function → tanım kümesi; of data → alan | |
| performance | of a model → başarım; of code → hız, hesaplama başarımı | performans every time |
| label | → etiket; of an axis → eksen etiketi | |
| number | a count → sayı (parametre sayısı); an identifier → numara | parametre numarası |
| continuous, smooth, bounded | sürekli; düzgün; sınırlı | pürüzsüz |
| test | test set, test error → test kümesi, test hatası; "we test the detector" → sınamak | |
| key, query, value (attention) | anahtar, sorgu, değer | |
| head | attention head → dikkat başı | kafa |
| pattern | → örüntü; a visual texture → desen | kalıp |

### 9.2 False friends and words that mislead

| English | Turkish | Not (because) |
|---|---|---|
| billion | milyar | bilyon |
| realize | fark etmek, görmek | gerçekleştirmek (= implement) |
| eventually | sonunda, zamanla | muhtemelen (= possibly) |
| actually | aslında, gerçekte | aktüel (= current) |
| argue | savunmak, ileri sürmek | tartışmak (= discuss) |
| argument (of a function) | argüman | tartışma |
| significant | belirgin, önemli; *anlamlı* only in the statistical sense | anlamlı for "large" |
| dramatically | belirgin biçimde, büyük ölçüde | dramatik biçimde |
| technically | kesin konuşmak gerekirse | teknik olarak (every time) |
| trivial | apaçık, kolay, önemsiz (by sense) | trivial, basit (every time) |
| arbitrary | keyfi, herhangi bir | rastgele (= random) |
| decade | on yıl | dekad |
| silicon | silisyum | silikon (= silicone) |
| control | denetlemek; *kontrol etmek* means "to check" | |
| consistent | tutarlı; *consistent estimator* → tutarlı tahmin edici; "consistent with" → ile uyumlu | |
| sensitive, sensitivity | duyarlı, duyarlılık; where *duyarlılık* as the metric (recall) is discussed nearby, use a verb: … girdiden ne kadar etkilendiği | hassasiyet (it is float precision) |
| "makes sense" | akla yatkın, anlamlı | mantık yapar |
| "sanity check" | hızlı doğrulama, tutarlılık denetimi | akıl sağlığı kontrolü |
| "rule of thumb" | pratik kural | başparmak kuralı |
| "defeats the purpose" | amaca ters düşer | amacı yener (the earlier edition) |

### 9.3 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "Note that", "It is important to note that" word for word | Drop it, or *Burada …*, *Dikkat edilirse …*; *… dikkat edin* at most once per page | ✗ Not edin ki ek girdi düzenlileştirilmez. ✗ Şunu belirtmek önemlidir ki … ✓ Ek girdi burada düzenlileştirilmez. |
| "Recall that" | *Hatırlanacağı gibi*, or the past tense with *-mIştIk* | ✗ Hatırlayın ki $b$ ek girdidir. ✓ $b$ ek girdiydi. |
| Passive + *tarafından* | Active, the agent as subject | §1 |
| "X allows us to", "X lets us" | *X sayesinde … -(y)Abiliriz*, *X ile …* | ✗ Bu, bize gradyanı hesaplama imkânı verir. ✓ Böylece gradyanı hesaplayabiliriz. |
| Inanimate subject + "tells us", "shows us" | *Buradan … anlaşılır*, *Bu sonuca göre …*; *göstermek* is fine with a figure or table as subject | ✗ Bu sonuç bize derinliğin yardımcı olduğunu söyler. ✓ Bu sonuca göre derinlik işe yarar. |
| "we can" every time as *-(y)Abiliriz* | The verb itself | "we can compute" → hesaplarız |
| "there is / there are" | *vardır* once; otherwise a full verb | "There are three reasons." → Bunun üç nedeni var. |
| "for 10 epochs" | *10 dönem boyunca* | ✗ 10 dönem için eğitmek |
| "in terms of" | *cinsinden* (a quantity), *açısından* (a viewpoint) | "in terms of $\partial \ell/\partial \mathbf{Y}$" → $\partial \ell/\partial \mathbf{Y}$ cinsinden |
| "based on" | *-e dayalı*, *-e dayanan*, *-e göre* | ✗ dikkat temelli (every time) ✓ dikkate dayalı bir model |
| "with respect to" | *-e göre*; a derivative: $x$ değişkenine göre türev | ✗ $x$ ile ilgili türev |
| "respectively" | *sırasıyla*, before the list | |
| "one of the most widely used" | *en yaygın … -lerden biri* | Adam en yaygın optimize edicilerden biridir. |
| "which", "where" clauses carried over as *ki* or *ki bu* | A new sentence with *Bu*, *Burada* | ✗ … bir matris elde ederiz ki bu seyrektir. ✓ … bir matris elde ederiz. Bu matris seyrektir. |
| *o*, *onlar*, *bunlar* for models and data | Repeat the noun, or leave the subject out | "they overfit" (models) → bu modeller aşırı öğrenir |
| English word order: the new information first | The known first, the new directly before the verb | ✗ Gradyanı hesaplarız zincir kuralıyla. ✓ Gradyanı zincir kuralıyla hesaplarız. |
| Idioms translated by image | The meaning | "under the hood" → iç işleyişinde ; "from scratch" → sıfırdan ; "out of the box" → hazır hâliyle ; "workhorse" → temel araç ; "to get our feet wet" → ısınmak için |
| Title Case in a bold lead-in or a caption; English plural or possessive | Sentence case; Turkish forms | ✗ **Daha Zengin İfade Gücü:** ✗ GPU's, ✗ Adam's |
| Content added to explain | Nothing added (TRANSLATING.md §3) | ✗ "Each layer widens the receptive field by 2" → … iki girdi konumu kadar genişletir ve sonunda … (Astra, Sol) ✓ Her katman alıcı alanı 2 genişletir ve 7 girdiye çıkarır. (Fable) |
| The source corrected in the translation | Translate what is written; report it in NOTES | ✗ "SGD recovers …" → Gradyan inişi … (Sol changed the claim) |

Logical and mathematical phrases are fixed: because → çünkü, *-DIğI için*;
therefore → dolayısıyla, bu nedenle; however → ancak, bununla birlikte; if →
*-sA*, eğer … ise; only if → yalnızca … ise; if and only if → ancak ve ancak;
at most → en fazla; at least → en az; "let $x$ be …" → $x$ … olsun; such that
→ öyle ki, … olacak biçimde; without loss of generality → genelliği bozmadan;
for all → her … için; there exists → öyle bir … vardır ki (the one fixed use
of *ki*); almost surely → neredeyse kesin olarak; i.i.d. → bağımsız ve özdeş
dağılımlı (then i.i.d., as the English); dot product → nokta çarpımı; inner
product → iç çarpım; elementwise → eleman eleman; "is given by" → şöyle
verilir; "denote by" → … ile gösterelim.

## 10. Sources

Checked on 2026-11-26:

- TDK, *Yazım Kuralları* (tdk.gov.tr/icerik/yazim-kurallari/): *Büyük
  harflerin kullanıldığı yerler* (title case; the lower-case words are "ve,
  ile, ya, veya, yahut, ki, da, de" and "mı, mi, mu, mü"; a capital after a
  colon before a sentence); *Sayıların yazılışı* (point for groups of three,
  comma for decimals, 15. or 15'inci, %25 without a space); *Noktalama
  işaretleri* (apostrophe after abbreviations, numerals and letters: TBMM'nin,
  1985'te, a'dan z'ye; none after a quoted word; no comma around *ve, veya,
  yahut* or after a converb; “ ” and ‘ ’); *Kısaltmalar* (endings follow the
  name of the last letter: TDK'den, THY'de; or the word: NATO'dan,
  ASELSAN'da; cm'yi, kg'dan; vb.leri).
- The repository: `glossary/terms/tr.jsonl`, `glossary/review/tr.md` (the
  earlier edition is post-edited machine translation with an editorial
  `TERMINOLOGY.md` after Alpaydın's *Yapay Öğrenme*), `glossary/notes/ar-tr-de.md`,
  the six answer sets in `evals/tr-conv-layer/`, `tools/l10n/protect.py`,
  `tools/l10n/glossary.py`.
- Quarto's `_language-tr.yml` in d2l-neu's `.venv-build` (Şekil, Tablo, Bölüm
  for chapter and section, Eşitlik; Not, Uyarı, İpucu), and the English build
  (equation references print "Equation (6.2.1)", sections "Section 6.1").
- The book's pandoc: `$k$'nin`, `` `X`'in ``, `2'nin`, `GPU'lar`, `"evrişim"i`
  render with ’ and “ ”; a role followed by `'de` is still recognised.
- `l10n check` on Turkish samples: `%92.3` passes; `92,3`, `10.000` and
  (without the setting) `10 000` warn; an apostrophe after a number does not
  hide the number; `İlkleme` is not matched to the glossary's *ilkleme*.

Not checked:

- The earlier edition's pages themselves. That it writes *… içinde* after a
  reference and uses *Özet*, *Alıştırmalar* and *Tartışmalar* is stated from
  the drafter's memory of tr.d2l.ai, not re-read.
- Alpaydın, *Yapay Öğrenme* (Boğaziçi Üniversitesi Yayınevi), the Turkish
  edition of Goodfellow et al., and Turkish university course notes: none was
  read. The statements about what Turkish teaching uses rest on the glossary
  review and the drafter's knowledge.
- TDK on a comma after *ancak, fakat, ama, çünkü*: the page read names only
  *ve, veya, yahut, ya … ya*; the rule in §2 is the one taught in schools.
- How Turkish engineers say individual names (§3): from the drafter's
  knowledge; see §12.

## 11. Settings

`languages.yml`, existing keys (`tools/l10n/config.py`, `protect.py`):

- `check: {thousands: space}`: accepts `10 000` for the English `10,000` and
  warns when a Turkish number keeps the English comma.
- `typography: {nbsp: true}`: the build puts no-break spaces into grouped
  numbers (10 000) and between a number and its unit (24 GiB). Its
  abbreviation rule (letter, full stop, space, letter, full stop) was written
  for German *z. B.*; in Turkish it only matches a rare sequence such as
  "a. b." and does no harm.
- `check: {order: warning}` (the default) stays: Turkish reorders tokens in
  many blocks and a NOTE acknowledges each one (§2.2).
- `dir: ltr`, `forum: english` as now.

`tr/strings.yml` (when `l10n strings tr` generates it):

- No `quarto.crossref-*` override is needed: Quarto's Turkish file already
  gives Şekil, Tablo, Bölüm (chapter and section) and Eşitlik.
- `book.exercises_heading: "Alıştırmalar"`, `book.discussions_link:
  "Tartışmalar"`.
- `pdf.lang: "tr"` only after a test: babel's Turkish makes `=`, `:` and `!`
  active characters under some engines, which breaks key=value options in
  LaTeX; if it fails, load it with `shorthands=off`.
- The fonts of the English build cover ğ, ş, ı, İ; no font change.

Needs new code:

- **Turkish case folding in the glossary check.** `glossary.py` uses Python's
  `lower()`, which turns İ into i + U+0307 and I into i; a capitalised term
  that begins with İ or I is then reported as missing (tested: İlkleme, İz,
  İzdüşüm, İnce Ayar, İstem, Isınma). Fix: for `tr`, map İ → i and I → ı
  before lower-casing both sides.
- **Upper-casing in the build.** Any CSS `text-transform: uppercase` or LaTeX
  `\MakeUppercase` on Turkish text needs the language set (`lang="tr"`), or
  *i* becomes *I* instead of *İ*. To be checked on kickers, running heads and
  the slide theme.
- **Optional: case endings on references.** The build knows the printed
  number, so it could add the ending (Şekil 6.2.1'de, Bölüm 6.3'te) from a
  marker the translator writes after a role. This guide does not assume it
  (§2.1); see §12, question 3.
- **Optional: a check for an apostrophe or a letter directly after a
  reference role**, as an error for `tr` (§2.1), like the CJK adjacency check.

## 12. For the author to confirm

1. **Decimal point and space for thousands** (§5), against TDK's comma and
   point. Proposed: as written, with `check: {thousands: space}`.
2. **Percent sign before the number** (`%92.3` for `92.3%`), although
   TRANSLATING.md §4 lists `92.3%` among the tokens to copy. Proposed: yes; the
   check compares only the digits and passes it.
3. **References take no ending** (*… içinde*, the reference as subject, in
   parentheses) (§2.1). Proposed: yes for now. Alternative: have the build
   attach the ending, which gives the more natural Şekil 6.2.1'de.
4. **Printed reference words:** Quarto's *Bölüm* for both chapter and section,
   and *Eşitlik* for equations (many Turkish textbooks print *Denklem*).
   Proposed: keep both.
5. **Title case in headings, slide titles and kickers, with TDK's list of
   lower-case words** (so *İçin* is capitalised: Görüntüler İçin Evrişimler;
   four of the six test translators wrote *için*). Proposed: TDK's list.
   Alternative: sentence case throughout (Astra's choice), which also avoids
   the İ/I false matches.
6. **Apostrophe after math** only for a single Latin letter, a carrier noun
   everywhere else, and never after code (§3). Proposed: as written. A
   stricter alternative is no apostrophe after math at all.
7. **Endings on acronyms by Turkish letter names** (CNN'nin, SGD'yi, GPU'nun),
   the TDK rule, although many engineers say the letters in English (CNN'in).
   Proposed: TDK.
8. **Endings on English names by spoken form** (PyTorch'ta, Python'da,
   JAX'te, Bayes'in, TensorFlow'da), and a carrier noun for AdamW, NumPy,
   BLEU, Newton, OpenAI. A native reviewer should confirm the list in §3.
9. **Function names in English spelling are not inflected** (softmax
   fonksiyonunun, never softmax'ın). Proposed: yes.
10. **The English gloss `(İng. …)` once per page for native Turkish terms**
    (§4.4), a wider set than in the German guide because the glossary's
    Turkish terms (ek girdi, seyreltme, belirteç, kerte, etmen) are far from
    the English ones. Proposed: yes.
11. **Open glossary choices** listed in `glossary/review/tr.md`: belirteç or
    token; ek girdi or bias; seyreltme or dropout; dönem or epok; kerte or
    rank; etmen or ajan; yığın. This guide follows the glossary; a ruling for
    loans would shorten the gloss list in §4.4.
12. **General words fixed in §4.1:** gerçekleştirmek / gerçekleştirim for
    "implement(ation)", çerçeve for "framework", makine öğrenmesi, yapay zekâ
    with the circumflex, kuramsal for "theoretical", başarım for
    "performance", sınamak for "to test".
13. **Exercises in *-(y)In*** (türetin), not *-(y)InIz* (§8). Proposed: yes.
14. **One word, *bölüm*, for chapter and section** in prose, and *epizot* for
    an RL episode (§6, §9.1). Proposed: yes.
15. **Proof → İspat, Corollary → Sonuç, Lemma → Lemma** (§6); *Kanıt* and
    *Yardımcı Teorem* (Quarto's word) are the alternatives.
16. **No circumflex on the adjective ending** (merkezi, not merkezî), as the
    glossary writes; TDK prefers merkezî. Proposed: as the glossary.

## 13. Proposed glossary corrections

The glossary was not edited. Entries that the rules above contradict:

| Concept | Now | Proposed | Rule |
|---|---|---|---|
| `transformer` | transformer | Transformer | A name: capitalised, apostrophe before an ending (§3, §4.1). The compounds already have it (görü Transformer'ı, difüzyon Transformer'ı). |
| `type-i-error`, `type-ii-error` | 1. tür hata, 2. tür hata | birinci tür hata, ikinci tür hata | A number and a full stop at a line start becomes a list item (§2). |
| `jacobian`, `jacobian-determinant` | Jakobiyen, Jakobiyen determinantı | Jacobi matrisi, Jacobi determinantı | People's names unchanged + Turkish noun (§6); also consistent with the next row. (to confirm) |
| `hessian`, `hessian-vector-product`, `hessian-eigenvalue` | Hessian matrisi, Hessian-vektör çarpımı, Hessian özdeğeri | Hesse matrisi, Hesse matrisi-vektör çarpımı, Hesse matrisinin özdeğeri | The name is Hesse; *Hessian* is the English adjective (§6). (to confirm; *Hessian matrisi* is common in Turkish usage) |
| `langevin-dynamics` | Langevin dinamikleri | Langevin dinamiği | English plural carried over (§9.3). |
| `self-supervised-learning` | öz gözetimli öğrenme | özgözetimli öğrenme | *öz-* is closed in özdikkat, özdeğer, özvektör, özilişki (§4.2). (to confirm) |
| `intersection-over-union`, `rotary-position-embedding`, `vision-transformer`, `diffusion-transformer`, `neural-ode`, `normalized-discounted-cumulative-gain`, `quasi-newton` | the term with "(IoU)", "(RoPE)", "(ViT)", "(DiT)", "(Neural ODE)", "(NDCG)", "(quasi-Newton)" inside the entry | the Turkish term alone as `preferred`, the acronym or English name as an alternative | An entry is a lemma; the acronym is given once per page by §4.4, not at every use. |
| `nonlinearity` | doğrusal olmama | add the alternative *doğrusal olmayan dönüşüm* for the countable sense (a layer) | Three test translators needed it (§4.1). |
| `minimizer` | minimize edici | en küçükleyen (to confirm) | §4.3 has *en aza indirmek* for the verb; a native reviewer should choose the noun. |

The check's handling of İ and I (§11) is a tool defect, not a glossary one:
the lower-case entries (ilkleme, iz, izdüşüm, ince ayar, ısınma, ışın arama)
are right as they are.
