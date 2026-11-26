# Turkish (tr) glossary review

Reviewed 2026-11-26. All 1,184 entries were read against the sense in
`concepts.jsonl`: the 209 `verified_d2l`, the 84 `verified_external`, and all
891 `proposed` entries (not a sample). Corrections are in
`tr-corrections.jsonl` (187 lines).

## Verdict

The glossary is usable as a base but not as it stands. Outright wrong terms are
few; the main defect is that one English head word is rendered two or three
ways, and the "verified" entries are the less consistent half.

| Status | Entries | To change | wrong sense | MT / calque | non-standard | inconsistent | minor (form) |
|---|---|---|---|---|---|---|---|
| `verified_d2l` | 209 | 69 (33%) | 4 | 2 | 19 | 28 | 16 |
| `verified_external` | 84 | 19 (23%) | 2 | 4 | 1 | 9 | 3 |
| `proposed` | 891 | 99 (11%) | 10 | 4 | 4 | 57 | 24 |
| Total | 1,184 | 187 (16%) | 16 | 10 | 24 | 94 | 43 |

Among entries with frequency ≥ 50, 46 of 143 verified and 16 of 145 proposed
entries change. The proposed entries are better than the verified ones: they
were written in one mainstream register, while the verified ones mix three
sources that disagree with each other.

## Provenance of the earlier Turkish edition

`d2l-ai/d2l-tr` is machine translation post-edited by one person, with an
editorial terminology list on top.

- `config.ini` has `translator = aws`, as the Japanese edition does.
- Git history shows chapters arriving as machine translation and then being
  revised: "add machine translation for CNN" (Aston Zhang, 2020-11-13),
  followed by "WIP: Revision of Turkish machine translation of CNN chapter"
  (2020-11-18). The NLP and recommender chapters were added in bulk on
  2022-01-25 and later received typo fixes only. One reviser accounts for 427
  of about 515 commits.
- Unedited MT survives in the body, e.g. "Bu amacı yenmek gibi görünüyor"
  for "this seems to defeat the purpose" (`chapter_optimization/gd.md`) and
  the heading "Çeviri Değişmezliği" for translation invariance.
- `TERMINOLOGY.md` is a human list that follows the glossary in Alpaydın's
  *Yapay Öğrenme* (the Deep Learning Türkiye glossary reproduces that source:
  `bias unit: ek girdi`, `variance: değişinti`, `hinge loss: menteşe yitimi`,
  `softmax function: eşiksiz en büyük işlevi`). These are legitimate terms of a
  purist school, not MT errors, but several are rare in current usage.

Consequence: a `d2l-tr-terminology` entry is a deliberate editorial choice; a
`d2l-tr-source` entry is a heading or emphasis pair and can be an MT residue or
a one-off. In 24 sampled chapters the body text often contradicts the entry the
glossary extracted:

| Concept | Glossary (verified) | d2l-tr body usage |
|---|---|---|
| variance | değişinti | varyans 75, değişinti 4 |
| mutual information | Ortak Bilgi | karşılıklı bilgi 26, ortak bilgi 1 |
| optimization | Eniyileme | optimizasyon 43, eniyileme 7 |
| GPU | grafik işleme birimi | GPU 23, spelled out 0 |
| generalization | genelleştirme (Google) | genelleme 30, genelleştirme 2 |
| neural network | nöral ağ (Google) | sinir ağı 64, nöral ağ 0 |
| training set | eğitim seti (Google) | eğitim kümesi 18, eğitim seti 0 |
| underfitting | eksik uyum (Google) | eksik öğrenme 7, eksik uyum 0 |

The Google glossary pages are machine-translated (the notes say so) and
supplied the four worst high-frequency entries above plus `batch size → grup
boyutu` and a wrong-sense `shrinkage`. The Amidi sheets are volunteer
translations of uneven quality ("Kesiştirilmiş Bölgeler" for IoU, "Maksimum
ihtimal tahmini" for maximum likelihood estimator, "Lagranj", and two entries
that still carry list numbers: "35. Koşullu bağımsızlık", "30. [Gibbs
örneklemesi").

## Systematic problems

1. **Wrong senses (16).** `translation-invariance` "Çeviri Değişmezliği"
   (language translation; should be öteleme), `curvature` "eğri" (curve),
   `line-search` "Doğrusal arama" (linear search), `decoupled-weight-decay`
   "Ayrık ağırlık çürümesi" (discrete weight rotting),
   `uncertainty-quantification` "nicemleme" (quantization), `rate-distortion`
   "çarpıklık" (skewness), `nuclear-norm` "Çekirdek normu" (kernel norm),
   `value-backup` "yedekleme" (data backup), `downstream-task` "Alt görev"
   (subtask), `linear-recurrence` "özyineleme" (recursion),
   `computational-graph` "grafiği" (chart), `critic-generative`.
2. **One head word, several renderings.** batch: grup / toplu / yığın;
   normalization: normalleştirme / normalizasyon; initialization: ilkleme /
   ilkletme / başlatma; latent: gizli / gizil; dual: ikilik / ikillik / ikili;
   interpolation: aradeğerleme / enterpolasyon / interpolasyon; rank: kerte /
   rütbe; loss: kayıp / yitim; function: fonksiyon / işlev; input/output:
   girdi-çıktı / giriş-çıkış; word: kelime / sözcük; image: imge / görüntü;
   validation: geçerleme / doğrulama; segmentation: bölünme / bölümleme;
   head: kafa / baş; stochastic: rasgele / stokastik; denoising: arındırma /
   giderme; decoder: kodçözücü / kod çözücü; approximation: yaklaşım /
   yaklaştırma; neural: sinir / nöral / sinirsel.
3. **Collisions between senses.** `gizli` = hidden and latent; `ikili` =
   binary, dual and pairwise; `yönlendirme` = routing and guidance; `eşlenik`
   = conjugate and adjoint; `seyreltme` = dropout and downsampling; `bölüm` =
   episode and book chapter; `başlatma` = initialization and launch; `örnek`
   = example, instance and statistical sample (should be örneklem); `ileri
   geçiş` = forward pass and forward transition; `verim` = throughput and
   efficiency; `grup` = batch and group (grouped convolution). The collisions
   the brief lists are otherwise handled correctly: ek girdi vs. yanlılık,
   getiri (RL return), skor (density) vs. puan (metric), kayıp vs. maliyet,
   politika / değer / durum / eylem / ödül, gradyan, etiket, ortalama,
   beklenti.
4. **Entries that are not lemmas.** 1,021 of 1,184 entries are capitalised as
   headings (all but 3 proposed ones) and 163 are lower-case; lower-case every
   entry except proper names (Adam, Gauss, Bayes, Newton). Seventeen carry
   parenthetical glosses or extraction junk ("Kerte (Rank)", "Bayes Kuramı
   (Teoremi)", "(matrisin) sağlamlık sayısı"); several are inflected
   ("merkez sözcüğün", "küresel vektörlerden", "Matrisi Çarpanlarına Ayırma")
   or miss the compound suffix ("Kullanıcı gömme", "Optimizasyon durum",
   "Üstel Ailesi"). The capitalisation fix is mechanical and is not listed
   per entry in the corrections file.
5. **Names.** Model and method names in Latin script are right (Adam, AdamW,
   BERT, DDPM, RMSNorm, Lasso). Exceptions: `transformer` "dönüştürücü" and
   its four compounds, `gpu` spelled out, `softmax` "softmaks", `skip-gram`
   "Atlamalı sözcük", `reflow`, `n-gram` "n-gramm", `bleu` "Bleu puanı".

## Decisions for the author or a native reviewer

These are choices between defensible options; the corrections file records my
recommendation.

1. **Register.** Alpaydın-school purist terms (değişinti, eniyileme, ilkleme,
   geçerleme, öznitelik, etmen, külliyat, ek girdi) or mainstream loans
   (varyans, optimizasyon, başlatma, doğrulama, özellik, ajan, derlem)? The
   glossary currently mixes both. I recommend mainstream where d2l-tr's own
   body text already uses it, and keeping öznitelik, etmen, ilkleme, evrişim,
   ortaklama, olabilirlik, sonsal/önsel, düzenlileştirme.
2. **token** (frequency 2,117): "belirteç" (kept), "andıç" (d2l-tr's more
   frequent form) or untranslated "token" (dominant in current LLM writing)?
3. **bias of a layer** (959): "ek girdi" is unambiguous but known only from
   Alpaydın and d2l-tr; alternatives are "bias" or "sapma". Kept, with a
   gloss "(bias)" suggested at first use.
4. **batch**: yığın (recommended) or d2l-tr's minigrup / toplu.
5. **Transformer** in Latin script (recommended) or "dönüştürücü".
6. **dropout**: "seyreltme" (kept) or untranslated "dropout"; d2l-tr's "hattan
   düşme" should not be used.
7. **episode**: "epizot" (recommended) or "bölüm", given that "bölüm" is the
   book's word for chapter and section.
8. **discriminator**: "ayırt edici" (recommended) or d2l-tr's "ayrımcı".
9. **global**: "küresel minimum / küresel ortaklama" (kept) or "global".
10. **expectation**: "beklenti" (kept) or the textbook "beklenen değer".
11. **ODE / SDE**: spelled out each time as now ("adi diferansiyel denklem"),
    or an acronym after first use, and then ODE/SDE or ADD/SDD?
12. **downsampling / upsampling**: aşağı / yukarı örnekleme (recommended) or
    d2l-tr's örnek seyreltme / sıklaştırma.
13. **rank**: "kerte" (Wikipedia, d2l-tr; recommended for consistency) or
    "rank".
14. Low-confidence items for a native check: olurlu (feasible), kılavuzlama
    (guidance), yaklaşıklama (approximation), aradeğerleme against
    Wikipedia's enterpolasyon (interpolation), alt akış görevi (downstream
    task), büzülme (shrinkage), özyükleme (bootstrapping), dualite.

## Sources

- d2l-tr at commit `fff29a8`: `config.ini`, `TERMINOLOGY.md`, git history, and
  24 chapter files counted for body usage.
- Deep Learning Türkiye, *Türkçe Yapay Zeka Terimleri* (based on the glossary
  of Alpaydın, *Yapay Öğrenme*):
  https://github.com/deeplearningturkiye/turkce-yapay-zeka-terimleri
- Turkish Wikipedia article titles and redirects: Varyans, Eğrilik, Kerte
  (lineer cebir), Optimizasyon (Eniyileme redirects to it), Transformer (derin
  öğrenme mimarisi), Uzun kısa süreli bellek, Bölüşüm fonksiyonu, Aktivasyon
  fonksiyonu, Derlemdilbilim, Bilgi erişim, Bayes teoremi, Birikimli dağılım
  fonksiyonu, Yapay sinir ağları, Çizge teorisi, Düalite, Sayısal analiz.
- Amidi Turkish sheets (CS 230 CNN and RNN, CS 229 deep learning) on
  stanford.edu/~shervine/l/tr/, read as volunteer translations.

Limits: Turkish Wikipedia is thin on machine learning, so phrase-frequency
checks there settled little; no Turkish university course notes or the
Turkish edition of Goodfellow et al. were consulted. Corrections marked
`proposed` rest on internal consistency and my own judgment and need a native
reviewer; only the 24 marked `verified_external` cite a source that uses the
term in this sense.
