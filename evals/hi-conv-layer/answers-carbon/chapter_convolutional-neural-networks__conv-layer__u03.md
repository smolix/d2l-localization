<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=hi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## अभ्यास

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **हाथ से डिज़ाइन किए गए कर्नेल।** कुछ कर्नेल स्वयं डिज़ाइन करें।
    1. एक दिशात्मक सदिश $\mathbf{v} = (v_1, v_2)$ दिए होने पर, एक ऐसा
       किनारा-पहचान कर्नेल व्युत्पन्न करें जो $\mathbf{v}$ के लांबिक किनारों,
       अर्थात् $(v_2, -v_1)$ दिशा के किनारों का पता लगाए।
    1. द्वितीय अवकलज के लिए एक परिमित अंतर संकारक (ऑपरेटर) व्युत्पन्न करें।
       छवियों में कौन-सी संरचनाएँ इसके प्रति सबसे प्रबल प्रतिक्रिया देती हैं?
    1. $d$ कोटि का अवकलज प्राप्त करने के लिए किसी कर्नेल का न्यूनतम आकार
       क्या होना चाहिए?
    1. आप एक ब्लर (धुँधला करने वाला) कर्नेल कैसे डिज़ाइन करेंगे? आप ऐसे
       कर्नेल का उपयोग क्यों करना चाहेंगे?
    1. एक फ़ोटोग्राफ़ लोड करें, उसे एक चैनल में बदलें, और लगभग
       $100 \times 100$ पिक्सेलों पर डाउनसैंपल करें (`corr2d` Python में लूप चलाता है)।
       एज डिटेक्टर, ब्लर कर्नेल, और शार्पनिंग कर्नेल "तत्समक का दोगुना
       ऋण ब्लर" लागू करें, तथा प्रत्येक परिणाम को मूल छवि के बगल में
       प्रदर्शित करें। किसी फ़ोटोग्राफ़ का एज मैप ऐसा क्या दिखाता है जो
       काली पट्टी का एज मैप नहीं दिखा सका?

    *Stanford CS231n,
    [असाइनमेंट 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks नोटबुक से अनुकूलित।*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **बैंडेड आव्यूह के रूप में कन्वोल्यूशन।** :numref:`subsec_conv_matmul`
   का पैच आव्यूह इनपुट को खोलता है। इसके विपरीत निर्माण कर्नेल को खोलता है।
   मान लें कि $\mathbf{X}$ एक $4 \times 4$ इनपुट है, $\mathbf{K}$ एक
   $3 \times 3$ कर्नेल है, और $\textrm{vec}(\cdot)$ पंक्ति-दर-पंक्ति
   सपाटीकरण है।
    1. ऐसा $4 \times 16$ आव्यूह $\mathbf{W}$ निर्मित करें जिसके लिए
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ हो,
       जहाँ $\mathbf{Y}$, $\mathbf{X}$ का $\mathbf{K}$ के साथ क्रॉस-सहसंबंध है।
       प्रत्येक पंक्ति में कितनी अशून्य प्रविष्टियाँ हैं, और $\mathbf{W}$ में
       कितने भिन्न मान दिखाई देते हैं?
    1. कोड में $\mathbf{W}$ का निर्माण `K` से करें और पुष्टि करें कि गुणनफल
       `corr2d(X, K)` को दोहराता है।
    1. एक $n \times n$ इनपुट और $k \times k$ कर्नेल के लिए, सघन ऐरे के रूप में
       संग्रहीत $\mathbf{W}$ की प्रविष्टियाँ, $\mathbf{W}$ की अशून्य प्रविष्टियाँ,
       तथा पैच आव्यूह की प्रविष्टियाँ गिनें। प्रत्येक संख्या $n$ के साथ कैसे
       बढ़ती है?
    1. $\mathbf{W}$ का विरलता पैटर्न बनाए रखें, परंतु प्रत्येक पंक्ति को उसके अपने
       $k^2$ मान दें। परिणाम एक लोकली कनेक्टेड लेयर होती है:
       चतुर्थ-कोटि टेंसर $\mathsf{V}$ (:numref:`sec_why-conv`), जो समीकरण
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$
       में $k \times k$ विंडो वाले ऑफ़सेट $(a, b)$ तक सीमित किया गया है,
       इससे पहले कि स्थानांतरण समचरता भारों को आपस में बाँधे। एक $n \times n$
       इनपुट के लिए, इस लेयर के, समान इनपुट और आउटपुट शेप वाली फ़ुली कनेक्टेड
       लेयर के, तथा कन्वोल्यूशन के पैरामीटर गिनें।

    *CMU 11-785,
    [होमवर्क 2 भाग 1](https://deeplearning.cs.cmu.edu/S21/index.html) से अनुकूलित।*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **प्रभावी ग्रहणशील क्षेत्र।** ● इस अनुभाग का सैद्धांतिक ग्रहणशील
   क्षेत्र $r$ यह सीमा तय करता है कि कौन-से इनपुट किसी आउटपुट को प्रभावित *कर सकते*
   हैं; प्रभावी ग्रहणशील क्षेत्र यह मापता है कि वे वास्तव में कितना प्रभावित
   करते हैं :cite:`Luo.Li.Urtasun.ea.2016`। $L$ एकल-चैनल $3 \times 3$ कन्वोल्यूशनल
   लेयरों का एक स्टैक (बिना पैडिंग, बिना बायस, और बिना अरैखिकता के) $32 \times 32$
   इनपुट पर :numref:`subsec_learning_kernel` की अंतर्निर्मित लेयर का उपयोग करके बनाएँ।
   प्रत्येक कर्नेल प्रविष्टि को स्वतंत्र रूप से मानक गाउसियन वितरण से चुनें।
    1. $L = 3$ और कर्नेलों के एकल चयन (ड्रॉ) के लिए, अंतिम लेयर के केंद्र के
       निकट स्थित किसी आउटपुट इकाई से एक इकाई ग्रेडिएंट को इनपुट तक बैकप्रॉपैगेट
       करें और इनपुट ग्रेडिएंट के परिमाण का आलेख बनाएँ। इसके आधार क्षेत्र (support)
       की तुलना $7 \times 7$ सैद्धांतिक ग्रहणशील क्षेत्र से करें, और बताएँ कि
       परिमाण कहाँ सबसे बड़ा है।
    1. कर्नेलों के 500 ड्रॉ पर वर्गित इनपुट ग्रेडिएंट का माध्य निकालें,
       $L = 3$ के लिए और $L = 10$ के लिए। प्रत्येक माध्य मैप का उसकी पंक्तियों पर
       योग करें और परिणामी स्तंभ प्रोफ़ाइल को इस प्रकार सामान्यीकृत करें कि उसका
       योग एक हो। गहराई के साथ प्रोफ़ाइल के मानक विचलन और सैद्धांतिक भुजा लंबाई
       $2L + 1$ के अनुपात में क्या परिवर्तन होता है?
    1. :numref:`fig_conv_rf_paths` एक विमा में $L = 3$ के लिए पथों को गिनता है।
       दिखाएँ कि व्यापक $L$ के लिए पंक्तियों पर योग किया गया प्रत्याशित वर्गित इनपुट
       ग्रेडिएंट, $L$ प्रतियों में लिए गए सदिश $(1, 1, 1)$ के कन्वोल्यूशन के समानुपाती
       होता है। व्युत्पन्न करें कि सामान्यीकृत प्रोफ़ाइल का मानक विचलन $L$ के साथ
       कैसे बढ़ता है।

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **कन्वोल्यूशन का बैकवर्ड पास।** ● मान लें कि $\mathbf{Y}$ किसी
   इनपुट $\mathbf{X}$ का कर्नेल $\mathbf{K}$ के साथ एकल-चैनल क्रॉस-सहसंबंध है,
   जैसा कि `corr2d` द्वारा परिकलित किया जाता है, और मान लें कि $\ell$ एक अदिश लॉस है
   जो $\mathbf{X}$ और $\mathbf{K}$ पर केवल $\mathbf{Y}$ के माध्यम से निर्भर करता है।
    1. $\partial \ell/\partial \mathbf{K}$ और
       $\partial \ell/\partial \mathbf{X}$ को
       $\partial \ell/\partial \mathbf{Y}$ के पदों में, :numref:`sec_backprop`
       के बैकप्रॉपैगेशन समीकरणों की शैली में व्युत्पन्न करें।
    1. $\partial \ell/\partial \mathbf{K}$ के लिए अपने व्यंजक को बिना स्वचालित अवकलन
       के लॉस $\ell = \sum_{i,j} G_{ij} Y_{ij}$ के लिए कार्यान्वित करें, जहाँ
       $\mathbf{G}$, $\mathbf{Y}$ के शेप का एक निश्चित यादृच्छिक आव्यूह है।
       चरण आकार $10^{-2}$ वाले केंद्रित परिमित-अंतर ग्रेडिएंट से एक छोटे यादृच्छिक
       `X` और `K` पर इसकी तुलना करें, तथा अधिकतम सापेक्ष त्रुटि बताएँ।
    1. $\partial \ell/\partial \mathbf{K}$ के अपने व्यंजक का मिलान उस कर्नेल
       ग्रेडिएंट से करें जिसकी गणना स्वचालित अवकलन :numref:`subsec_learning_kernel`
       के प्रयोग में करता है।

    *Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    अध्याय 6, तथा Stanford CS231n,
    [असाइनमेंट 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks नोटबुक से अनुकूलित।*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[चर्चा](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[चर्चा](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[चर्चा](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[चर्चा](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Register & voice (style.md convention): Colleague-to-colleague academic Hindi using inclusive "हम" in exposition and "-एँ" polite imperatives ("रीशेप करें", "न्यूनतम करें") in instructions/slides.
- Punctuation & numbers (style.md convention): Use Devanagari purna viram "।" at the end of declarative sentences (omitted on headings and lines introducing display equations); retain Western Arabic numerals (`0`, `1`, `12`, `0.03`, `45%`) throughout technical prose and slides; maintain consistent nuqta in borrowed terms (`फ़ीचर`, `फ़िल्टर`, `फ़ॉरवर्ड`, `फ़्रेमवर्क`, `सफ़ेद`).
- English terms in parentheses (style.md convention): On first prose occurrence of non-transliterated Sanskrit/Hindi technical terms (`स्थानांतरण समचरता (translation equivariance)`, `स्थानीयता (locality)`, `क्रॉस-सहसंबंध (cross-correlation)`, `ग्रहणशील क्षेत्र (receptive field)`, `परिमित अंतर (finite difference)`, `अवकलज (derivative)`), include the English term in parentheses once; omit in subsequent uses and on slides.
- Grammatical gender & plural inflection: Treat `लेयर` and `विंडो` as feminine (`कन्वोल्यूशनल लेयरें`, `स्लाइडिंग विंडो`); treat `कर्नेल`, `चैनल`, `फ़िल्टर`, `टेंसर`, `पिक्सेल`, `बायस`, `लॉस`, `ग्रेडिएंट`, `फ़ीचर मैप`, `एज मैप`, and `शेप` as masculine, inflecting oblique plurals natively (`लेयरों`, `कर्नेलों`, `फ़िल्टरों`, `पिक्सेलों`, `टेंसरों`).
- "image" → "छवि"
- "locality" → "स्थानीयता"
- "edge" / "edge detection" / "edge detector" / "edge map" → "किनारा" / "किनारा पहचान" / "एज डिटेक्टर" / "एज मैप" (paralleling glossary `फ़ीचर मैप`)
- "sliding window" → "स्लाइडिंग विंडो"
- "elementwise" → "तत्ववार"
- "dense layer" → "सघन लेयर" (pairing symmetrically with glossary `विरलता` / `विरल` for sparsity/sparse)
- "synthetic image" → "सिंथेटिक छवि"
- "built-in" → "अंतर्निर्मित"
- "spatial dimensions" / "spatial shape" → "स्थानिक विमाएँ" / "स्थानिक शेप"
- "patch" / "patch matrix" → "पैच" / "पैच आव्यूह"
- "flatten" / "flattened" / "flattening" → "सपाट करना" / "सपाट" / "सपाटीकरण"
- "unfold" / "unroll" → "खोलना"
- "sparse banded matrix" → "विरल बैंडेड आव्यूह"
- "theoretical receptive field" / "effective receptive field" → "सैद्धांतिक ग्रहणशील क्षेत्र" / "प्रभावी ग्रहणशील क्षेत्र"
- "stacked layers" → "स्टैक की गई लेयरें"
- "Gaussian bell curve" → "गाउसियन घंटी वक्र"
- "trade-off" → "समझौता"
- "locally connected layer" → "लोकली कनेक्टेड लेयर" (paralleling glossary `फ़ुली कनेक्टेड लेयर`)
- "standard normal distribution" → "मानक गाउसियन वितरण" (aligning with glossary `गाउसियन वितरण`)
- "blur kernel" / "sharpening kernel" → "ब्लर कर्नेल" / "शार्पनिंग कर्नेल"
- "downsample" → "डाउनसैंपल करना"
- "Discussions" (tab link) → "चर्चा"

=== NOTES ===
- 54bcee26: "bias" denotes additive layer offset (`बायस`), not estimator bias (`अभिनति`).
- 6ab5b67c: Kept Michael Nielsen's English book title *Neural Networks and Deep Learning* untranslated per TRANSLATING.md §7 (rather than inserting `न्यूरल नेटवर्क` inside the proper English title).
