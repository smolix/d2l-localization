<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=hi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# इमेज के लिए कन्वोल्यूशन

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` में हमने दो ज्यामितीय सिद्धांतों, स्थानांतरण समचरता (translation equivariance) और स्थानीयता (locality), से कागज़ पर कन्वोल्यूशनल लेयर (convolutional layer) व्युत्पन्न की थी। अब हम देखेंगे कि वह सूत्र किसी इमेज पर वास्तव में क्या गणना करता है, और क्या ग्रेडिएंट डिसेंट (gradient descent) डेटा से अपने-आप उपयोगी कर्नेल (kernel) खोज सकता है।

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->
मान लीजिए हमें वे ऊर्ध्वाधर किनारे ढूँढने हैं जहाँ कोई चमकीला क्षेत्र किसी गहरे क्षेत्र से मिलता है। हर पिक्सेल में से उसके दाईं ओर वाले पड़ोसी को घटाने पर ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) एकसमान क्षेत्रों के भीतर शून्य मिलता है, जहाँ सफ़ेद काले में बदलता है वहाँ $+1$, और जहाँ काला फिर सफ़ेद हो जाता है वहाँ $-1$। दो भारों वाले इस डिटेक्टर $(1, -1)$ को इमेज की हर वैध स्थिति पर फिराने से एक *एज मैप* (edge map) बनता है। इस खंड में हम द्वि-आयामी क्रॉस-सहसंबंध (cross-correlation) लागू करेंगे, हाथ से बनाए अपने एज डिटेक्टर को परखेंगे, ग्रेडिएंट डिसेंट को $(1, -1)$ सीधे डेटा से *सीखने* देंगे, दिखाएँगे कि एक कन्वोल्यूशन कैसे एक ही आव्यूह गुणन (*im2col*) बन जाता है, और व्युत्पन्न करेंगे कि लेयरों को एक के ऊपर एक रखने से किसी यूनिट का ग्रहणशील क्षेत्र (receptive field) कैसे चौड़ा होता है।

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## क्रॉस-सहसंबंध

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### संक्रिया

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
फ़िलहाल चैनलों को अलग रखते हुए, मान लीजिए इनपुट $\mathbf{X}$ एक $n_\textrm{h} \times n_\textrm{w}$ आव्यूह है और कर्नेल $\mathbf{K}$ एक $k_\textrm{h} \times k_\textrm{w}$ आव्यूह। कर्नेल $k_\textrm{h} \times k_\textrm{w}$ शेप की एक खिसकती हुई *विंडो* (window) परिभाषित करता है। विंडो के ऊपरी-बाएँ कोने को $(i, j)$ पर रखकर हम ढके हुए इनपुट पैच को $\mathbf{K}$ से अवयव-वार गुणा करते हैं और गुणनफलों का योग करके $(i, j)$ पर आउटपुट प्रविष्टि प्राप्त करते हैं:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
यह $\mathbf{X}$ और $\mathbf{K}$ का *द्वि-आयामी क्रॉस-सहसंबंध* है, जिसे $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ लिखा जाता है; इसके सूचकांक कोड की तरह शून्य से शुरू होते हैं। :numref:`sec_why-conv` में ऑफ़सेट $-\Delta$ से $\Delta$ तक सममित रूप से केंद्रित थे, जबकि ऊपरी-बाएँ कोने पर टिकाने से केवल $\Delta$ का एक स्थिर खिसकाव आता है, और साथ ही उन सम आकार वाले कर्नेलों (जैसे $2 \times 2$ कर्नेल या हमारा $1 \times 2$ एज डिटेक्टर) को भी सहज रूप से सँभाला जा सकता है जिनका कोई केंद्रीय पिक्सेल नहीं होता।

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![द्वि-आयामी क्रॉस-सहसंबंध। छायांकित भाग पहला आउटपुट अवयव और उसकी गणना में प्रयुक्त इनपुट और कर्नेल अवयव हैं: $0\times0+1\times1+3\times2+4\times3=19$।](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` में हम एक $2 \times 2$ कर्नेल को $3 \times 3$ इनपुट पर एक-एक पिक्सेल करके, बाएँ से दाएँ और ऊपर से नीचे, खिसकाते हैं। विंडो चार स्थितियों पर पूरी तरह फ़िट होती है, जिससे $2 \times 2$ आउटपुट मिलता है:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
ध्यान दें कि आउटपुट हर अक्ष के साथ थोड़ा सिकुड़ जाता है, क्योंकि कर्नेल को पूरी तरह इनपुट के भीतर रहना होता है। चौड़ाई-$k$ वाली विंडो को लंबाई $n$ में रखने पर $n - k + 1$ वैध स्थितियाँ बचती हैं, इसलिए आउटपुट का शेप है

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
इस प्रकार $3 \times 3$ इनपुट पर $2 \times 2$ कर्नेल से $2 \times 2$ आउटपुट मिलता है, $6 \times 8$ इमेज पर $1 \times 2$ कर्नेल से $6 \times 7$, और $28 \times 28$ की Fashion-MNIST इमेज पर $5 \times 5$ कर्नेल से $24 \times 24$। जब भी हम स्थानिक आयामों को बनाए रखना चाहते हैं, इनपुट के चारों ओर शून्यों का एक घेरा (*पैडिंग*, padding) लगा देते हैं, जिसे हम स्ट्राइड (stride) और डाइलेशन (dilation) के साथ :numref:`sec_padding` में देखेंगे।

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### कार्यान्वयन

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
हम :eqref:`eq_corr2d` को सीधे `corr2d` में लागू करते हैं, जिसमें हर आउटपुट निर्देशांक पर चलने वाले दो लूप हैं।

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` को :numref:`fig_correlation` में हाथ से की गई अपनी गणना से जाँचने के लिए हम $3 \times 3$ इनपुट टेंसर `X` और $2 \times 2$ कर्नेल `K` बनाते हैं और `corr2d(X, K)` का मान निकालते हैं।

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
आउटपुट :numref:`fig_correlation` को प्रविष्टि-दर-प्रविष्टि दोहराता है (और जैसा :numref:`sec_why-conv` में दिखाया गया था, पलटे हुए कर्नेल $\mathbf{K}'$ के साथ सख़्त गणितीय कन्वोल्यूशन ठीक लेयरों का वही परिवार सीखता है, इसलिए हम डीप लर्निंग की मानक शब्दावली अपनाते हुए क्रॉस-सहसंबंध को "कन्वोल्यूशन" कहते हैं)।

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## कन्वोल्यूशनल लेयर

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### लेयर

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
द्वि-आयामी कन्वोल्यूशनल लेयर अपने इनपुट का एक सीखे गए कर्नेल के साथ क्रॉस-सहसंबंध लेती है और एक अदिश बायस (bias) जोड़ती है। :numref:`chap_perceptrons` के अपने MLP भारों की तरह ही हम कर्नेल को $\mathcal{N}(0, 0.01^2)$ से इनिशियलाइज़ करते हैं और बायस को शून्य पर। नीचे `Conv2D` दोनों पैरामीटरों को दर्ज करता है और अपने फ़ॉरवर्ड पास (forward pass) में `corr2d` को बुलाता है; ऊँचाई $h$ और चौड़ाई $w$ वाले कर्नेल की लेयर को $h \times w$ कन्वोल्यूशनल लेयर कहा जाता है।

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
डेंस लेयर के विपरीत, कन्वोल्यूशनल लेयर किसी निश्चित इनपुट रिज़ॉल्यूशन से बँधी नहीं होती, क्योंकि उसके पैरामीटर केवल कर्नेल के आकार पर निर्भर करते हैं। ठीक वही $2 \times 2$ लेयर (कुल 5 पैरामीटर) हमारे $3 \times 3$ टेंसर `X` और एक $6 \times 8$ इमेज, दोनों को प्रोसेस करती है और $2 \times 2$ तथा $5 \times 7$ शेप के आउटपुट देती है:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### एज डिटेक्शन

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
अब हम शुरुआत में बताए गए ऊर्ध्वाधर एज डिटेक्टर को एक $6 \times 8$ कृत्रिम इमेज पर परखते हैं, जिसके बीच के चार कॉलम काले (`0`) हैं और बाहरी कॉलम सफ़ेद (`1`)।

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
हम $1 \times 2$ कर्नेल परिभाषित करते हैं

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
हर स्थिति $(i, j)$ पर यह क्षैतिज *परिमित अंतर* (finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना करता है, जो चमक के क्षैतिज अवकलज (derivative) का (चिह्न को छोड़कर) एक असतत सन्निकटन है। यह सपाट क्षेत्रों में शून्य हो जाता है और जहाँ भी तीव्रता उछलती है वहाँ तेज़ी से उभर आता है।

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ धारीदार इमेज `X` का $1 \times 2$ क्षैतिज-अंतर कर्नेल `K` के साथ `corr2d(X, K)` द्वारा क्रॉस-सहसंबंध लेने पर $6 \times 7$ का एज मैप `Y` मिलता है:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
जैसी उम्मीद थी, `Y` सफ़ेद-से-काले सीमा पर `+1`, काले-से-सफ़ेद सीमा पर `-1`, और बाकी हर जगह `0` देता है। ठीक यही कर्नेल ट्रांसपोज़ की गई इमेज पर लगाने से (जिससे ऊर्ध्वाधर पट्टी क्षैतिज बन जाती है) सब शून्य मिलते हैं:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
स्पष्ट है कि `[1, -1]` केवल ऊर्ध्वाधर किनारों पर प्रतिक्रिया देता है। क्षैतिज किनारों, विकर्णों, कोनों या बनावटों का पता लगाने के लिए बिल्कुल अलग कर्नेल चाहिए।

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### कर्नेल सीखना

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->
जब हमें ठीक-ठीक पता हो कि कौन-सा फ़ीचर (feature) चाहिए, तब `[1, -1]` हाथ से बनाना आसान है, लेकिन कोई भी इंजीनियर किसी डीप नेटवर्क के हज़ारों मध्यवर्ती फ़िल्टर (filter) हाथ से तय नहीं कर सकता। क्या ग्रेडिएंट डिसेंट कर्नेल `[1, -1]` को केवल इनपुट–आउटपुट जोड़ी `(X, Y)` से वापस पा सकता है? इसे हम हर फ़्रेमवर्क की अंतर्निहित 2D कन्वोल्यूशनल लेयर से परखते हैं (बायस के बिना, और भारों को $\mathcal{N}(0, 0.01^2)$ से इनिशियलाइज़ करके): सभी 42 आउटपुट प्रविष्टियों पर जोड़ी गई वर्ग त्रुटि (squared error) को 10 ग्रेडिएंट चरणों में, लर्निंग रेट (learning rate) $0.03$ के साथ, न्यूनतम करते हैं। चूँकि अंतर्निहित लेयरें बैच और चैनल अक्षों सहित 4D टेंसर की अपेक्षा करती हैं (:numref:`sec_channels`), हम `X` और `Y` का शेप उसी के अनुसार बदल देते हैं।

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
शुरुआत में लॉस 12 के आसपास रहता है (लगभग शून्य यादृच्छिक कर्नेल सभी 12 $\pm 1$ एज पिक्सेलों से लगभग 1 से चूक जाता है), और सिर्फ़ नौ चरणों के भीतर यह 0.01 से नीचे आ जाता है। सीखे गए कर्नेल को देखकर पुष्टि होती है कि उसने क्या पाया:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
दोनों प्रविष्टियाँ `[1, -1]` से लगभग 0.02 के भीतर आ जाती हैं। इस लूप में कुछ भी किनारों के बारे में नहीं जानता था: लॉस का संकेत मिलने पर ग्रेडिएंट डिसेंट ने फ़िल्टर खुद खोज लिया। यहाँ हमने इस एकल लेयर को सीधे `Y` से पर्यवेक्षित किया; किसी डीप CNN में बैकप्रॉपैगेशन (backpropagation) केवल अंतिम कार्य के लॉस से हर लेयर के कर्नेल एंड-टू-एंड सीख लेता है।

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
इमेज के लिए कन्वोल्यूशन<br>**2D क्रॉस-सहसंबंध · एज डिटेक्शन · सीखे गए कर्नेल · im2col · ग्रहणशील क्षेत्र**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="पूरी इमेज में स्थानीय किनारों का पता लगाना"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[प्रेरणा और मूल मानसिक मॉडल]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **कृत्रिम $6 \times 8$ परीक्षण इमेज:** सफ़ेद बाहरी कॉलम (`1`), जो 4 कॉलम चौड़ी काली ऊर्ध्वाधर पट्टी (`0`) को घेरे हैं।

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **क्षैतिज परिमित अंतर:** $\mathbf{K} = [1, -1]$ को फिराने से $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना होती है, जो सफ़ेद-से-काले किनारों पर `+1` और काले-से-सफ़ेद किनारों पर `-1` देता है।

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **सीखना बनाम हाथ से बनाना:** डीप CNN ऐसे हज़ारों कर्नेल ग्रेडिएंट डिसेंट से अपने-आप खोज लेते हैं।

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2D क्रॉस-सहसंबंध संक्रिया"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[खिसकती विंडो के आंतरिक गुणनफल]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
एक $n_\textrm{h} \times n_\textrm{w}$ इनपुट आव्यूह $\mathbf{X}$ और एक $k_\textrm{h} \times k_\textrm{w}$ कर्नेल $\mathbf{K}$ दिए होने पर, 2D क्रॉस-सहसंबंध $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ विंडो के ऊपरी-बाएँ कोने को $(i, j)$ पर रखता है और अवयव-वार गुणनफलों का योग करता है:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **ऊपरी-बायाँ आधार-बिंदु:** शून्य से शुरू होने वाले सूचकांक $a, b \ge 0$ कोड के ऐरे सूचकांकों से मेल खाते हैं और सम आकार वाले कर्नेलों (जैसे $2 \times 2$ या $1 \times 2$) को आसानी से सँभालते हैं।

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **छायांकित ऊपरी-बायाँ आउटपुट $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$।

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="वैध विंडो स्थितियाँ और आउटपुट का स्थानिक शेप"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[पैडिंग-रहित कन्वोल्यूशन ग्रिड को क्यों सिकोड़ते हैं]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **वैध विंडो स्थितियाँ:** एक $k_\textrm{h} \times k_\textrm{w}$ विंडो $n_\textrm{h} \times n_\textrm{w}$ ग्रिड के भीतर हर अक्ष पर $n - k + 1$ स्थितियों पर फ़िट होती है।

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **शेप के उदाहरण:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$।

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="कोड में 2D क्रॉस-सहसंबंध लागू करना"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[सीधा नेस्टेड-लूप कार्यान्वयन]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
फ़ंक्शन `corr2d` $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ आउटपुट टेंसर आवंटित करता है और हर प्रविष्टि को `X[i:i+h, j:j+w] * K` के योग के रूप में निकालता है:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
हर पुनरावृत्ति `X` की एक $k_\textrm{h} \times k_\textrm{w}$ उप-विंडो काटती है, उसे `K` से अवयव-वार गुणा करती है और `sum` से उसका योग निकालती है।

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d की जाँच और पलटे कर्नेल की तुल्यता"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[क्रॉस-सहसंबंध बनाम गणितीय कन्वोल्यूशन]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° पलटा कर्नेल $\mathbf{K}'$:** $\mathbf{K}$ के साथ क्रॉस-सहसंबंध, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ के साथ कन्वोल्यूशन के बराबर है।

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **सीखा जाने वाला वर्ग वही:** फ़्रेमवर्क क्रॉस-सहसंबंध की गणना करते हैं और उसे कन्वोल्यूशन कहते हैं।

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2D कन्वोल्यूशनल लेयर मॉड्यूल बनाना"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[सीखने योग्य कर्नेल भार और अदिश बायस]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
कन्वोल्यूशनल लेयर `corr2d` को एक सीखने योग्य $k_\textrm{h} \times k_\textrm{w}$ भार पैरामीटर ($\mathcal{N}(0, 0.01^2)$ से इनिशियलाइज़) और एक अदिश बायस ($0$ पर इनिशियलाइज़) के साथ लपेटती है:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="कन्वोल्यूशनल लेयरों की रिज़ॉल्यूशन से स्वतंत्रता"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[किसी भी इनपुट आकार के लिए वही पैरामीटर]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ से स्वतंत्र:** ठीक वही 5 पैरामीटर ($2 \times 2$ कर्नेल + 1 बायस) $3 \times 3$, $6 \times 8$ या मेगापिक्सेल इनपुट को भारों का शेप बदले बिना प्रोसेस करते हैं।

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="ऊर्ध्वाधर किनारे वाली कृत्रिम इमेज बनाना"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[एक 6×8 द्विआधारी परीक्षण पैटर्न]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
हम $6 \times 8$ परीक्षण इमेज `X` बनाते हैं, जिसके बाएँ और दाएँ सिरों पर इकाई मान हैं और बीच के चार कॉलमों में शून्य (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **खंडशः स्थिर कॉलम:** तीव्रता (`1` या `0`) कॉलम $0\text{--}1$, $2\text{--}5$ और $6\text{--}7$ के भीतर स्थिर रहती है और केवल कॉलम $1 \to 2$ तथा $5 \to 6$ के बीच उछलती है।

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="क्षैतिज परिमित-अंतर कर्नेल लगाना"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] से ऊर्ध्वाधर किनारों का पता लगाना]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` का $1 \times 2$ कर्नेल $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ के साथ क्रॉस-सहसंबंध हर क्षैतिज जोड़ी पर $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना करता है:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **चिह्नयुक्त एज मैप:** कॉलम 1 पर `+1` (सफ़ेद से काले में बदलाव), कॉलम 5 पर `-1` (काले से सफ़ेद में बदलाव), और सभी स्थिर भीतरी कॉलमों पर `0` आता है।

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="एज कर्नेलों की दिशा-चयनात्मकता"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[एक कर्नेल सभी दिशाओं का पता क्यों नहीं लगा सकता]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **क्षैतिज किनारों के प्रति अंधा:** `X` को ट्रांसपोज़ करने पर (`X.T`) क्षैतिज पड़ोसी एक जैसे हो जाते हैं, इसलिए $[1, -1]$ सब शून्य देता है।

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **फ़िल्टर बैंक (§{sec_channels}):** विज़न लेयरें क्षैतिज, ऊर्ध्वाधर और विकर्ण दिशाओं के लिए समानांतर चैनल सीखती हैं।

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="ग्रेडिएंट डिसेंट से एज डिटेक्टर सीखना"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) पर अंतर्निहित कन्वोल्यूशनल लेयर का पर्यवेक्षण]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **फ़्रेमवर्क के 4D टेंसर:** `X` और `Y` को 4D टेंसर (एक उदाहरण, एक चैनल) में बदलें और $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ को $\eta = 0.03$ पर न्यूनतम करें।

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="सीखे गए कर्नेल भारों की जाँच"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[आरंभिक लॉस 12 क्यों है और [1, -1] पर अभिसरण क्यों होता है]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **लॉस $12.0$ से क्यों शुरू होता है:** लगभग शून्य भार $\hat{Y} \approx 0$ का पूर्वानुमान देते हैं और `Y` के बारह $\pm 1$ एज पिक्सेलों से चूक जाते हैं ($12 \times 1^2 = 12$)।

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 चरणों में अभिसरण:** SGD डेटा से अपने-आप $[0.99, -0.99]$ प्राप्त कर लेता है।

=== DECISIONS ===
- Register: the book's "we" → "हम", reader-facing imperatives in polite form ("ध्यान दें", "मान लीजिए", "बदलें"); sentence end "।", other punctuation as in English (`:`, `;`, `,`, parentheses).
- Digits stay Western Arabic (0–9) in prose, as in the English.
- English term in parentheses after the Hindi on the first use of a key term in a section, e.g. ग्रहणशील क्षेत्र (receptive field).
- "image" → "इमेज" (not छवि), consistent with the transliterated ML vocabulary (पिक्सेल, चैनल, फ़ीचर मैप).
- "edge" (in an image) → "किनारा"; "edge map" → "एज मैप" (parallel to फ़ीचर मैप); "edge detector" → "एज डिटेक्टर"; "edge detection" → "एज डिटेक्शन".
- "vertical"/"horizontal" → "ऊर्ध्वाधर"/"क्षैतिज"; "diagonal" → "विकर्ण".
- "window" (sliding window) → "विंडो"; "patch" → "पैच"; "entry" (of an array/output) → "प्रविष्टि"; "element" → "अवयव"; "elementwise" → "अवयव-वार".
- "input"/"output" → "इनपुट"/"आउटपुट"; "parameter" → "पैरामीटर"; "framework" → "फ़्रेमवर्क"; "dense layer" → "डेंस लेयर"; "initialize" → "इनिशियलाइज़ करना".
- "built-in" → "अंतर्निहित"; "gradient step" → "ग्रेडिएंट चरण"; "axis" → "अक्ष"; "index" → "सूचकांक".
- "लेयर" is feminine ("कन्वोल्यूशनल लेयर लेती है"); plural "लेयरें"/oblique "लेयरों".
- Headings: "Cross-Correlation" → "क्रॉस-सहसंबंध"; "The Operation" → "संक्रिया"; "Implementation" → "कार्यान्वयन".

=== NOTES ===
- 1d94295a and other slide_title blocks: only the title attribute is translated; identifiers in titles (`corr2d` in 4ee9f6b1) stay English.
- 0a4b456b: "ones" rendered as "इकाई मान" (unit values) to avoid inserting a digit not present in the English.
