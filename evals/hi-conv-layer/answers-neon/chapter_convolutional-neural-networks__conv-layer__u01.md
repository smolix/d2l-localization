<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=hi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# छवियों के लिए कन्वोल्यूशन

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` में हमने दो ज्यामितीय सिद्धांतों—स्थानांतरण समचरता और स्थानीयता—से कागज़ पर कन्वोल्यूशनल लेयर को व्युत्पन्न किया था। अब हम देखते हैं कि वह सूत्र किसी छवि पर वास्तव में क्या गणना करता है, और क्या ग्रेडिएंट डिसेंट डेटा से उपयोगी कर्नेल स्वतः खोज सकता है।

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
उन ऊर्ध्वाधर किनारों का पता लगाने पर विचार कीजिए जहाँ कोई चमकीला क्षेत्र किसी गहरे क्षेत्र से मिलता है। प्रत्येक पिक्सेल में से उसके दाएँ पड़ोसी को घटाने ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) पर एकसमान क्षेत्रों के भीतर शून्य प्राप्त होता है, जहाँ सफ़ेद से काला संक्रमण होता है वहाँ $+1$, और जहाँ काला वापस सफ़ेद में बदलता है वहाँ $-1$ मिलता है। इस दो-भार वाले संसूचक $(1, -1)$ को छवि के प्रत्येक वैध स्थान पर चलाने से एक *एज मैप* (edge map) प्राप्त होता है। इस खंड में, हम द्वि-आयामी क्रॉस-सहसंबंध को लागू करते हैं, अपने हाथ से बनाए गए किनारा संसूचक का परीक्षण करते हैं, ग्रेडिएंट डिसेंट को सीधे डेटा से $(1, -1)$ *सीखने* देते हैं, दर्शाते हैं कि कन्वोल्यूशन कैसे एक एकल आव्यूह गुणन (*im2col*) बन जाता है, और व्युत्पन्न करते हैं कि लेयरों को स्टैक करने से किसी इकाई का ग्रहणशील क्षेत्र कैसे विस्तृत होता है।

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
फ़िलहाल चैनलों को अलग रखते हुए, मान लीजिए कि इनपुट $\mathbf{X}$ एक $n_\textrm{h} \times n_\textrm{w}$ आव्यूह है और कर्नेल $\mathbf{K}$ एक $k_\textrm{h} \times k_\textrm{w}$ आव्यूह है। कर्नेल $k_\textrm{h} \times k_\textrm{w}$ शेप की एक खिसकती हुई *विंडो* को परिभाषित करता है। विंडो के ऊपरी-बाएँ कोने को $(i, j)$ पर रखकर, हम आच्छादित इनपुट पैच का $\mathbf{K}$ के साथ तत्व-वार गुणन करते हैं और $(i, j)$ पर आउटपुट प्रविष्टि प्राप्त करने के लिए गुणनफलों का योग करते हैं:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
यह $\mathbf{X}$ का $\mathbf{K}$ के साथ *द्वि-आयामी क्रॉस-सहसंबंध* (two-dimensional cross-correlation) है, जिसे कोड के अनुरूप शून्य-आधारित सूचकांकों के साथ $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ लिखा जाता है। जबकि :numref:`sec_why-conv` में ऑफ़सेट को $-\Delta$ से $\Delta$ तक सममित रूप से केंद्रित किया गया था, ऊपरी-बाएँ कोने पर स्थिर करने से केवल $\Delta$ का एक निश्चित विस्थापन आता है और साथ ही यह सम-आकार वाले कर्नेलों (जैसे $2 \times 2$ कर्नेल या हमारा $1 \times 2$ किनारा संसूचक), जिनका कोई केंद्रीय पिक्सेल नहीं होता, को स्वाभाविक रूप से संभाल लेता है।

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![द्वि-आयामी क्रॉस-सहसंबंध। छायांकित भाग पहला आउटपुट तत्व तथा उसकी गणना में प्रयुक्त इनपुट और कर्नेल तत्व हैं: $0\times0+1\times1+3\times2+4\times3=19$।](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` में, हम एक $2 \times 2$ कर्नेल को $3 \times 3$ इनपुट पर एक बार में एक पिक्सेल—बाएँ से दाएँ और ऊपर से नीचे—खिसकाते हैं। विंडो चार स्थानों पर पूरी तरह फ़िट बैठती है, जिससे $2 \times 2$ आउटपुट प्राप्त होता है:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
ध्यान दीजिए कि आउटपुट प्रत्येक अक्ष के अनुदिश थोड़ा सिकुड़ जाता है क्योंकि कर्नेल को पूरी तरह इनपुट के भीतर रहना होता है। लंबाई $n$ में चौड़ाई-$k$ की विंडो को फ़िट करने पर $n - k + 1$ वैध स्थान मिलते हैं, इसलिए आउटपुट का शेप होता है

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
इस प्रकार $2 \times 2$ कर्नेल के साथ $3 \times 3$ इनपुट से $2 \times 2$, $1 \times 2$ कर्नेल के साथ $6 \times 8$ छवि से $6 \times 7$, और $5 \times 5$ कर्नेल के साथ $28 \times 28$ Fashion-MNIST छवि से $24 \times 24$ आउटपुट मिलता है। जब भी हम स्थानिक विमाओं को संरक्षित रखना चाहते हैं, तो हम इनपुट के चारों ओर शून्यों की एक सीमा (*पैडिंग*) जोड़ देते हैं, जिसका अध्ययन हम :numref:`sec_padding` में स्ट्राइड और डाइलेशन के साथ करेंगे।

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### कार्यान्वयन

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
हम प्रत्येक आउटपुट निर्देशांक पर लूपों के एक युग्म का उपयोग करके `corr2d` में सीधे :eqref:`eq_corr2d` को लागू करते हैं।

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
:numref:`fig_correlation` में हमारी हस्तलिखित गणना के विरुद्ध `corr2d` को सत्यापित करने के लिए, हम $3 \times 3$ इनपुट टेंसर `X` और $2 \times 2$ कर्नेल `K` बनाते हैं और `corr2d(X, K)` का मूल्यांकन करते हैं।

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
आउटपुट :numref:`fig_correlation` को प्रविष्टि-दर-प्रविष्टि हूबहू पुनरुत्पादित करता है (और जैसा कि :numref:`sec_why-conv` में दिखाया गया है, उलटे गए कर्नेल $\mathbf{K}'$ के साथ शुद्ध गणितीय कन्वोल्यूशन ठीक उसी लेयर परिवार को सीखता है, इसलिए हम मानक डीप लर्निंग शब्दावली का अनुसरण करते हुए क्रॉस-सहसंबंध को "कन्वोल्यूशन" कहते हैं)।

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
एक द्वि-आयामी कन्वोल्यूशनल लेयर अपने इनपुट का सीखे गए कर्नेल के साथ क्रॉस-सहसंबंध निकालती है और उसमें एक अदिश बायस जोड़ती है। :numref:`chap_perceptrons` में हमारे MLP भारों की तरह ही, हम कर्नेल को $\mathcal{N}(0, 0.01^2)$ से और बायस को शून्य से आरंभीकृत करते हैं। नीचे, `Conv2D` दोनों पैरामीटरों को पंजीकृत करता है और अपने फ़ॉरवर्ड पास में `corr2d` को कॉल करता है; ऊँचाई $h$ और चौड़ाई $w$ के कर्नेल वाली लेयर को $h \times w$ कन्वोल्यूशनल लेयर कहा जाता है।

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
डेंस लेयर के विपरीत, कन्वोल्यूशनल लेयर किसी निश्चित इनपुट रिज़ॉल्यूशन तक सीमित नहीं होती क्योंकि इसके पैरामीटर केवल कर्नेल के आकार पर निर्भर करते हैं। ठीक वही $2 \times 2$ लेयर (कुल 5 पैरामीटर) हमारे $3 \times 3$ टेंसर `X` और $6 \times 8$ छवि दोनों को संसाधित करती है, और क्रमशः $2 \times 2$ तथा $5 \times 7$ शेप के आउटपुट देती है:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### किनारा पहचान

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
अब हम परिचय वाले ऊर्ध्वाधर किनारा संसूचक का परीक्षण एक $6 \times 8$ कृत्रिम छवि पर करते हैं, जिसके मध्य चार स्तंभ काले (`0`) और बाहरी स्तंभ सफ़ेद (`1`) हैं।

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
हम $1 \times 2$ कर्नेल को इस प्रकार परिभाषित करते हैं

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
प्रत्येक स्थान $(i, j)$ पर यह क्षैतिज *परिमित अंतर* (finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना करता है, जो चमक के क्षैतिज अवकलज का (चिह्न के अंतर तक) एक विविक्त सन्निकटन है; यह सपाट क्षेत्रों में शून्य हो जाता है और जहाँ भी तीव्रता में अचानक परिवर्तन होता है वहाँ तीक्ष्ण मान दर्शाता है।

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
`corr2d(X, K)` के माध्यम से $6 \times 8$ पट्टी वाली छवि `X` का $1 \times 2$ क्षैतिज-अंतर कर्नेल `K` के साथ क्रॉस-सहसंबंध निकालने पर $6 \times 7$ एज मैप `Y` प्राप्त होता है:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
निस्संदेह, `Y` सफ़ेद-से-काली सीमा पर `+1`, काली-से-सफ़ेद सीमा पर `-1`, और शेष सभी स्थानों पर `0` देता है। ठीक उसी कर्नेल को परिवर्तित (transposed) छवि पर लागू करने (जिससे ऊर्ध्वाधर पट्टी क्षैतिज पट्टी में बदल जाती है) पर सभी मान शून्य प्राप्त होते हैं:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
स्पष्ट है कि `[1, -1]` केवल ऊर्ध्वाधर किनारों पर ही प्रतिक्रिया करता है। क्षैतिज किनारों, विकर्णों, कोनों या बनावटों (textures) का पता लगाने के लिए पूरी तरह अलग कर्नेलों की आवश्यकता होती है।

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
जब हमें ठीक-ठीक पता हो कि कौन-सा फ़ीचर चाहिए, तो `[1, -1]` को हाथ से बनाना आसान है, लेकिन कोई भी इंजीनियर किसी गहरे नेटवर्क के हज़ारों मध्यवर्ती फ़िल्टरों को हाथ से निर्दिष्ट नहीं कर सकता। क्या ग्रेडिएंट डिसेंट केवल इनपुट–आउटपुट युग्म `(X, Y)` से कर्नेल `[1, -1]` को पुनः प्राप्त कर सकता है? हम प्रत्येक फ़्रेमवर्क की अंतर्निहित 2D कन्वोल्यूशनल लेयर का उपयोग करके इसका परीक्षण करते हैं (बायस को हटाकर और भारों को $\mathcal{N}(0, 0.01^2)$ से आरंभीकृत करके), जिसमें $0.03$ के लर्निंग रेट पर 10 ग्रेडिएंट चरणों में सभी 42 आउटपुट प्रविष्टियों पर योग की गई वर्ग त्रुटि को न्यूनतम किया जाता है। चूँकि अंतर्निहित लेयरें बैच और चैनल अक्षों सहित 4D टेंसर की अपेक्षा करती हैं (:numref:`sec_channels`), इसलिए हम `X` और `Y` को उसी के अनुरूप रीशेप करते हैं।

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
प्रारंभ में लॉस 12 के निकट रहता है (शून्य के करीब का यादृच्छिक कर्नेल सभी 12 $\pm 1$ किनारा पिक्सेलों से लगभग 1 से चूक जाता है), और केवल नौ चरणों में यह 0.01 से नीचे आ जाता है। सीखे गए कर्नेल का निरीक्षण करने से पुष्टि होती है कि उसने क्या खोजा:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
दोनों प्रविष्टियाँ `[1, -1]` के लगभग 0.02 के भीतर पहुँच जाती हैं। इस लूप में किसी को भी किनारों के बारे में कुछ नहीं पता था: एक लॉस संकेत मिलने पर ग्रेडिएंट डिसेंट ने फ़िल्टर को स्वयं खोज लिया। यहाँ हमने एकल लेयर को सीधे `Y` से पर्यवेक्षित किया; एक गहरे CNN में, बैकप्रॉपैगेशन केवल अंतिम कार्य के लॉस से हर लेयर के कर्नेल को आरंभ से अंत तक सीख लेता है।

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
छवियों के लिए कन्वोल्यूशन<br>**2D क्रॉस-सहसंबंध · किनारा पहचान · सीखे गए कर्नेल · im2col · ग्रहणशील क्षेत्र**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="छवि में स्थानीय किनारों की पहचान"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[प्रेरणा और मूल मानसिक मॉडल]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **कृत्रिम $6 \times 8$ परीक्षण छवि:** 4-स्तंभ वाली काली ऊर्ध्वाधर पट्टी (`0`) को घेरते सफ़ेद बाहरी स्तंभ (`1`)।

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **क्षैतिज परिमित अंतर:** $\mathbf{K} = [1, -1]$ को चलाने से $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना होती है, जो सफ़ेद-से-काले किनारे पर `+1` और काले-से-सफ़ेद किनारे पर `-1` देता है।

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **सीखना बनाम हाथ से बनाना:** गहरे CNN ग्रेडिएंट डिसेंट के माध्यम से ऐसे हज़ारों कर्नेल स्वतः खोज लेते हैं।

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2D क्रॉस-सहसंबंध संक्रिया"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[स्लाइडिंग-विंडो आंतरिक गुणनफल]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ इनपुट आव्यूह $\mathbf{X}$ और $k_\textrm{h} \times k_\textrm{w}$ कर्नेल $\mathbf{K}$ के लिए, 2D क्रॉस-सहसंबंध $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ विंडो के ऊपरी-बाएँ कोने को $(i, j)$ पर रखता है और तत्व-वार गुणनफलों का योग करता है:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **ऊपरी-बायाँ आधार:** शून्य-आधारित सूचकांक $a, b \ge 0$ कोड की ऐरे इंडेक्सिंग से मेल खाते हैं और सम-आकार कर्नेलों (जैसे $2 \times 2$ या $1 \times 2$) को सहजता से संभालते हैं।

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **छायांकित ऊपरी-बायाँ आउटपुट $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$।

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="वैध विंडो स्थान और आउटपुट का स्थानिक शेप"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[बिना पैडिंग वाले कन्वोल्यूशन ग्रिड को क्यों सिकोड़ते हैं]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **वैध विंडो स्थान:** एक $k_\textrm{h} \times k_\textrm{w}$ विंडो $n_\textrm{h} \times n_\textrm{w}$ ग्रिड के भीतर प्रति अक्ष $n - k + 1$ स्थानों पर फ़िट होती है।

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **शेप उदाहरण:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$।

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="कोड में 2D क्रॉस-सहसंबंध का कार्यान्वयन"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[प्रत्यक्ष नेस्टेड-लूप कार्यान्वयन]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
फ़ंक्शन `corr2d` $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ आउटपुट टेंसर आवंटित करता है और प्रत्येक प्रविष्टि की गणना `X[i:i+h, j:j+w] * K` के योग के रूप में करता है:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
प्रत्येक पुनरावृत्ति `X` की एक $k_\textrm{h} \times k_\textrm{w}$ उप-विंडो को स्लाइस करती है, `K` के साथ तत्व-वार गुणा करती है, और `sum` से योग निकालती है।

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d का सत्यापन और उलटे कर्नेल की तुल्यता"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[क्रॉस-सहसंबंध बनाम गणितीय कन्वोल्यूशन]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° उलटा गया कर्नेल $\mathbf{K}'$:** $\mathbf{K}$ के साथ क्रॉस-सहसंबंध, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ के साथ कन्वोल्यूशन के बराबर है।

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **समान सीखा गया वर्ग:** फ़्रेमवर्क क्रॉस-सहसंबंध की गणना करते हैं और इसे कन्वोल्यूशन कहते हैं।

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2D कन्वोल्यूशनल लेयर मॉड्यूल का निर्माण"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[सीखने योग्य कर्नेल भार और अदिश बायस]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
एक कन्वोल्यूशनल लेयर `corr2d` को एक सीखने योग्य $k_\textrm{h} \times k_\textrm{w}$ भार पैरामीटर ($\mathcal{N}(0, 0.01^2)$ से आरंभीकृत) और एक अदिश बायस ($0$ से आरंभीकृत) के साथ लपेटती है:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="कन्वोल्यूशनल लेयरों की रिज़ॉल्यूशन स्वतंत्रता"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[किसी भी इनपुट आकार पर स्थिर पैरामीटर]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ से स्वतंत्र:** ठीक वही 5 पैरामीटर ($2 \times 2$ कर्नेल + 1 बायस) भारों को रीशेप किए बिना $3 \times 3$, $6 \times 8$, या मेगापिक्सेल इनपुट को संसाधित करते हैं।

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="कृत्रिम ऊर्ध्वाधर-किनारा छवि का निर्माण"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[एक 6×8 द्विआधारी परीक्षण पैटर्न]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
हम बाईं और दाईं सीमाओं पर 1 तथा मध्य के चार स्तंभों में 0 (`X[:, 2:6] = 0`) रखकर $6 \times 8$ परीक्षण छवि `X` बनाते हैं:

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **खंडशः-अचर स्तंभ:** स्तंभ $0\text{--}1$, $2\text{--}5$, और $6\text{--}7$ के भीतर तीव्रता स्थिर (`1` या `0`) रहती है, और केवल स्तंभ $1 \to 2$ तथा $5 \to 6$ के बीच बदलती है।

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="क्षैतिज परिमित-अंतर कर्नेल लागू करना"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] से ऊर्ध्वाधर किनारों की पहचान]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` का $1 \times 2$ कर्नेल $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ के साथ क्रॉस-सहसंबंध प्रत्येक क्षैतिज युग्म पर $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ की गणना करता है:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **चिह्नित एज मैप:** स्तंभ 1 पर `+1` (सफ़ेद-से-काला संक्रमण), स्तंभ 5 पर `-1` (काले-से-सफ़ेद संक्रमण), और सभी स्थिर आंतरिक स्तंभों पर `0` मिलता है।

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="किनारा कर्नेलों की अभिविन्यास चयनात्मकता"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[एक कर्नेल सभी अभिविन्यासों को क्यों नहीं पहचान सकता]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **क्षैतिज किनारों के प्रति असंवेदनशील:** `X` को ट्रांसपोज़ (`X.T`) करने पर क्षैतिज पड़ोसी समान हो जाते हैं, इसलिए $[1, -1]$ का आउटपुट पूर्णतः शून्य आता है।

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **फ़िल्टर बैंक (§{sec_channels}):** विज़न लेयरें क्षैतिज, ऊर्ध्वाधर और विकर्ण अभिविन्यासों के लिए समानांतर चैनल सीखती हैं।

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="ग्रेडिएंट डिसेंट से किनारा संसूचक सीखना"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) पर अंतर्निहित कन्वोल्यूशनल लेयर का पर्यवेक्षण]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4D फ़्रेमवर्क टेंसर:** `X` और `Y` को 4D टेंसर (एक उदाहरण, एक चैनल) में रीशेप कीजिए और $\eta = 0.03$ पर $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ को न्यूनतम कीजिए।

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="सीखे गए कर्नेल भारों का निरीक्षण"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[प्रारंभिक लॉस 12 क्यों है और [1, -1] पर अभिसरण क्यों होता है]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **लॉस $12.0$ से क्यों शुरू होता है:** शून्य के निकट के भार $\hat{Y} \approx 0$ का पूर्वानुमान लगाते हैं और `Y` के बारह $\pm 1$ किनारा पिक्सेलों से चूकते हैं ($12 \times 1^2 = 12$)।

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 चरणों में अभिसरण:** SGD डेटा से स्वतः $[0.99, -0.99]$ को पुनः प्राप्त कर लेता है।

=== DECISIONS ===
- Register: inclusive first-person plural "हम" in exposition and polite imperative ("-इए", e.g., विचार कीजिए, रीशेप कीजिए) for reader instructions.
- Punctuation & numerals: Devanagari purna viram (`।`) at the end of complete sentences; Western/Arabic numerals (`0–9`) throughout.
- First-use English in parentheses: include English in parentheses on first defining introduction of key terms (e.g., *एज मैप* (edge map), *द्वि-आयामी क्रॉस-सहसंबंध* (two-dimensional cross-correlation), *परिमित अंतर* (finite difference)); keep standard Latin abbreviations/identifiers (`CNN`, `SGD`, `MLP`, `im2col`, `Fashion-MNIST`) as-is.
- "edge" → "किनारा", "edge detection" → "किनारा पहचान", "edge detector" → "किनारा संसूचक", "edge map" → "एज मैप" (parallel to glossary "फ़ीचर मैप").
- "locality" → "स्थानीयता".
- "window" → "विंडो", "patch" → "पैच".
- "dense layer" → "डेंस लेयर" (parallel to glossary "कन्वोल्यूशनल लेयर" and "फ़ुली कनेक्टेड लेयर").
- "elementwise" → "तत्व-वार".
- "parameter" → "पैरामीटर", "resolution" → "रिज़ॉल्यूशन".
- "reshape" → "रीशेप" (standard Hindi ML usage, matching glossary "शेप").

=== NOTES ===
- 4741e012: math tokens reordered for natural Hindi SOV/postpositional word order ("लंबाई $n$ में चौड़ाई-$k$ की विंडो").
- 12d81c14: math tokens reordered for natural Hindi postpositional phrasing ("$2 \times 2$ कर्नेल के साथ $3 \times 3$ इनपुट से...").
- e810b3a9: code tokens reordered for natural Hindi clause structure ("`corr2d(X, K)` के माध्यम से ... `X` ... `K`").
- 61cacfaf: code tokens reordered for natural Hindi SOV order ("`(X, Y)` से कर्नेल `[1, -1]` को पुनः प्राप्त").
- 05fceed0: code tokens reordered because Hindi participial modifier precedes the head noun ("(`0`) को घेरते सफ़ेद बाहरी स्तंभ (`1`)").
- 0a4b456b: code tokens reordered for natural Hindi adverbial-before-object order ("(`X[:, 2:6] = 0`) रखकर ... `X` बनाते हैं").
- a569780c: math tokens reordered for natural Hindi SOV order ("$\eta = 0.03$ पर $\mathcal{L} = ...$ को न्यूनतम कीजिए").
