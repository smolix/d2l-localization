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
:numref:`sec_why-conv` में हमने दो ज्यामितीय सिद्धांतों, स्थानांतरण समचरता और स्थानीयता, से कन्वोल्यूशनल लेयर का सूत्र कागज़ पर निकाला था। अब हम देखेंगे कि यह सूत्र किसी छवि पर वास्तव में क्या गणना करता है और क्या ग्रेडिएंट डिसेंट डेटा से अपने आप उपयोगी कर्नेल खोज सकता है।

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
मान लें कि हमें उन ऊर्ध्वाधर किनारों का पता लगाना है जहाँ कोई उजला क्षेत्र किसी गहरे क्षेत्र से मिलता है। हर पिक्सेल के मान से उसके ठीक दाईं ओर के पिक्सेल का मान घटाने पर ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) एकसमान क्षेत्रों के भीतर शून्य मिलता है, जहाँ सफ़ेद से काले में बदलाव होता है वहाँ $+1$, और जहाँ काले से फिर सफ़ेद में बदलाव होता है वहाँ $-1$। दो भार वाले संसूचक $(1, -1)$ को छवि के हर मान्य स्थान पर लगाने से एक *किनारा मानचित्र* मिलता है। इस खंड में हम द्विविमीय क्रॉस-सहसंबंध लागू करेंगे, हाथ से बनाए अपने किनारा संसूचक को परखेंगे और ग्रेडिएंट डिसेंट से सीधे डेटा के आधार पर $(1, -1)$ *सीखेंगे*। हम यह भी दिखाएँगे कि कन्वोल्यूशन को एक ही आव्यूह गुणन (*im2col*) के रूप में कैसे लिख सकते हैं और लेयरों को एक के ऊपर एक रखने से किसी इकाई का ग्रहणशील क्षेत्र कैसे बढ़ता है।

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
फ़िलहाल चैनलों को अलग रखकर मान लें कि इनपुट $\mathbf{X}$ एक $n_\textrm{h} \times n_\textrm{w}$ आव्यूह है और कर्नेल $\mathbf{K}$ एक $k_\textrm{h} \times k_\textrm{w}$ आव्यूह है। कर्नेल $k_\textrm{h} \times k_\textrm{w}$ की शेप वाली एक सरकती *विंडो* तय करता है। विंडो का ऊपरी बायाँ कोना $(i, j)$ पर रखकर हम उसके नीचे आने वाले इनपुट पैच के हर अवयव को $\mathbf{K}$ के संगत अवयव से गुणा करते हैं। इन गुणनफलों का योग $(i, j)$ पर आउटपुट की प्रविष्टि देता है।

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
यह $\mathbf{X}$ का $\mathbf{K}$ के साथ *द्विविमीय क्रॉस-सहसंबंध* है। इसे $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ लिखते हैं और कोड की तरह सूचकांक शून्य से शुरू करते हैं। :numref:`sec_why-conv` में ऑफ़सेट $-\Delta$ से $\Delta$ तक केंद्र के दोनों ओर सममित थे। ऊपरी बाएँ कोने को आधार बनाने से केवल $\Delta$ का एक नियत खिसकाव आता है, और सम आकार वाले कर्नेल भी आसानी से इस्तेमाल किए जा सकते हैं (जैसे $2 \times 2$ का कर्नेल या हमारा $1 \times 2$ का किनारा संसूचक), जिनमें कोई केंद्रीय पिक्सेल नहीं होता।

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![द्विविमीय क्रॉस-सहसंबंध। छायांकित हिस्सों में आउटपुट का पहला अवयव और उसकी गणना में इस्तेमाल हुए इनपुट तथा कर्नेल के अवयव दिखाए गए हैं। गणना है $0\times0+1\times1+3\times2+4\times3=19$।](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` में हम $2 \times 2$ के कर्नेल को $3 \times 3$ के इनपुट पर एक बार में एक पिक्सेल सरकाते हैं। हम बाएँ से दाएँ और ऊपर से नीचे बढ़ते हैं। विंडो चार स्थानों पर पूरी तरह समाती है, जिससे $2 \times 2$ का आउटपुट मिलता है।

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
ध्यान दें कि आउटपुट हर अक्ष पर थोड़ा छोटा हो जाता है, क्योंकि कर्नेल पूरी तरह इनपुट के भीतर रहना चाहिए। $k$ चौड़ी विंडो $n$ लंबाई में $n - k + 1$ मान्य स्थानों पर समाती है। इसलिए आउटपुट की शेप इस प्रकार है।

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
इस तरह $3 \times 3$ के इनपुट पर $2 \times 2$ का कर्नेल लगाने से $2 \times 2$ का आउटपुट मिलता है, $6 \times 8$ की छवि पर $1 \times 2$ का कर्नेल लगाने से $6 \times 7$ का, और $28 \times 28$ की Fashion-MNIST छवि पर $5 \times 5$ का कर्नेल लगाने से $24 \times 24$ का। जब भी हमें स्थानिक आयाम बनाए रखने होते हैं, हम इनपुट के चारों ओर शून्यों की सीमा लगाते हैं (*पैडिंग*)। इसे हम स्ट्राइड और डाइलेशन के साथ :numref:`sec_padding` में देखेंगे।

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### कार्यान्वयन

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
हम :eqref:`eq_corr2d` को सीधे `corr2d` में लागू करते हैं। दो लूप मिलकर आउटपुट के हर निर्देशांक पर जाते हैं।

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` के परिणाम को :numref:`fig_correlation` में हाथ से की गई गणना से मिलाने के लिए हम $3 \times 3$ का इनपुट टेंसर `X` और $2 \times 2$ का कर्नेल `K` बनाते हैं, फिर `corr2d(X, K)` की गणना करते हैं।

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
आउटपुट की हर प्रविष्टि :numref:`fig_correlation` से मेल खाती है। (और जैसा कि :numref:`sec_why-conv` में दिखाया गया है, पलटे हुए कर्नेल $\mathbf{K}'$ के साथ गणितीय परिभाषा वाला कन्वोल्यूशन ठीक उसी परिवार की लेयरें सीखता है। इसलिए हम डीप लर्निंग की प्रचलित शब्दावली के अनुसार क्रॉस-सहसंबंध को “कन्वोल्यूशन” कहते हैं।)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## कन्वोल्यूशनल लेयरें

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### लेयर की संरचना

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
एक द्विविमीय कन्वोल्यूशनल लेयर अपने इनपुट और सीखे गए कर्नेल का क्रॉस-सहसंबंध निकालती है और उसमें एक अदिश बायस जोड़ती है। :numref:`chap_perceptrons` में MLP के भारों की तरह हम कर्नेल के शुरुआती मान $\mathcal{N}(0, 0.01^2)$ से लेते हैं और बायस का शुरुआती मान शून्य रखते हैं। नीचे `Conv2D` दोनों पैरामीटर पंजीकृत करता है और अपने फ़ॉरवर्ड पास में `corr2d` को कॉल करता है। जिस लेयर के कर्नेल की ऊँचाई $h$ और चौड़ाई $w$ हो, उसे $h \times w$ की कन्वोल्यूशनल लेयर कहते हैं।

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
सघन लेयर के विपरीत, कन्वोल्यूशनल लेयर किसी नियत इनपुट रिज़ॉल्यूशन से बँधी नहीं होती, क्योंकि उसके पैरामीटर केवल कर्नेल के आकार पर निर्भर करते हैं। ठीक वही $2 \times 2$ की लेयर (कुल 5 पैरामीटर) हमारे $3 \times 3$ के टेंसर `X` और $6 \times 8$ की छवि, दोनों को संसाधित करती है। इनके आउटपुट की शेप क्रमशः $2 \times 2$ और $5 \times 7$ होती है।

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### किनारों का पता लगाना

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
अब हम परिचय में बताए गए ऊर्ध्वाधर किनारा संसूचक को $6 \times 8$ की एक कृत्रिम छवि पर परखते हैं। इसके बीच के चार स्तंभ काले (`0`) और बाहरी स्तंभ सफ़ेद (`1`) हैं।

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
हम $1 \times 2$ के कर्नेल को इस प्रकार परिभाषित करते हैं।

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
हर स्थान $(i, j)$ पर यह क्षैतिज *परिमित अंतर* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ निकालता है। चिह्न का अंतर छोड़ दें तो यह चमक के क्षैतिज अवकलज का एक विविक्त सन्निकटन है। यह एकसमान क्षेत्रों में शून्य होता है और जहाँ तीव्रता अचानक बदलती है वहाँ इसका परिमाण उछलता है।

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ की पट्टी वाली छवि `X` और $1 \times 2$ के क्षैतिज अंतर वाले कर्नेल `K` का क्रॉस-सहसंबंध `corr2d(X, K)` से निकालने पर $6 \times 7$ का किनारा मानचित्र `Y` मिलता है।

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
जैसी उम्मीद थी, `Y` में सफ़ेद से काले की सीमा पर `+1`, काले से सफ़ेद की सीमा पर `-1` और बाकी सभी जगह `0` मिलता है। ठीक उसी कर्नेल को पंक्तियों और स्तंभों की अदला-बदली से बनी छवि पर लगाने से (जिसमें ऊर्ध्वाधर पट्टी क्षैतिज हो जाती है) सभी मान शून्य मिलते हैं।

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
ज़ाहिर है कि `[1, -1]` केवल ऊर्ध्वाधर किनारों पर प्रतिक्रिया देता है। क्षैतिज किनारों, तिरछी रेखाओं, कोनों या बनावटों का पता लगाने के लिए अलग-अलग कर्नेल चाहिए।

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
जब हमें ठीक-ठीक पता हो कि कौन-सा फ़ीचर चाहिए, तब `[1, -1]` को हाथ से बनाना आसान है। लेकिन कोई इंजीनियर किसी गहरे नेटवर्क की बीच की लेयरों के हज़ारों फ़िल्टर हाथ से तय नहीं कर सकता। क्या ग्रेडिएंट डिसेंट कर्नेल `[1, -1]` को केवल इनपुट–आउटपुट युग्म `(X, Y)` से खोज सकता है? इसे परखने के लिए हम हर फ़्रेमवर्क की अंतर्निर्मित द्विविमीय कन्वोल्यूशनल लेयर इस्तेमाल करते हैं (बायस हटाकर और भारों के शुरुआती मान $\mathcal{N}(0, 0.01^2)$ से लेकर)। हम सभी 42 आउटपुट प्रविष्टियों की वर्ग त्रुटियों के योग को $0.03$ के लर्निंग रेट पर ग्रेडिएंट के 10 चरणों में न्यूनतम करते हैं। अंतर्निर्मित लेयरें 4D टेंसर लेती हैं, जिनमें बैच और चैनल के अक्ष भी होते हैं (:numref:`sec_channels`)। इसलिए हम `X` और `Y` की शेप उसी के अनुसार बदलते हैं।

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
शुरू में लॉस 12 के करीब है (लगभग शून्य मानों वाला यादृच्छिक कर्नेल सभी 12 ऐसे किनारा पिक्सेलों पर करीब 1 से चूकता है जिनका मान $\pm 1$ है)। केवल नौ चरणों में लॉस 0.01 से नीचे आ जाता है। सीखे गए कर्नेल को देखने से पता चलता है कि उसे क्या मिला।

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
दोनों प्रविष्टियाँ `[1, -1]` के अपने-अपने मानों से लगभग 0.02 के भीतर हैं। इस लूप को किनारों की कोई जानकारी नहीं थी। लॉस के संकेत से ग्रेडिएंट डिसेंट ने अपने आप फ़िल्टर खोज लिया। यहाँ हमने सीधे `Y` को लक्ष्य बनाकर एक ही लेयर को प्रशिक्षित किया। किसी गहरे CNN में बैकप्रॉपैगेशन केवल अंतिम कार्य के लॉस से शुरू से अंत तक सभी लेयरों के कर्नेल सीखता है।

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
छवियों के लिए कन्वोल्यूशन<br>**द्विविमीय क्रॉस-सहसंबंध · किनारों का पता लगाना · सीखे गए कर्नेल · im2col · ग्रहणशील क्षेत्र**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="छवि में स्थानीय किनारों का पता लगाना"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[प्रेरणा और मूल अवधारणा]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ की कृत्रिम परीक्षण छवि।** बाहरी स्तंभ सफ़ेद (`1`) हैं और उनके बीच 4 स्तंभों की काली ऊर्ध्वाधर पट्टी (`0`) है।

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **क्षैतिज परिमित अंतर।** $\mathbf{K} = [1, -1]$ को सरकाने पर $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ मिलता है, यानी सफ़ेद से काले के किनारे पर `+1` और काले से सफ़ेद के किनारे पर `-1`।

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **सीखना और हाथ से बनाना।** गहरे CNN ग्रेडिएंट डिसेंट से ऐसे हज़ारों कर्नेल अपने आप खोजते हैं।

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="द्विविमीय क्रॉस-सहसंबंध की संक्रिया"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[सरकती विंडो के आंतरिक गुणनफल]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ के इनपुट आव्यूह $\mathbf{X}$ और $k_\textrm{h} \times k_\textrm{w}$ के कर्नेल $\mathbf{K}$ के लिए द्विविमीय क्रॉस-सहसंबंध $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ विंडो का ऊपरी बायाँ कोना $(i, j)$ पर रखकर संगत अवयवों के गुणनफलों का योग निकालता है।

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **आधार के रूप में ऊपरी बायाँ कोना।** शून्य से शुरू होने वाले सूचकांक $a, b \ge 0$ कोड में ऐरे के सूचकांकों से मेल खाते हैं और सम आकार वाले कर्नेल (जैसे $2 \times 2$ या $1 \times 2$) पर भी आसानी से लागू होते हैं।

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **ऊपरी बाएँ कोने का छायांकित आउटपुट $(0,0)$।**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$।

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="विंडो के मान्य स्थान और आउटपुट की स्थानिक शेप"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[पैडिंग के बिना कन्वोल्यूशन में ग्रिड क्यों सिकुड़ता है]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **विंडो के मान्य स्थान।** $k_\textrm{h} \times k_\textrm{w}$ की विंडो $n_\textrm{h} \times n_\textrm{w}$ के ग्रिड में हर अक्ष पर $n - k + 1$ स्थानों पर समाती है।

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **शेप के उदाहरण।** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) और $28 \times 28 \star 5 \times 5 \to 24 \times 24$।

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="द्विविमीय क्रॉस-सहसंबंध का कोड"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[एक के भीतर दूसरे लूप से सीधा कार्यान्वयन]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
फ़ंक्शन `corr2d` आउटपुट के लिए $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ का टेंसर आवंटित करता है। हर प्रविष्टि `X[i:i+h, j:j+w] * K` के अवयवों का योग है।

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
हर चरण में $k_\textrm{h} \times k_\textrm{w}$ की उपविंडो का स्लाइस `X` से लेते हैं, उसके अवयवों को `K` के संगत अवयवों से गुणा करते हैं और `sum` से योग निकालते हैं।

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d और पलटे हुए कर्नेल की तुल्यता की जाँच"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[क्रॉस-सहसंबंध और गणितीय कन्वोल्यूशन]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° घुमाया गया कर्नेल $\mathbf{K}'$।** $\mathbf{K}$ के साथ क्रॉस-सहसंबंध, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ के साथ कन्वोल्यूशन के बराबर है।

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **सीखी जा सकने वाली लेयरों का वही वर्ग।** फ़्रेमवर्क क्रॉस-सहसंबंध की गणना करते हैं और उसे कन्वोल्यूशन कहते हैं।

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="द्विविमीय कन्वोल्यूशनल लेयर का मॉड्यूल"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[सीखे जा सकने वाले कर्नेल भार और अदिश बायस]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
कन्वोल्यूशनल लेयर `corr2d` के साथ $k_\textrm{h} \times k_\textrm{w}$ का सीखने योग्य भार पैरामीटर (शुरुआती मान $\mathcal{N}(0, 0.01^2)$ से) और एक अदिश बायस (शुरुआती मान $0$) इस्तेमाल करती है।

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="इनपुट रिज़ॉल्यूशन से कन्वोल्यूशनल लेयरों की स्वतंत्रता"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[किसी भी इनपुट आकार के लिए वही पैरामीटर]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ से स्वतंत्र।** ठीक वही 5 पैरामीटर ($2 \times 2$ का कर्नेल + 1 बायस) भारों की शेप बदले बिना $3 \times 3$, $6 \times 8$ या मेगापिक्सेल वाले इनपुट संसाधित करते हैं।

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="ऊर्ध्वाधर किनारों वाली कृत्रिम छवि"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 का बाइनरी परीक्षण पैटर्न]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
हम $6 \times 8$ की परीक्षण छवि `X` बनाते हैं। बाईं और दाईं सीमाओं पर मान 1 और बीच के चार स्तंभों में 0 रखते हैं (`X[:, 2:6] = 0`)।

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **स्तंभ-समूहों में स्थिर तीव्रता।** तीव्रता का मान (`1` या `0`) हर समूह के भीतर स्थिर है, चाहे वह $0\text{--}1$, $2\text{--}5$ या $6\text{--}7$ हो। यह केवल स्तंभ $1 \to 2$ और $5 \to 6$ पर बदलती है।

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="क्षैतिज परिमित अंतर वाले कर्नेल का प्रयोग"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] से ऊर्ध्वाधर किनारों का पता लगाना]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` और $1 \times 2$ के कर्नेल $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ का क्रॉस-सहसंबंध हर क्षैतिज युग्म पर $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ निकालता है।

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **चिह्न सहित किनारा मानचित्र।** स्तंभ 1 में `+1` (सफ़ेद से काला), स्तंभ 5 में `-1` (काले से सफ़ेद) और एकसमान हिस्सों के सभी भीतरी स्तंभों में `0` मिलता है।

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="विशिष्ट दिशाओं के लिए किनारा कर्नेल"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[एक कर्नेल हर दिशा के किनारे क्यों नहीं पकड़ सकता]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **क्षैतिज किनारों पर कोई प्रतिक्रिया नहीं।** `X` के पंक्ति-स्तंभ बदलने पर (`X.T`) क्षैतिज पड़ोसियों के मान समान हो जाते हैं, इसलिए $[1, -1]$ के सभी आउटपुट शून्य होते हैं।

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **फ़िल्टर समूह (§{sec_channels})।** कंप्यूटर विज़न की लेयरें क्षैतिज, ऊर्ध्वाधर और तिरछी दिशाओं के लिए समानांतर चैनल सीखती हैं।

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="ग्रेडिएंट डिसेंट से किनारा संसूचक सीखना"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) से अंतर्निर्मित कन्वोल्यूशनल लेयर का पर्यवेक्षित प्रशिक्षण]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **फ़्रेमवर्क के 4D टेंसर।** `X` और `Y` को 4D टेंसर की शेप दें (एक उदाहरण, एक चैनल) और $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ को $\eta = 0.03$ पर न्यूनतम करें।

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="सीखे गए कर्नेल भारों की जाँच"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[शुरुआती लॉस 12 क्यों है और भार [1, -1] की ओर क्यों बढ़ते हैं]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **शुरुआती लॉस $12.0$ क्यों है।** लगभग शून्य भार $\hat{Y} \approx 0$ का अनुमान देते हैं, जिससे $\pm 1$ मान वाले बारह किनारा पिक्सेलों पर `Y` से चूक होती है ($12 \times 1^2 = 12$)।

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 चरणों में अभिसरण।** ग्रेडिएंट डिसेंट डेटा से अपने आप $[0.99, -0.99]$ खोज लेता है।

=== DECISIONS ===
- Use collaborative “हम” in exposition and polite imperatives in exercises. Use Devanagari prose, । for sentence endings, and 0–9 digits for prose numerals. Keep names, acronyms, and identifiers in Latin script. Add English parentheticals only where the source gives them.
- “locality” → “स्थानीयता”; “window” → “विंडो”; “patch” → “पैच”; “edge map” → “किनारा मानचित्र”; “edge detector” → “किनारा संसूचक”.
- “input/output” → “इनपुट/आउटपुट”; “parameter” → “पैरामीटर”; “dense” → “सघन”; “array” → “ऐरे”; “unit” (network element) → “इकाई”; “two-dimensional/2D” → “द्विविमीय”; “row/column” → “पंक्ति/स्तंभ”; “index” → “सूचकांक”; “entry/element” → “प्रविष्टि/अवयव”.
- “vertical/horizontal” → “ऊर्ध्वाधर/क्षैतिज”; “intensity” → “तीव्रता”; “offset” → “ऑफ़सेट”; “framework” → “फ़्रेमवर्क”; “built-in” → “अंतर्निर्मित”; “deep learning” → “डीप लर्निंग”; “stack” (layers, verb) → “एक के ऊपर एक रखना”.
- “forward propagation” → “फ़ॉरवर्ड पास”, following the glossary's shared entry.

=== NOTES ===
- 55f0a41b: “returns” describes a change back to white, rather than the RL glossary sense.
- 29cd9061: “returning” describes the layer producing outputs, rather than the RL glossary sense.
- 7e4ef44b: The weights approach [1, -1]; the loss starts near 12 and decreases toward zero.
- 456670c5: The loop uses the full input–output pair and sums all 42 errors at every step, so the slide's “SGD” is rendered as “ग्रेडिएंट डिसेंट”.
