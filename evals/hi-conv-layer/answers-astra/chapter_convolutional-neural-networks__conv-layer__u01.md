<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=hi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# छवियों पर कन्वोल्यूशन

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` में हमने दो ज्यामितीय सिद्धांतों, स्थानांतरण समचरता और स्थानीयता, से कन्वोल्यूशनल लेयर का सूत्र निकाला था। अब देखते हैं कि किसी छवि पर यह सूत्र वास्तव में क्या गणना करता है और क्या ग्रेडिएंट डिसेंट अपने आप डेटा से उपयोगी कर्नेल खोज सकता है।

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
मान लें कि हमें वे खड़े किनारे ढूँढ़ने हैं जहाँ कोई उजला क्षेत्र किसी गहरे क्षेत्र से मिलता है। हर पिक्सेल के मान में से उसके दाएँ पड़ोसी का मान घटाएँ ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), तो एकसमान क्षेत्रों के भीतर शून्य मिलता है, सफ़ेद से काला होने पर $+1$ और काले से फिर सफ़ेद होने पर $-1$। दो भारों वाले संसूचक $(1, -1)$ को छवि में हर मान्य स्थान पर खिसकाने से *किनारा मैप* बनता है। इस खंड में हम द्वि-आयामी क्रॉस-सहसंबंध लागू करेंगे, अपने बनाए किनारा संसूचक को परखेंगे और देखेंगे कि ग्रेडिएंट डिसेंट सीधे डेटा से $(1, -1)$ कैसे *सीखता* है। फिर हम दिखाएँगे कि एक ही आव्यूह गुणन (*im2col*) से कन्वोल्यूशन कैसे होता है और गणना करेंगे कि एक के बाद एक लेयर लगाने से किसी इकाई का ग्रहणशील क्षेत्र कितना चौड़ा होता है।

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## क्रॉस-सहसंबंध

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### गणना की विधि

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
अभी चैनल को अलग रखें। मान लें कि इनपुट $\mathbf{X}$ एक $n_\textrm{h} \times n_\textrm{w}$ आव्यूह है और कर्नेल $\mathbf{K}$ एक $k_\textrm{h} \times k_\textrm{w}$ आव्यूह है। कर्नेल से एक खिसकती हुई *विंडो* तय होती है जिसकी शेप $k_\textrm{h} \times k_\textrm{w}$ है। विंडो का ऊपरी बायाँ कोना $(i, j)$ पर रखें। विंडो के भीतर आने वाले इनपुट पैच के हर तत्व को $\mathbf{K}$ के संगत तत्व से गुणा करें। इन गुणनफलों को जोड़ने पर $(i, j)$ की आउटपुट प्रविष्टि मिलती है।

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
यह $\mathbf{X}$ और $\mathbf{K}$ का *द्वि-आयामी क्रॉस-सहसंबंध* है। इसे हम $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ लिखते हैं और कोड की तरह सूचकांक शून्य से शुरू करते हैं। :numref:`sec_why-conv` में हमने केंद्र के दोनों ओर $-\Delta$ से $\Delta$ तक सममित ऑफ़सेट लिए थे। ऊपरी बाएँ कोने को आधार बनाने से केवल $\Delta$ का एक निश्चित खिसकाव आता है। इससे सम आकार वाले कर्नेल भी आसानी से संभल जाते हैं, जैसे $2 \times 2$ का कर्नेल या हमारा $1 \times 2$ का किनारा संसूचक, जिनमें कोई एक केंद्रीय पिक्सेल नहीं होता।

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![द्वि-आयामी क्रॉस-सहसंबंध। छायांकित हिस्से पहला आउटपुट तत्व और उसकी गणना में प्रयुक्त इनपुट तथा कर्नेल के तत्व दिखाते हैं। गणना है $0\times0+1\times1+3\times2+4\times3=19$।](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` में हम $2 \times 2$ के कर्नेल को $3 \times 3$ के इनपुट पर एक बार में एक पिक्सेल खिसकाते हैं, बाएँ से दाएँ और ऊपर से नीचे। विंडो चार स्थानों पर पूरी तरह समाती है, जिससे $2 \times 2$ का आउटपुट मिलता है।

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
ध्यान दें कि हर अक्ष पर आउटपुट थोड़ा छोटा हो जाता है, क्योंकि पूरा कर्नेल इनपुट के भीतर रहना चाहिए। $k$ की चौड़ाई वाली विंडो को $n$ की लंबाई में रखने के लिए $n - k + 1$ मान्य स्थान मिलते हैं। इसलिए आउटपुट की शेप है

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
इस तरह $3 \times 3$ के इनपुट पर $2 \times 2$ का कर्नेल लगाने से $2 \times 2$ का आउटपुट मिलता है। $6 \times 8$ की छवि पर $1 \times 2$ का कर्नेल लगाने से $6 \times 7$ और $28 \times 28$ की Fashion-MNIST छवि पर $5 \times 5$ का कर्नेल लगाने से $24 \times 24$ का आउटपुट मिलता है। जब हमें स्थानिक आयाम बनाए रखने होते हैं, तब हम इनपुट के चारों ओर शून्यों की पट्टी लगाते हैं (*पैडिंग*)। :numref:`sec_padding` में हम स्ट्राइड और डाइलेशन के साथ पैडिंग का भी अध्ययन करेंगे।

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### कार्यान्वयन

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
हम :eqref:`eq_corr2d` को सीधे `corr2d` में लागू करते हैं। दो लूप की मदद से हम हर आउटपुट निर्देशांक पर गणना करते हैं।

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` का परिणाम :numref:`fig_correlation` की हाथ से की गई गणना से मिलाने के लिए हम $3 \times 3$ का इनपुट टेंसर `X` और $2 \times 2$ का कर्नेल `K` बनाते हैं और `corr2d(X, K)` का मान निकालते हैं।

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
आउटपुट की हर प्रविष्टि :numref:`fig_correlation` से मेल खाती है। (जैसा कि हमने :numref:`sec_why-conv` में देखा, उलटे कर्नेल $\mathbf{K}'$ के साथ गणितीय परिभाषा वाला कन्वोल्यूशन भी ठीक वही लेयर सीख सकता है। इसलिए डीप लर्निंग की प्रचलित शब्दावली के अनुसार हम क्रॉस-सहसंबंध को “कन्वोल्यूशन” कहते हैं।)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## कन्वोल्यूशनल लेयर

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### लेयर की बनावट

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
द्वि-आयामी कन्वोल्यूशनल लेयर अपने इनपुट और सीखे हुए कर्नेल का क्रॉस-सहसंबंध निकालती है और उसमें एक अदिश बायस जोड़ती है। :numref:`chap_perceptrons` में MLP के भारों की तरह हम कर्नेल के आरंभिक भार $\mathcal{N}(0, 0.01^2)$ से लेते हैं और बायस का आरंभिक मान शून्य रखते हैं। नीचे `Conv2D` दोनों पैरामीटर दर्ज करता है और अपने फ़ॉरवर्ड पास में `corr2d` को बुलाता है। जिस लेयर के कर्नेल की ऊँचाई $h$ और चौड़ाई $w$ हो, उसे $h \times w$ की कन्वोल्यूशनल लेयर कहते हैं।

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
डेंस लेयर के विपरीत, कन्वोल्यूशनल लेयर किसी निश्चित इनपुट रिज़ॉल्यूशन तक सीमित नहीं होती, क्योंकि उसके पैरामीटर केवल कर्नेल के आकार पर निर्भर करते हैं। वही $2 \times 2$ की लेयर (कुल 5 पैरामीटर) हमारे $3 \times 3$ के टेंसर `X` और $6 \times 8$ की छवि, दोनों को संसाधित करती है। आउटपुट की शेप क्रमशः $2 \times 2$ और $5 \times 7$ है।

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### किनारों की पहचान

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
अब हम परिचय में दिए गए खड़े किनारों के संसूचक को $6 \times 8$ की एक कृत्रिम छवि पर परखते हैं। इसके बीच के चार स्तंभ काले (`0`) और बाहरी स्तंभ सफ़ेद (`1`) हैं।

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
हम $1 \times 2$ का कर्नेल इस प्रकार परिभाषित करते हैं।

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
हर स्थान $(i, j)$ पर यह क्षैतिज *परिमित अंतर* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ निकालता है। चिह्न को छोड़ दें तो यह चमक के क्षैतिज अवकलज का एक विविक्त सन्निकटन है। एकसमान चमक वाले क्षेत्रों में यह शून्य होता है और जहाँ तीव्रता अचानक बदलती है वहाँ इसका परिमाण तेज़ी से बढ़ता है।

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ की पट्टी वाली छवि `X` और क्षैतिज अंतर निकालने वाले $1 \times 2$ के कर्नेल `K` का क्रॉस-सहसंबंध हम `corr2d(X, K)` से निकालते हैं। इससे $6 \times 7$ का किनारा मैप `Y` मिलता है।

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
जैसी अपेक्षा थी, `Y` का मान सफ़ेद से काले की सीमा पर `+1`, काले से सफ़ेद की सीमा पर `-1` और बाकी सभी जगह `0` है। इसी कर्नेल को छवि का ट्रांसपोज़ लेने के बाद लगाएँ (जिससे खड़ी पट्टी आड़ी हो जाती है), तो हर जगह शून्य मिलता है।

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
स्पष्ट है कि `[1, -1]` केवल खड़े किनारों पर प्रतिक्रिया देता है। आड़े या तिरछे किनारे, कोने या बनावट पहचानने के लिए अलग कर्नेल चाहिए।

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
जब हमें ठीक-ठीक पता हो कि कौन-सा फ़ीचर चाहिए, तब `[1, -1]` जैसा कर्नेल हाथ से बनाना आसान है। फिर भी कोई इंजीनियर किसी गहरे नेटवर्क की बीच की लेयरों के हज़ारों फ़िल्टर हाथ से तय नहीं कर सकता। क्या ग्रेडिएंट डिसेंट कर्नेल `[1, -1]` को केवल इनपुट–आउटपुट युग्म `(X, Y)` से खोज सकता है? इसे परखने के लिए हम हर फ़्रेमवर्क की अपनी 2D कन्वोल्यूशनल लेयर लेते हैं। हम बायस हटा देते हैं और भारों के आरंभिक मान $\mathcal{N}(0, 0.01^2)$ से लेते हैं। सभी 42 आउटपुट प्रविष्टियों की वर्ग त्रुटि का योग न्यूनतम करने के लिए हम $0.03$ की लर्निंग रेट से ग्रेडिएंट के 10 कदम लेते हैं। फ़्रेमवर्क की लेयर को 4D टेंसर चाहिए जिसमें बैच और चैनल के अक्ष भी हों (:numref:`sec_channels`), इसलिए हम `X` और `Y` की शेप उसी के अनुसार बदलते हैं।

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
शुरू में लॉस लगभग 12 है (यादृच्छिक कर्नेल के भार लगभग शून्य हैं, इसलिए किनारे के सभी 12 पिक्सेल के $\pm 1$ मानों के अनुमान में लगभग 1 की त्रुटि है)। सिर्फ़ नौ कदमों में लॉस 0.01 से नीचे आ जाता है। सीखे हुए कर्नेल को देखकर पुष्टि होती है कि उसे क्या मिला।

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
दोनों प्रविष्टियाँ `[1, -1]` के संगत मानों के क़रीब हैं। दोनों में अंतर लगभग 0.02 या उससे कम है। इस लूप में कहीं भी किनारों की जानकारी नहीं दी गई थी। लॉस का संकेत पाकर ग्रेडिएंट डिसेंट ने ख़ुद फ़िल्टर खोज लिया। यहाँ हमने सीधे `Y` को लक्ष्य बनाकर इस एक लेयर को प्रशिक्षित किया। गहरे CNN में केवल अंतिम कार्य के लॉस से बैकप्रॉपैगेशन के ज़रिए सभी लेयरों के कर्नेल शुरू से अंत तक एक साथ सीखे जाते हैं।

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
छवियों पर कन्वोल्यूशन<br>**2D क्रॉस-सहसंबंध · किनारों की पहचान · सीखे हुए कर्नेल · im2col · ग्रहणशील क्षेत्र**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="पूरी छवि में स्थानीय किनारों की पहचान"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[प्रेरणा और मूल विचार]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ की कृत्रिम परीक्षण छवि:** सफ़ेद बाहरी स्तंभों (`1`) के बीच 4 स्तंभ चौड़ी काली खड़ी पट्टी (`0`) है।

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **क्षैतिज परिमित अंतर:** $\mathbf{K} = [1, -1]$ को खिसकाने पर हर जगह $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ मिलता है। सफ़ेद से काले किनारे पर मान `+1` और काले से सफ़ेद किनारे पर `-1` है।

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **सीखना और हाथ से बनाना:** गहरे CNN ग्रेडिएंट डिसेंट से ऐसे हज़ारों कर्नेल अपने आप खोज लेते हैं।

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2D क्रॉस-सहसंबंध की गणना"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[खिसकती विंडो के आंतरिक गुणनफल]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ के इनपुट आव्यूह $\mathbf{X}$ और $k_\textrm{h} \times k_\textrm{w}$ के कर्नेल $\mathbf{K}$ का 2D क्रॉस-सहसंबंध $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ निकालने के लिए हम विंडो का ऊपरी बायाँ कोना $(i, j)$ पर रखते हैं और संगत तत्वों के गुणनफल जोड़ते हैं।

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **ऊपरी बायाँ कोना आधार:** शून्य से शुरू होने वाले सूचकांक $a, b \ge 0$ कोड में ऐरे के सूचकांकों से मेल खाते हैं। सम आकार के कर्नेल (जैसे $2 \times 2$ या $1 \times 2$) पर भी ये आसानी से लागू होते हैं।

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **ऊपरी बायाँ छायांकित आउटपुट $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$।

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="विंडो के मान्य स्थान और आउटपुट की स्थानिक शेप"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[पैडिंग के बिना कन्वोल्यूशन से ग्रिड क्यों सिकुड़ता है?]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **विंडो के मान्य स्थान:** $k_\textrm{h} \times k_\textrm{w}$ की विंडो $n_\textrm{h} \times n_\textrm{w}$ के ग्रिड में हर अक्ष पर $n - k + 1$ स्थानों पर पूरी तरह समाती है।

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **शेप के उदाहरण:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) और $28 \times 28 \star 5 \times 5 \to 24 \times 24$।

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="कोड में 2D क्रॉस-सहसंबंध"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[एक लूप के भीतर दूसरे लूप से सीधी गणना]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
फ़ंक्शन `corr2d` पहले $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ के आउटपुट टेंसर के लिए मेमोरी आवंटित करता है। फिर हर प्रविष्टि के लिए `X[i:i+h, j:j+w] * K` के तत्वों का योग निकालता है।

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
हर चक्र में $k_\textrm{h} \times k_\textrm{w}$ की एक छोटी विंडो `X` से स्लाइस की जाती है। उसके तत्वों को `K` के संगत तत्वों से गुणा करके `sum` से जोड़ा जाता है।

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d और उलटे कर्नेल की समतुल्यता की जाँच"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[क्रॉस-सहसंबंध और गणितीय कन्वोल्यूशन]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° घुमाया हुआ कर्नेल $\mathbf{K}'$:** $\mathbf{K}$ से क्रॉस-सहसंबंध करने पर वही परिणाम मिलता है जो $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ से कन्वोल्यूशन करने पर मिलता है।

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **सीखी जा सकने वाली लेयरों का एक ही वर्ग:** फ़्रेमवर्क क्रॉस-सहसंबंध की गणना करते हैं और उसे कन्वोल्यूशन कहते हैं।

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2D कन्वोल्यूशनल लेयर का मॉड्यूल बनाना"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[सीखने योग्य कर्नेल भार और अदिश बायस]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
कन्वोल्यूशनल लेयर `corr2d` के साथ $k_\textrm{h} \times k_\textrm{w}$ का सीखने योग्य भार पैरामीटर (आरंभिक मान $\mathcal{N}(0, 0.01^2)$ से) और एक अदिश बायस (आरंभिक मान $0$) इस्तेमाल करती है।

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="इनपुट रिज़ॉल्यूशन से स्वतंत्र कन्वोल्यूशनल लेयर"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[अलग-अलग इनपुट आकारों के लिए एक ही पैरामीटर समूह]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ से स्वतंत्र:** वही 5 पैरामीटर ($2 \times 2$ का कर्नेल और 1 बायस) भारों की शेप बदले बिना $3 \times 3$, $6 \times 8$ या मेगापिक्सेल आकार के इनपुट संसाधित करते हैं।

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="खड़े किनारों वाली कृत्रिम छवि बनाना"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 का बाइनरी परीक्षण पैटर्न]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
हम $6 \times 8$ की परीक्षण छवि `X` बनाते हैं। बाईं और दाईं सीमा पर एक और बीच के चार स्तंभों में शून्य रखते हैं (`X[:, 2:6] = 0`)।

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **खंडों में एकसमान स्तंभ:** तीव्रता (`1` या `0`) तीनों स्तंभ समूहों $0\text{--}1$, $2\text{--}5$ और $6\text{--}7$ में से हर एक के भीतर स्थिर रहती है। यह केवल $1 \to 2$ और $5 \to 6$ पर बदलती है।

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="क्षैतिज परिमित अंतर का कर्नेल लगाना"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] से खड़े किनारों की पहचान]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` और $1 \times 2$ के कर्नेल $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ का क्रॉस-सहसंबंध लेने से क्षैतिज पड़ोसियों के हर जोड़े के लिए $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ मिलता है।

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **चिह्न सहित किनारा मैप:** स्तंभ 1 में `+1` (सफ़ेद से काला), स्तंभ 5 में `-1` (काले से सफ़ेद) और एकसमान तीव्रता वाले सभी भीतरी स्तंभों में `0` मिलता है।

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="किनारों के कर्नेल की दिशा के प्रति चयनशीलता"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[एक कर्नेल सभी दिशाओं के किनारे क्यों नहीं पहचान सकता?]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **आड़े किनारों पर शून्य प्रतिक्रिया:** `X` का ट्रांसपोज़ (`X.T`) लेने पर क्षैतिज पड़ोसियों के मान एक जैसे हो जाते हैं, इसलिए $[1, -1]$ हर जगह शून्य देता है।

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **फ़िल्टर बैंक (§{sec_channels}):** दृष्टि संबंधी कार्यों में लेयर आड़ी, खड़ी और तिरछी दिशाओं के लिए समानांतर चैनल सीखती हैं।

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="ग्रेडिएंट डिसेंट से किनारा संसूचक सीखना"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) से फ़्रेमवर्क की कन्वोल्यूशनल लेयर का पर्यवेक्षित प्रशिक्षण]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **फ़्रेमवर्क के 4D टेंसर:** `X` और `Y` की शेप बदलकर 4D टेंसर बनाएँ (एक उदाहरण, एक चैनल) और $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ को $\eta = 0.03$ पर न्यूनतम करें।

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="सीखे हुए कर्नेल भारों की जाँच"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[लॉस 12 से क्यों शुरू होता है और कर्नेल [1, -1] की ओर क्यों बढ़ता है?]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **आरंभिक लॉस $12.0$ क्यों है:** लगभग शून्य भारों से अनुमान $\hat{Y} \approx 0$ आता है, इसलिए $\pm 1$ वाले किनारों के बारह पिक्सेल के लिए अनुमान `Y` के मानों से चूकता है ($12 \times 1^2 = 12$)।

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 कदमों में अभिसरण:** SGD अपने आप डेटा से $[0.99, -0.99]$ सीख लेता है।


=== DECISIONS ===
- Use “हम” for the shared derivation and polite imperatives such as “निकालें” in exercises. Use । to end Hindi statements, Arabic digits for numeric notation, and retain Latin identifiers and familiar acronyms without routinely repeating English terms in parentheses.
- image → छवि; edge → किनारा; edge detector → किनारा संसूचक; edge map → किनारा मैप; vertical/horizontal edges → खड़े/आड़े किनारे.
- locality → स्थानीयता; window → विंडो; patch → पैच; parameter → पैरामीटर; dense layer → डेंस लेयर; unit → इकाई.

=== NOTES ===
- 7e4ef44b: The English kicker makes the loss converge to a kernel. The translation assigns the initial value 12 to the loss and convergence toward [1, -1] to the kernel.
