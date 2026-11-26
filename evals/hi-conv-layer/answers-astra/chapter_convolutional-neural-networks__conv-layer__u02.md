<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=hi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## आव्यूह गुणन के रूप में कन्वोल्यूशन

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
`corr2d` में एक लूप के भीतर दूसरा लूप होने से गणित साफ़ दिखता है, पर वास्तविक काम में Python से एक बार में एक आउटपुट पिक्सेल की गणना करना बहुत धीमा पड़ता है। इसका एक पुराना उपाय आज भी लाइब्रेरी में उपलब्ध एल्गोरिदम में शामिल है। हम क्रॉस-सहसंबंध को एक बड़े आव्यूह गुणन के रूप में लिखते हैं, जिसे GPU अपनी अधिकतम गति पर कर सकता है। ध्यान दें कि :numref:`fig_correlation` की हर आउटपुट प्रविष्टि एक डॉट गुणनफल है। इसके लिए $k_\textrm{h} k_\textrm{w}$ तत्वों वाले कर्नेल और संगत इनपुट विंडो, दोनों को एक-एक सदिश में फैलाते हैं। सभी $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ विंडो निकालकर उन्हें पंक्तियों के रूप में एक के नीचे एक रखें, तो एक पैच आव्यूह मिलता है। इस आव्यूह को कर्नेल के सदिश से गुणा करने पर पूरा क्रॉस-सहसंबंध एक साथ निकल आता है। डेटा को इस तरह व्यवस्थित करने की मानक विधि को *im2col* कहते हैं (अर्थात “छवि से स्तंभ”, हालाँकि यहाँ पंक्तियाँ बनती हैं)। :numref:`fig_conv_im2col` में यही विधि :numref:`fig_correlation` के उदाहरण पर दिखाई गई है।

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` का क्रॉस-सहसंबंध एक आव्यूह गुणन के रूप में। इनपुट की चारों $2 \times 2$ विंडो में से हर एक पैच आव्यूह की एक पंक्ति बनती है। इसे सदिश के रूप में फैलाए गए कर्नेल से गुणा करने पर आउटपुट भी सदिश रूप में मिलता है। छायांकित खाने दिखाते हैं कि पहली विंडो से कौन-सी पंक्ति बनती है और उससे आउटपुट 19 कैसे मिलता है। बीच का पिक्सेल, जिसका मान 4 है, चारों विंडो में है, इसलिए वह हर पंक्ति में आता है।](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
इस विधि में काम करने के लिए अतिरिक्त मेमोरी चाहिए, क्योंकि हर भीतरी पिक्सेल की प्रति अधिकतम $k_\textrm{h} k_\textrm{w}$ पैच में आती है। नीचे हम :numref:`fig_correlation` के उदाहरण पर जाँचते हैं कि एक आव्यूह गुणन और `corr2d` ठीक एक ही परिणाम देते हैं।

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
अंतर हर जगह शून्य है। एक आव्यूह गुणन से ठीक वही परिणाम मिलता है जो `corr2d` से मिलता है।

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
इसी गणना का एक और तरीका है। इनपुट के बजाय *कर्नेल* को एक विरल पट्टीदार आव्यूह $\mathbf{W}$ में फैलाएँ और सदिश के रूप में फैलाई गई छवि को इससे गुणा करें। यह ठीक :numref:`fig_why_conv_weight_sharing` के पैनल (c) का द्वि-आयामी रूप है, जहाँ हर पंक्ति में कर्नेल के वही चंद भार दोहराए जाते हैं। कन्वोल्यूशन को एक विशेष संरचना वाली डेंस लेयर की तरह देखने पर हम फिर :numref:`sec_why-conv` की व्युत्पत्ति पर लौट आते हैं (अभ्यास में हम $\mathbf{W}$ को सीधे बनाएँगे)।

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## फ़ीचर मैप और ग्रहणशील क्षेत्र

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
एक कन्वोल्यूशनल लेयर अपने कर्नेल के आकार जितना ही पैच देखती है, जबकि पूरी वस्तु को पहचानने के लिए कहीं बड़े क्षेत्र से जानकारी जुटाकर उसे जोड़ना पड़ता है। :numref:`subsec_why-conv-channels` से याद करें कि कन्वोल्यूशनल लेयर के आउटपुट को *फ़ीचर मैप* कहते हैं (जैसे खड़े किनारों का हमारा मैप `Y`)। फ़ीचर मैप के किसी तत्व $x$ का *ग्रहणशील क्षेत्र* इनपुट छवि (या किसी पिछली लेयर) का वह क्षेत्र है जिसके मान फ़ॉरवर्ड पास में $x$ को प्रभावित कर सकते हैं। यह शब्द :numref:`field_visual` में वर्णित तंत्रिका शरीरक्रिया विज्ञान से लिया गया है।

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### एक के बाद एक लगी लेयर

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` में छायांकित आउटपुट `19` का ग्रहणशील क्षेत्र $2 \times 2$ का है। इसमें $\mathbf{X}$ के ऊपरी बाएँ कोने के चार पिक्सेल आते हैं। अब एक *दूसरी* $2 \times 2$ की कन्वोल्यूशनल लेयर को $2 \times 2$ के आउटपुट $\mathbf{Y}$ पर लगाएँ। इससे एक अदिश $z$ मिलता है। $z$ का मान $\mathbf{Y}$ के चारों तत्वों पर निर्भर करता है, जो मिलकर $\mathbf{X}$ के सभी नौ तत्वों पर निर्भर करते हैं। इस तरह एक के बाद एक लेयर लगाने से गहराई के साथ ग्रहणशील क्षेत्र धीरे-धीरे फैलता है।

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
यह कितनी तेज़ी से बढ़ता है? हर $k \times k$ की कन्वोल्यूशनल लेयर की पहुँच उसके इनपुट के किसी एक तत्व की पहुँच से $k - 1$ पिक्सेल अधिक होती है। इसलिए $L$ लेयरों को क्रम से लगाने पर, यदि उनके कर्नेल के आकार $k_1, \ldots, k_L$ हों, तो ग्रहणशील क्षेत्र की भुजा की लंबाई होती है

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
यदि लगातार $L$ लेयरों में से हर एक $3 \times 3$ की है, तो हर लेयर भुजा की लंबाई में 2 पिक्सेल जोड़ती है। इस तरह ग्रहणशील क्षेत्र $(2L + 1) \times (2L + 1)$ इनपुट को समेटता है। दो लेयर के बाद यह $5 \times 5$ और तीन के बाद $7 \times 7$ होता है (:numref:`fig_conv_receptive_field`)। यह रैखिक बढ़त बड़े-बड़े कर्नेल अपनाने की वजह नहीं, बल्कि एक फ़ायदा है। एक के बाद एक लगी तीन $3 \times 3$ की लेयर उतना ही बड़ा $7 \times 7$ का ग्रहणशील क्षेत्र देती हैं जितना अकेली $7 \times 7$ की लेयर। इसके लिए बहुत कम पैरामीटर चाहिए (प्रति चैनल युग्म 49 की जगह $3 \times 9 = 27$ भार) और एक की जगह तीन बार अरैखिकता लागू होती है।

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![3 के कर्नेल आकार वाली तीन एक-आयामी लेयरों के बाद एक इकाई का ग्रहणशील क्षेत्र। छायांकित खाने वे इकाइयाँ हैं जिन पर सबसे ऊपर की इकाई निर्भर करती है। रेखाएँ इनमें से हर इकाई को उसके इनपुट से जोड़ती हैं। हर लेयर ग्रहणशील क्षेत्र को 2 से चौड़ा करती है, जिससे अंत में 7 इनपुट उसके भीतर आते हैं।](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### प्रभावी ग्रहणशील क्षेत्र

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
ध्यान रखें कि $r$ से *सैद्धांतिक* ग्रहणशील क्षेत्र मापा जाता है। इसमें हर वह इनपुट पिक्सेल आता है जिससे आउटपुट इकाई तक *कोई भी* पथ जाता हो। इससे यह पता नहीं चलता कि वास्तव में हर पिक्सेल का कितना योगदान है। इनपुट ग्रेडिएंट को सीधे मापकर :citet:`Luo.Li.Urtasun.ea.2016` ने दिखाया कि प्रशिक्षित और अप्रशिक्षित दोनों तरह के नेटवर्क में *प्रभावी* ग्रहणशील क्षेत्र का प्रभाव केंद्र से दूर जाने पर तेज़ी से घटता है। इसका वक्र लगभग गाउसियन घंटी के आकार का होता है। सिद्धांततः गहरी लेयर की इकाइयाँ छवि में दूर तक देख सकती हैं, पर उनकी संवेदनशीलता मुख्यतः केंद्र के पास होती है।

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
पथ गिनने से घंटी का आकार समझ में आता है। ग्रहणशील क्षेत्र के किनारे का इनपुट कनेक्शनों की एक ही शृंखला से आउटपुट इकाई तक पहुँचता है, जबकि केंद्र के इनपुट से बीच की लेयरों के ज़रिए कई रास्ते जाते हैं। :numref:`fig_conv_rf_paths` में ये पथ :numref:`fig_conv_receptive_field` में दिखाई गई लेयरों के लिए गिने गए हैं। केंद्र के इनपुट से सबसे ऊपर की इकाई तक सात पथ जाते हैं और किनारे के हर इनपुट से एक। यदि सभी भारों का परिमाण लगभग एक जैसा हो, तो किसी इनपुट का प्रभाव उसके पथों की संख्या के साथ बढ़ता है।

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![हर इकाई से :numref:`fig_conv_receptive_field` की सबसे ऊपर वाली इकाई तक पथों की संख्या (3 के कर्नेल आकार वाली तीन एक-आयामी लेयर)। किसी खाने की संख्या उन इकाइयों की संख्याओं का योग है जिन्हें वह खाना इनपुट देता है। गहरे रंग के खानों से उनकी लेयर के दूसरे खानों की तुलना में अधिक पथ जाते हैं। नीचे की पट्टियाँ इनपुट पंक्ति की संख्याएँ फिर दिखाती हैं: 1, 3, 6, 7, 6, 3, 1।](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
हमारी बुनियादी लेयर की दो व्यावहारिक सीमाएँ अभी बाकी हैं। पैडिंग के बिना हर कन्वोल्यूशन स्थानिक आयाम घटाता है और सीमा के पिक्सेल को कम बार इस्तेमाल करता है। साथ ही, एक कर्नेल केवल एक पैटर्न पहचानता है। आगे :numref:`sec_padding` में हम स्थानिक आयाम तय करने और सीमाओं को संभालने पर विचार करेंगे और :numref:`sec_channels` में कई चैनल वाले फ़ीचर मैप देखेंगे।

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## सारांश

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
कन्वोल्यूशनल लेयर सीखे हुए कर्नेल को इनपुट की हर मान्य विंडो पर खिसकाकर द्वि-आयामी क्रॉस-सहसंबंध निकालती है और एक अदिश बायस जोड़ती है। $1 \times 2$ का सरल परिमित अंतर कर्नेल `[1, -1]` छवि में कहीं भी खड़े किनारे पहचान लेता है। ग्रेडिएंट डिसेंट केवल इनपुट–आउटपुट डेटा से ठीक यही भार सीख लेता है, जैसे गहरे CNN फ़िल्टरों का पूरा पदानुक्रम शुरू से अंत तक सीखते हैं। गणितीय कन्वोल्यूशन कर्नेल को उलटता है, लेकिन जब कर्नेल सीखे जाते हैं तो इससे कोई अंतर नहीं पड़ता। हम एक ही आव्यूह गुणन (*im2col*) से कन्वोल्यूशन की गणना कर सकते हैं। $L$ कन्वोल्यूशनल लेयर क्रम से लगाने पर हर इकाई का ग्रहणशील क्षेत्र रैखिक रूप से बढ़ता है, प्रति लेयर $k_i - 1$ की दर से।

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="आव्यूह गुणन से कन्वोल्यूशन की तेज़ गणना"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col रूपांतरण (छवि से स्तंभ)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **एक सघन GEMM:** हर $k_\textrm{h} \times k_\textrm{w}$ विंडो को फैलाकर `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ की एक पंक्ति बनाएँ और इस आव्यूह को $\textrm{vec}(\mathbf{K})$ से गुणा करें।

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **मेमोरी की लागत:** GPU Tensor Core की अधिकतम गति पाने के बदले पैच की प्रतियाँ रखने के लिए $k_\textrm{h} k_\textrm{w}\times$ जगह चाहिए।

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="दूसरा दृष्टिकोण: कर्नेल को पट्टीदार आव्यूह में फैलाना"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Toeplitz भार आव्यूह से गुणन]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
इनपुट छवि को ऐसे पैच में बाँटने के बजाय जिनमें पिक्सेल दोहराए जाते हैं, हम *कर्नेल* को एक विरल पट्टीदार आव्यूह $\mathbf{W}$ में फैला सकते हैं। यह आव्यूह इनपुट को फैलाकर बने सदिश $\textrm{vec}(\mathbf{X})$ से गुणा होता है।

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **आव्यूह–सदिश रूप:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, जहाँ $\mathbf{W}$ में $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ पंक्तियाँ और $n_\textrm{h} n_\textrm{w}$ स्तंभ हैं।

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} और बैकप्रॉपैगेशन से संबंध:** $\mathbf{W}$ की हर पंक्ति में कर्नेल की वही $k_\textrm{h} k_\textrm{w}$ प्रविष्टियाँ अलग स्तंभों में खिसकी हुई हैं (पैनल c)। इसलिए $\mathbf{X}$ के सापेक्ष बैकवर्ड पास केवल $\mathbf{W}^\top$ से गुणन है (एक ट्रांसपोज़्ड कन्वोल्यूशन)।

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="गहराई के साथ फ़ीचर मैप और ग्रहणशील क्षेत्र"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[क्रमिक लेयरों में ग्रहणशील क्षेत्र की रैखिक वृद्धि]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
किसी इकाई का **ग्रहणशील क्षेत्र** मूल इनपुट छवि का वह क्षेत्र है जिसके पिक्सेल फ़ॉरवर्ड पास में उस इकाई के ऐक्टिवेशन को प्रभावित कर सकते हैं।

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **एक लेयर ($L=1$):** $k_1 \times k_1$ का कर्नेल $k_1 \times k_1$ का इनपुट पैच देखता है।

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **1 की स्ट्राइड वाली $L$ क्रमिक लेयर:** लेयर $i$ के कर्नेल का आकार $k_i$ है। इसकी पहुँच हर अक्ष पर $k_i - 1$ पिक्सेल अधिक है, यदि तुलना लेयर $i-1$ के किसी एक तत्व की पहुँच से करें।
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **क्रम से लगी $L$ लेयर, प्रत्येक $3 \times 3$ की:** हर लेयर $3 - 1 = 2$ पिक्सेल जोड़ती है, जिससे भुजा की लंबाई $r = 2L + 1$ होती है ($3 \to 5 \to 7$)।

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="एक बड़े कर्नेल की तुलना में कई क्रमिक 3×3 कर्नेल बेहतर क्यों हैं?"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[कम पैरामीटर और अरैखिक अभिव्यक्ति क्षमता]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **एक जैसा $7 \times 7$ का ग्रहणशील क्षेत्र:** क्रम से लगी तीन $3 \times 3$ की लेयर ($r = 1 + 2 + 2 + 2 = 7$) ठीक वही $7 \times 7$ पैच देखती हैं जो अकेली $7 \times 7$ की लेयर देखती है।

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% कम पैरामीटर:** भारों की संख्या $3 \times 9 c^2 = 27 c^2$ है, जबकि अकेली लेयर के लिए $49 c^2$ भार चाहिए। दोनों में $c$ चैनल हैं।

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **अधिक अभिव्यक्ति क्षमता:** 1 की जगह **3 अरैखिक ऐक्टिवेशन** क्रम से लगते हैं (§{sec_blocks} में VGG की डिज़ाइन का मूल नियम)।

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="सैद्धांतिक और प्रभावी ग्रहणशील क्षेत्र"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[केंद्र से दूर जाने पर संवेदनशीलता में गाउसियन गिरावट]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **सैद्धांतिक ग्रहणशील क्षेत्र ($r = 2L+1$):** उन सभी पिक्सेल का बाउंडिंग बॉक्स जिनसे आउटपुट इकाई तक *कम से कम एक* पथ जाता है।

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **केंद्रीय सीमा प्रमेय:** केंद्र के पिक्सेल से पथों की संख्या गहराई के साथ घातांकीय रूप से बढ़ती है, जबकि हर कोने के पिक्सेल से केवल **1 पथ** जाता है।

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **प्रभावी ग्रहणशील क्षेत्र (Luo एवं अन्य, 2016):** इनपुट ग्रेडिएंट $\partial y / \partial \mathbf{X}$ का परिमाण गाउसियन घंटी के आकार के वक्र के अनुसार घटता है, जिसका मानक विचलन $\sigma \propto \sqrt{L}$ है।

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="अपनी समझ परखें: क्रॉस-सहसंबंध और ग्रहणशील क्षेत्र"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[अपनी समझ परखें]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
आउटपुट की शेप, `im2col` और ग्रहणशील क्षेत्र की गणना के बारे में अपनी समझ परखें।

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col आव्यूह के आयाम:** आप बिना पैडिंग के $3 \times 3$ का एकल-चैनल कर्नेल $10 \times 10$ की छवि पर `im2col` से लगाते हैं। आउटपुट की स्थानिक शेप क्या है? `patches` आव्यूह और कर्नेल को फैलाकर बने सदिश के सटीक आयाम क्या हैं?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **पूरी Fashion-MNIST छवि के लिए गहराई:** 1 की स्ट्राइड पर $3 \times 3$ की कितनी लेयर क्रम से लगाएँ कि एक इकाई का सैद्धांतिक ग्रहणशील क्षेत्र पूरी $28 \times 28$ की छवि को समेट ले? $5 \times 5$ की कितनी लेयर चाहिए?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **सीमा खिसकाने से किनारों के लॉस में बदलाव:** $6 \times 8$ की छवि में खड़े किनारे सीखने के हमारे प्रयोग में आरंभिक लॉस $\approx 12$ क्यों था? यदि काली पट्टी दाईं सीमा तक फैली होती (`X[:, 2:8] = 0`), तो आरंभिक लॉस कितना होता?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**उत्तर:**
(1) आउटपुट की शेप $(10-3+1) \times (10-3+1) = 8 \times 8$ है ($64$ आउटपुट)। `patches` आव्यूह की शेप $(64, 9)$ है और इसे $(9, 1)$ के कर्नेल सदिश से गुणा किया जाता है।
(2) $3 \times 3$ के लिए $2L + 1 \ge 28 \implies L = 14$ लेयर चाहिए। $5 \times 5$ के लिए $4L + 1 \ge 28 \implies L = 7$ लेयर चाहिए।
(3) सभी 6 पंक्तियों में $\pm 1$ मान वाले दो किनारे थे ($6 \times 2 \times 1^2 = 12$)। काली पट्टी को दाईं सीमा तक बढ़ाने से दाईं ओर का काले से सफ़ेद वाला किनारा हट जाता है। हर पंक्ति में एक किनारा बचता है और आरंभिक लॉस आधा होकर $\approx 6$ रह जाता है।

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="सारांश और मुख्य बातें"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[खंड का सार]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2D क्रॉस-सहसंबंध और किनारे सीखना:** $\mathbf{K}$ को सभी मान्य विंडो ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$) पर खिसकाते हैं। SGD 9 कदमों में $[1, -1]$ सीख लेता है।

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col से आव्यूह गुणन:** पैच की प्रतियों के लिए $k_\textrm{h} k_\textrm{w}\times$ जगह लेकर GPU पर GEMM का अधिकतम थ्रूपुट मिलता है।

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **क्रमिक लेयरों के ग्रहणशील क्षेत्र:** $r = 1 + \sum (k_i - 1) = 2L+1$ उन लेयरों के लिए है जिनके कर्नेल $3 \times 3$ के हैं। संवेदनशीलता का गाउसियन शिखर केंद्र पर होता है।


=== DECISIONS ===
- patch matrix → पैच आव्यूह; sparse banded matrix → विरल पट्टीदार आव्यूह; flatten → सदिश के रूप में फैलाना (or specify the resulting row).
- theoretical/effective receptive field → सैद्धांतिक/प्रभावी ग्रहणशील क्षेत्र; path → पथ; stacked layers → एक के बाद एक लगी लेयर (क्रमिक लेयर in compact slide text).
