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
`corr2d` में नेस्टेड लूप गणित को स्पष्ट करते हैं, लेकिन Python में एक बार में एक आउटपुट पिक्सेल पर जाना वास्तविक काम के लिए बहुत धीमा है। एक पुराना उपाय, जो आज भी लाइब्रेरी द्वारा चुने जाने वाले एल्गोरिद्मों में से एक है, क्रॉस-सहसंबंध को एक बड़े आव्यूह गुणन के रूप में ढालना है। यही वह संक्रिया है जिसे GPU अधिकतम गति से करते हैं। ध्यान दें कि :numref:`fig_correlation` में हर आउटपुट प्रविष्टि दरअसल चपटे किए गए $k_\textrm{h} k_\textrm{w}$-अवयव वाले कर्नेल और संबंधित चपटी इनपुट विंडो का डॉट गुणनफल है। सभी $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ विंडो निकालकर उन्हें पंक्तियों के रूप में रखने से पैच आव्यूह मिलता है; इसे चपटे कर्नेल से गुणा करने पर पूरा क्रॉस-सहसंबंध एक ही बार में गणना होता है। इस मानक लेआउट परिवर्तन को *im2col* ("छवि से स्तंभ", या यहाँ पंक्तियाँ) कहते हैं। :numref:`fig_conv_im2col` इसे :numref:`fig_correlation` के उदाहरण पर करता है।

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` का क्रॉस-सहसंबंध एक आव्यूह गुणनफल के रूप में। इनपुट की चार $2 \times 2$ विंडो में से प्रत्येक पैच आव्यूह की एक पंक्ति बनती है और चपटे कर्नेल से गुणा करने पर चपटा आउटपुट मिलता है। छायांकित खाने पहली विंडो से उसकी पंक्ति और आउटपुट 19 तक का रास्ता दिखाते हैं। केंद्रीय पिक्सेल 4 चारों विंडो में है, इसलिए वह हर पंक्ति में आता है।](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
इसके बदले अतिरिक्त कार्यक्षेत्र मेमोरी लगती है, क्योंकि हर भीतरी पिक्सेल की प्रतिलिपि अधिकतम $k_\textrm{h} k_\textrm{w}$ पैच में बनती है। नीचे हम :numref:`fig_correlation` पर जाँचते हैं कि यह एक आव्यूह गुणन बिल्कुल `corr2d` से मेल खाता है:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
अंतर सर्वत्र शून्य है: एक आव्यूह गुणनफल बिल्कुल `corr2d` का परिणाम देता है।

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
समतुल्य रूप से हम इनपुट के बजाय *कर्नेल* को खोलकर एक विरल पट्टीदार आव्यूह $\mathbf{W}$ बना सकते हैं जो चपटी छवि से गुणा होता है। यह :numref:`fig_why_conv_weight_sharing` के पैनल (c) का ठीक 2D समरूप है, जहाँ हर पंक्ति वही कुछ कर्नेल भार दोहराती है। कन्वोल्यूशन को संरचित डेंस लेयर मानने से हम वापस :numref:`sec_why-conv` की व्युत्पत्ति तक पहुँचते हैं (और अभ्यासों में $\mathbf{W}$ स्पष्ट रूप से बनाया जाता है)।

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## फ़ीचर मानचित्र और ग्रहणशील क्षेत्र

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
एक कन्वोल्यूशनल लेयर केवल अपने कर्नेल के आकार का पैच देखती है, जबकि पूरी वस्तुओं को पहचानने के लिए कहीं बड़े क्षेत्रों के संदर्भ को एकीकृत करना पड़ता है। :numref:`subsec_why-conv-channels` से याद करें कि कन्वोल्यूशनल लेयर के आउटपुट को *फ़ीचर मानचित्र* कहते हैं (जैसे हमारा ऊर्ध्वाधर किनारा मानचित्र `Y`)। फ़ीचर मानचित्र के किसी भी अवयव $x$ का *ग्रहणशील क्षेत्र* इनपुट छवि (या किसी पिछली लेयर) का वह क्षेत्र है जिसके मान फ़ॉरवर्ड प्रॉपैगेशन के दौरान $x$ को प्रभावित कर सकते हैं। यह तंत्रिका शरीरविज्ञान का शब्द :numref:`field_visual` से लिया गया है।

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### एक के ऊपर एक लेयर

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` में छायांकित आउटपुट `19` का ग्रहणशील क्षेत्र $2 \times 2$ है, जो $\mathbf{X}$ के ऊपरी बाएँ चार पिक्सेलों को समेटता है। अब $2 \times 2$ आउटपुट $\mathbf{Y}$ के ऊपर एक *दूसरी* $2 \times 2$ कन्वोल्यूशनल लेयर रखकर एक अदिश $z$ बनाएँ: $z$, $\mathbf{Y}$ के चारों अवयवों पर निर्भर करता है, जो मिलकर $\mathbf{X}$ के सभी नौ अवयवों पर निर्भर करते हैं। लेयरों को एक के ऊपर एक रखने से गहराई के साथ ग्रहणशील क्षेत्र सुचारु रूप से फैलता है।

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
यह कितनी तेजी से बढ़ता है? हर $k \times k$ कन्वोल्यूशनल लेयर अपने इनपुट के एक अवयव से $k - 1$ पिक्सेल आगे तक पहुँचती है, इसलिए $k_1, \ldots, k_L$ कर्नेल आकारों वाली $L$ लेयरों को एक के ऊपर एक रखने पर ग्रहणशील क्षेत्र की भुजा की लंबाई मिलती है

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
लगातार $3 \times 3$ आकार की $L$ लेयरों में हर लेयर 2 पिक्सेल जोड़ती है और $(2L + 1) \times (2L + 1)$ इनपुट समेटती है: दो लेयरों के बाद $5 \times 5$ और तीन के बाद $7 \times 7$ (:numref:`fig_conv_receptive_field`)। यह रैखिक वृद्धि बड़े कर्नेल लेने की वजह के बजाय एक लाभ है: एक के ऊपर एक रखी तीन $3 \times 3$ लेयरें, एक $7 \times 7$ लेयर के समान $7 \times 7$ ग्रहणशील क्षेत्र बहुत कम पैरामीटरों से प्राप्त करती हैं (प्रति चैनल जोड़ी $3 \times 9 = 27$ भार बनाम 49), और एक के बजाय तीन अरैखिकताएँ बीच-बीच में लगाती हैं।

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![कर्नेल आकार 3 वाली तीन एक-आयामी लेयरों के बाद एक इकाई का ग्रहणशील क्षेत्र। छायांकित खाने वे इकाइयाँ हैं जिन पर सबसे ऊपर की इकाई निर्भर करती है और रेखाएँ उनमें से प्रत्येक को उसके इनपुट से जोड़ती हैं। हर लेयर ग्रहणशील क्षेत्र को 2 से चौड़ा करती है, जिससे यह 7 इनपुट तक पहुँचता है।](../img/conv-receptive-field.svg)

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
ध्यान रखें कि $r$ *सैद्धांतिक* ग्रहणशील क्षेत्र मापता है (हर वह इनपुट पिक्सेल जिसका आउटपुट इकाई तक *कोई भी* रास्ता है), लेकिन यह नहीं बताता कि हर पिक्सेल वास्तव में कितना योगदान देता है। इनपुट ग्रेडिएंट सीधे मापकर :citet:`Luo.Li.Urtasun.ea.2016` ने दिखाया कि अप्रशिक्षित और प्रशिक्षित दोनों नेटवर्कों में *प्रभावी* ग्रहणशील क्षेत्र केंद्र से दूर जाने पर लगभग गाउसी घंटी के आकार के वक्र में तेज़ी से घटता है। गहरी इकाइयाँ सैद्धांतिक रूप से छवि में दूर तक देख सकती हैं, लेकिन उनकी संवेदनशीलता बहुत हद तक केंद्र के पास केंद्रित होती है।

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
रास्तों की गिनती से घंटी का आकार समझ में आता है। ग्रहणशील क्षेत्र के किनारे का इनपुट कनेक्शनों की एक ही श्रृंखला से आउटपुट इकाई तक पहुँचता है, जबकि केंद्रीय इनपुट बीच की लेयरों से कई रास्ते ले सकता है। :numref:`fig_conv_rf_paths`, :numref:`fig_conv_receptive_field` की लेयरों के ढेर के लिए इनकी गिनती करता है: केंद्रीय इनपुट से सबसे ऊपर की इकाई तक सात रास्ते हैं और हर किनारे के इनपुट से एक। यदि सभी भारों का परिमाण समान हो, तो इनपुट का प्रभाव उसके रास्तों की संख्या के साथ बढ़ता है।

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![:numref:`fig_conv_receptive_field` की प्रत्येक इकाई से सबसे ऊपर की इकाई तक रास्तों की संख्या (कर्नेल आकार 3 वाली तीन एक-आयामी लेयरें)। हर गिनती उन इकाइयों की गिनतियों का योग है जिन्हें वह खाना इनपुट देता है। गहरे खानों के रास्ते उनकी लेयर के अन्य खानों से अधिक हैं और बार इनपुट पंक्ति की गिनती दोहराते हैं: 1, 3, 6, 7, 6, 3, 1।](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
हमारी बुनियादी लेयर में दो व्यावहारिक कमियाँ बाकी हैं: बिना पैडिंग का हर कन्वोल्यूशन स्थानिक आयामों को सिकोड़ता है और सीमा के पिक्सेलों का कम नमूनाकरण करता है, तथा एक कर्नेल केवल एक पैटर्न पहचानता है। आगे हम :numref:`sec_padding` में स्थानिक आकार और सीमाएँ, और :numref:`sec_channels` में बहु-चैनल फ़ीचर मानचित्र देखेंगे।

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
कन्वोल्यूशनल लेयर इनपुट की हर वैध विंडो पर कर्नेल सरकाकर सीखे हुए कर्नेल से द्वि-आयामी क्रॉस-सहसंबंध करती है और एक अदिश बायस जोड़ती है। एक सरल $1 \times 2$ परिमित-अंतर कर्नेल `[1, -1]` छवि में कहीं भी ऊर्ध्वाधर किनारे पहचानता है और ग्रेडिएंट डिसेंट केवल इनपुट–आउटपुट डेटा से ठीक वे भार प्राप्त करता है, जैसे गहरे CNN फ़िल्टरों का पूरा पदानुक्रम शुरू से अंत तक सीखते हैं। गणितीय कन्वोल्यूशन कर्नेल को उलटता है, लेकिन कर्नेल सीख लिए जाने पर इससे कोई अंतर नहीं पड़ता। कन्वोल्यूशन का मूल्यांकन एक आव्यूह गुणन (*im2col*) से किया जा सकता है और $L$ कन्वोल्यूशनल लेयरों को एक के ऊपर एक रखने से हर इकाई का ग्रहणशील क्षेत्र प्रति लेयर $k_i - 1$ से रैखिक रूप से चौड़ा होता है।

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="तेज़ निष्पादन: आव्यूह गुणन से कन्वोल्यूशन"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col (छवि से स्तंभ) परिवर्तन]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **एक डेंस GEMM:** हर $k_\textrm{h} \times k_\textrm{w}$ विंडो को `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ की पंक्ति में चपटा करें और $\textrm{vec}(\mathbf{K})$ से गुणा करें।

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **समझौता:** $k_\textrm{h} k_\textrm{w}\times$ पैच कार्यक्षेत्र की प्रतिलिपि की कीमत पर GPU Tensor Core की अधिकतम गति।

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="दूसरा दृष्टिकोण: कर्नेल को पट्टीदार आव्यूह में खोलना"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[टोप्लिट्ज़ भार आव्यूह गुणन]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
इनपुट छवि को प्रतिलिपि किए गए पैचों में खोलने के बजाय हम *कर्नेल* को विरल पट्टीदार आव्यूह $\mathbf{W}$ में खोल सकते हैं, जो चपटे इनपुट सदिश $\textrm{vec}(\mathbf{X})$ से गुणा होता है:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **आव्यूह–सदिश रूप:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, जहाँ $\mathbf{W}$ में $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ पंक्तियाँ और $n_\textrm{h} n_\textrm{w}$ स्तंभ हैं।

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} और बैकप्रॉप से संबंध:** $\mathbf{W}$ की हर पंक्ति में वही $k_\textrm{h} k_\textrm{w}$ कर्नेल प्रविष्टियाँ स्तंभों में खिसकी हुई हैं (पैनल c), इसलिए $\mathbf{X}$ के सापेक्ष बैकवर्ड पास केवल $\mathbf{W}^\top$ से गुणन है (एक ट्रांसपोज़्ड कन्वोल्यूशन)।

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="गहराई के साथ फ़ीचर मानचित्र और ग्रहणशील क्षेत्र"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[एक के ऊपर एक लेयरों में ग्रहणशील क्षेत्र की रैखिक वृद्धि]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
किसी इकाई का **ग्रहणशील क्षेत्र** मूल इनपुट छवि का वह भाग है जिसके पिक्सेल फ़ॉरवर्ड प्रॉपैगेशन के दौरान उस इकाई के सक्रियण को प्रभावित कर सकते हैं:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **एक लेयर ($L=1$):** $k_1 \times k_1$ कर्नेल $k_1 \times k_1$ इनपुट पैच देखता है।

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **स्ट्राइड 1 वाली $L$ लेयरें एक के ऊपर एक:** कर्नेल आकार $k_i$ वाली लेयर $i$, लेयर $i-1$ के एक अवयव से हर अक्ष पर $k_i - 1$ पिक्सेल आगे पहुँचती है:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **एक के ऊपर एक $L$ $3 \times 3$ लेयरें:** हर लेयर $3 - 1 = 2$ पिक्सेल जोड़ती है, जिससे भुजा की लंबाई $r = 2L + 1$ ($3 \to 5 \to 7$) होती है।

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 कर्नेलों की गहरी श्रृंखला एक बड़े कर्नेल से बेहतर क्यों है"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[पैरामीटर दक्षता और अरैखिक अभिव्यक्ति क्षमता]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **समान $7 \times 7$ ग्रहणशील क्षेत्र:** एक के ऊपर एक तीन $3 \times 3$ लेयरें ($r = 1 + 2 + 2 + 2 = 7$), एक $7 \times 7$ लेयर के बिल्कुल समान $7 \times 7$ पैच को देखती हैं।

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% कम पैरामीटर:** $c$ चैनलों पर $49 c^2$ की जगह $3 \times 9 c^2 = 27 c^2$ भार।

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **अधिक अभिव्यक्ति क्षमता:** 1 की जगह **3 अरैखिक सक्रियणों** को संयोजित करता है (§{sec_blocks} में VGG का मूल डिज़ाइन नियम)।

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="सैद्धांतिक बनाम प्रभावी ग्रहणशील क्षेत्र"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[केंद्र से दूर संवेदनशीलता का गाउसी क्षय]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **सैद्धांतिक RF ($r = 2L+1$):** उन सभी पिक्सेलों का सीमांकन बॉक्स जिनका आउटपुट इकाई तक *कम से कम एक* रास्ता है।

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **केंद्रीय सीमा प्रमेय:** केंद्रीय पिक्सेल घातीय संख्या में रास्तों से जुड़ते हैं, जबकि कोने के पिक्सेल केवल **1 रास्ते** से जुड़ते हैं।

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **प्रभावी RF (Luo et al., 2016):** इनपुट ग्रेडिएंट का परिमाण $\partial y / \partial \mathbf{X}$ मानक विचलन $\sigma \propto \sqrt{L}$ वाले गाउसी घंटी वक्र में घटता है।

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="अवधारणा जाँच: क्रॉस-सहसंबंध और ग्रहणशील क्षेत्र"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[अवधारणा जाँच]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
आउटपुट शेप, `im2col` और ग्रहणशील क्षेत्र की गणना की अपनी समझ जाँचें:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col आव्यूह के आयाम:** आप बिना पैडिंग वाला $3 \times 3$ एकल-चैनल कर्नेल `im2col` से $10 \times 10$ छवि पर लगाते हैं। आउटपुट का स्थानिक शेप क्या है और `patches` आव्यूह तथा चपटे कर्नेल सदिश के सटीक आयाम क्या हैं?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST को समेटने की गहराई:** स्ट्राइड 1 पर कितनी $3 \times 3$ लेयरें एक के ऊपर एक रखनी होंगी ताकि एक इकाई का सैद्धांतिक ग्रहणशील क्षेत्र पूरी $28 \times 28$ छवि समेट ले? $5 \times 5$ लेयरों के लिए कितनी?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **किनारा-लॉस की सीमा में बदलाव:** हमारे $6 \times 8$ ऊर्ध्वाधर किनारा सीखने के प्रयोग में लॉस $\approx 12$ से क्यों शुरू हुआ, और अगर काली पट्टी दाएँ किनारे तक जाती (`X[:, 2:8] = 0`) तो शुरुआती लॉस क्या होता?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**उत्तर:**
(1) आउटपुट शेप $(10-3+1) \times (10-3+1) = 8 \times 8$ है ($64$ आउटपुट)। `patches` आव्यूह का शेप $(64, 9)$ है और यह $(9, 1)$ कर्नेल सदिश से गुणा होता है।
(2) $3 \times 3$ के लिए: $2L + 1 \ge 28 \implies L = 14$ लेयरें। $5 \times 5$ के लिए: $4L + 1 \ge 28 \implies L = 7$ लेयरें।
(3) 6 पंक्तियों में से हर एक में दो $\pm 1$ किनारे थे ($6 \times 2 \times 1^2 = 12$)। काली पट्टी को दाईं सीमा तक बढ़ाने से दायाँ काले से सफ़ेद किनारा हट जाता है, प्रति पंक्ति एक किनारा बचता है और शुरुआती लॉस आधा होकर $\approx 6$ हो जाता है।

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
- **2D क्रॉस-सहसंबंध और किनारा सीखना:** $\mathbf{K}$ को वैध विंडो पर सरकाता है ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD 9 चरणों में $[1, -1]$ सीखता है।

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col आव्यूह गुणन:** GPU GEMM की अधिकतम थ्रूपुट के लिए $k_\textrm{h} k_\textrm{w}\times$ पैच प्रतिलिपि की कीमत चुकाता है।

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **एक के ऊपर एक ग्रहणशील क्षेत्र:** $3 \times 3$ की श्रृंखला के लिए $r = 1 + \sum (k_i - 1) = 2L+1$, केंद्र पर गाउसी संवेदनशीलता के साथ।

=== DECISIONS ===
- सहयोगी शैली में “हम” और सामान्य हिंदी विराम चिह्न प्रयुक्त हैं; तकनीकी संक्षिप्त नाम और संख्याएँ मूल रूप में रखे हैं।
- edge map → किनारा मानचित्र; edge detector → किनारा संसूचक; feature map → फ़ीचर मानचित्र; window → विंडो; patch → पैच; dense layer → डेंस लेयर; parameter → पैरामीटर; forward propagation → फ़ॉरवर्ड प्रॉपैगेशन; activation → सक्रियण; vector → सदिश; banded matrix → पट्टीदार आव्यूह; sparse → विरल।

=== NOTES ===
- 24a25ec4, f95c9394, b302e3ef, 01c175bf, 755ac11a, 3af8acd0: हिंदी वाक्यक्रम के अनुसार गणित या कोड के टोकन का क्रम बदला है।
