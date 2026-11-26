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
`corr2d` के नेस्टेड लूप गणित को पारदर्शी तो बना देते हैं, पर Python में एक-एक आउटपुट पिक्सेल पर जाना असली वर्कलोड के लिए बहुत धीमा है। इसका एक पुराना उपाय, जो आज भी उन एल्गोरिद्मों में से है जिनमें से लाइब्रेरियाँ चुनती हैं, यह है कि क्रॉस-सहसंबंध को एक ही बड़े आव्यूह गुणन का रूप दे दिया जाए, यानी ठीक वही संक्रिया जिसे GPU अपनी अधिकतम गति से चलाते हैं। ध्यान दें कि :numref:`fig_correlation` की हर आउटपुट प्रविष्टि बस समतल किए गए $k_\textrm{h} k_\textrm{w}$ अवयवों वाले कर्नेल और संगत समतल की गई इनपुट विंडो का डॉट गुणनफल है। सभी $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ विंडो निकालकर उन्हें पंक्तियों के रूप में एक के ऊपर एक रखने से एक पैच आव्यूह बनता है; उसे समतल किए गए कर्नेल से गुणा करने पर पूरा क्रॉस-सहसंबंध एक ही बार में निकल आता है। लेआउट के इस मानक रूपांतरण को *im2col* कहते हैं ("image to columns", यानी इमेज से स्तंभ; यहाँ पंक्तियाँ)। :numref:`fig_conv_im2col` इसे :numref:`fig_correlation` के उदाहरण पर करके दिखाता है।

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![एक ही आव्यूह गुणनफल के रूप में :numref:`fig_correlation` का क्रॉस-सहसंबंध। इनपुट की चारों $2 \times 2$ विंडो में से हर एक पैच आव्यूह की एक पंक्ति बनती है, और समतल किए गए कर्नेल से गुणा करने पर समतल किया गया आउटपुट मिलता है। छायांकित खाने पहली विंडो को उसकी पंक्ति तक और फिर आउटपुट 19 तक दिखाते हैं। केंद्र का पिक्सेल 4 चारों विंडो में आता है, इसलिए वह हर पंक्ति में दिखता है।](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
इसकी क़ीमत अतिरिक्त वर्कस्पेस मेमोरी है, क्योंकि हर भीतरी पिक्सेल की प्रतियाँ अधिकतम $k_\textrm{h} k_\textrm{w}$ पैचों में बनती हैं। नीचे हम :numref:`fig_correlation` पर जाँचते हैं कि यह अकेला आव्यूह गुणन `corr2d` से ठीक-ठीक मेल खाता है:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
अंतर ठीक शून्य है: एक आव्यूह गुणनफल `corr2d` को पूरी तरह दोहरा देता है।

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
इसी के तुल्य, हम इनपुट की जगह *कर्नेल* को खोलकर एक विरल पट्टीदार (banded) आव्यूह $\mathbf{W}$ बना सकते हैं, जो समतल की गई इमेज को गुणा करता है। यह :numref:`fig_why_conv_weight_sharing` के पैनल (c) का ठीक 2D प्रतिरूप है, जहाँ हर पंक्ति कर्नेल के उन्हीं गिने-चुने भारों को दोहराती है। कन्वोल्यूशन को एक संरचित डेंस लेयर के रूप में देखना हमें वापस :numref:`sec_why-conv` की व्युत्पत्ति पर ले आता है (और अभ्यासों में $\mathbf{W}$ को स्पष्ट रूप से बनाया जाता है)।

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
अकेली कन्वोल्यूशनल लेयर को केवल अपने कर्नेल के आकार का पैच दिखता है, जबकि पूरी वस्तुओं को पहचानने के लिए कहीं चौड़े क्षेत्रों का संदर्भ जोड़ना पड़ता है। :numref:`subsec_why-conv-channels` से याद करें कि कन्वोल्यूशनल लेयर के आउटपुट को *फ़ीचर मैप* कहते हैं (जैसे हमारा ऊर्ध्वाधर एज मैप `Y`)। फ़ीचर मैप के किसी भी अवयव $x$ का *ग्रहणशील क्षेत्र* (receptive field) इनपुट इमेज का (या किसी पिछली लेयर का) वह क्षेत्र है जिसके मान फ़ॉरवर्ड पास के दौरान $x$ को प्रभावित कर सकते हैं; यह शब्द :numref:`field_visual` की तंत्रिका-क्रियाविज्ञान (neurophysiology) की शब्दावली से लिया गया है।

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### एक के ऊपर एक रखी लेयरें

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` में छायांकित आउटपुट `19` का ग्रहणशील क्षेत्र $2 \times 2$ है, जो $\mathbf{X}$ के ऊपरी-बाएँ चार पिक्सेल ढकता है। अब एक *दूसरी* $2 \times 2$ कन्वोल्यूशनल लेयर को $2 \times 2$ आउटपुट $\mathbf{Y}$ के ऊपर रखिए, जो एक अकेला अदिश $z$ देती है: $z$ आव्यूह $\mathbf{Y}$ के चारों अवयवों पर निर्भर है, और वे मिलकर $\mathbf{X}$ के सभी नौ अवयवों पर निर्भर हैं। लेयरों को एक के ऊपर एक रखने से ग्रहणशील क्षेत्र गहराई के साथ सुचारु रूप से फैलता जाता है।

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
यह कितनी तेज़ी से बढ़ता है? हर $k \times k$ कन्वोल्यूशनल लेयर अपने इनपुट के एक अकेले अवयव की तुलना में $k - 1$ पिक्सेल आगे तक पहुँचती है, इसलिए $L$ लेयरें, जिनके कर्नेल आकार $k_1, \ldots, k_L$ हों, एक के ऊपर एक रखने पर ग्रहणशील क्षेत्र की भुजा की लंबाई होती है

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
एक के बाद एक रखी $L$ लेयरों के स्टैक में, जिसकी सभी लेयरें $3 \times 3$ हों, हर लेयर 2 पिक्सेल जोड़ती है, जिससे $(2L + 1) \times (2L + 1)$ इनपुट ढकते हैं: दो लेयरों के बाद $5 \times 5$ और तीन के बाद $7 \times 7$ (:numref:`fig_conv_receptive_field`)। यह रैखिक वृद्धि असल में एक वरदान है, बहुत बड़े कर्नेल अपनाने का कारण नहीं: एक के ऊपर एक रखी तीन $3 \times 3$ लेयरें वही $7 \times 7$ ग्रहणशील क्षेत्र पाती हैं जो एक अकेली $7 \times 7$ लेयर पाती है, पर कहीं कम पैरामीटरों में (हर चैनल-जोड़े के लिए $3 \times 9 = 27$ भार बनाम 49), और साथ ही उनके बीच-बीच में एक की जगह तीन अरैखिकताएँ आती हैं।

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![कर्नेल आकार 3 वाली तीन एक-आयामी लेयरों के बाद एक इकाई का ग्रहणशील क्षेत्र। छायांकित खाने वे इकाइयाँ हैं जिन पर सबसे ऊपर की इकाई निर्भर है, और रेखाएँ उनमें से हर एक को उसके इनपुटों से जोड़ती हैं। हर लेयर ग्रहणशील क्षेत्र को 2 से चौड़ा करती है, 7 इनपुट तक।](../img/conv-receptive-field.svg)

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
ध्यान रखें कि $r$ *सैद्धांतिक* ग्रहणशील क्षेत्र मापता है (हर वह इनपुट पिक्सेल जिससे आउटपुट इकाई तक *कोई भी* पथ हो), और यह नहीं बताता कि हर पिक्सेल वास्तव में कितना योगदान देता है। इनपुट ग्रेडिएंट को सीधे मापकर :citet:`Luo.Li.Urtasun.ea.2016` ने दिखाया कि *प्रभावी* ग्रहणशील क्षेत्र केंद्र से दूर जाने पर तेज़ी से घटता है, लगभग गाउसियन घंटी-वक्र की तरह, और ऐसा अप्रशिक्षित तथा प्रशिक्षित दोनों नेटवर्कों में होता है। गहरी इकाइयाँ सैद्धांतिक रूप से इमेज में दूर तक देख सकती हैं, पर उनकी संवेदनशीलता भारी मात्रा में केंद्र के पास ही सिमटी रहती है।

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
पथों को गिनने से घंटी का आकार समझ में आता है। ग्रहणशील क्षेत्र के किनारे का इनपुट आउटपुट इकाई तक कनेक्शनों की एक ही शृंखला से पहुँचता है, जबकि केंद्र का इनपुट बीच की लेयरों से होकर कई रास्ते ले सकता है। :numref:`fig_conv_rf_paths` इन्हें :numref:`fig_conv_receptive_field` के टावर के लिए गिनता है: केंद्र के इनपुट से सबसे ऊपर की इकाई तक सात पथ हैं और किनारे के हर इनपुट से एक। यदि सभी भारों का परिमाण लगभग एक जैसा हो, तो किसी इनपुट का प्रभाव उसके पथों की संख्या के साथ बढ़ता है।

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![हर इकाई से :numref:`fig_conv_receptive_field` की सबसे ऊपर की इकाई तक पथों की संख्या (कर्नेल आकार 3 वाली तीन एक-आयामी लेयरें)। हर गिनती उन इकाइयों की गिनतियों का योग है जिन्हें वह खाना इनपुट देता है। गहरे रंग के खानों में अपनी लेयर के बाक़ी खानों से अधिक पथ हैं, और बार इनपुट पंक्ति की गिनतियाँ दोहराते हैं: 1, 3, 6, 7, 6, 3, 1।](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
हमारी बुनियादी लेयर में दो व्यावहारिक कमियाँ बची हैं: बिना पैडिंग का हर कन्वोल्यूशन स्थानिक आयामों को सिकोड़ता है और साथ ही सीमा के पिक्सेलों को कम बार गिनता है, और एक अकेला कर्नेल केवल एक पैटर्न पकड़ता है। स्थानिक आकार और सीमाओं को हम आगे :numref:`sec_padding` में देखेंगे, और बहु-चैनल फ़ीचर मैप को :numref:`sec_channels` में।

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
कन्वोल्यूशनल लेयर एक सीखे हुए कर्नेल के साथ द्वि-आयामी क्रॉस-सहसंबंध लगाती है और एक अदिश बायस जोड़ती है; इसके लिए वह कर्नेल को इनपुट की हर वैध विंडो पर खिसकाती है। $1 \times 2$ का एक सरल परिमित अंतर कर्नेल `[1, -1]` इमेज में कहीं भी ऊर्ध्वाधर एज पकड़ लेता है, और ग्रेडिएंट डिसेंट ठीक वही भार केवल इनपुट–आउटपुट डेटा से पा लेता है, वैसे ही जैसे डीप CNN फ़िल्टरों के पूरे पदानुक्रम एक सिरे से दूसरे सिरे तक सीखते हैं। गणितीय कन्वोल्यूशन कर्नेल को पलट देता है, पर जब कर्नेल सीखे जाते हैं तो इससे कोई अंतर नहीं पड़ता। कन्वोल्यूशन को एक ही आव्यूह गुणन (*im2col*) के रूप में निकाला जा सकता है, और $L$ कन्वोल्यूशनल लेयरें एक के ऊपर एक रखने से हर इकाई का ग्रहणशील क्षेत्र रैखिक रूप से, प्रति लेयर $k_i - 1$ से, चौड़ा होता है।

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="तेज़ निष्पादन: आव्यूह गुणन से कन्वोल्यूशन"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col (इमेज से स्तंभ) रूपांतरण]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **एक ही डेंस GEMM:** हर $k_\textrm{h} \times k_\textrm{w}$ विंडो को समतल करके `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ की एक पंक्ति बनाएँ और $\textrm{vec}(\mathbf{K})$ से गुणा करें।

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **समझौता:** GPU Tensor Core की अधिकतम गति, जिसकी क़ीमत पैचों की प्रतियों के लिए $k_\textrm{h} k_\textrm{w}\times$ वर्कस्पेस है।

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="दूसरा दृष्टिकोण: कर्नेल को पट्टीदार आव्यूह में खोलना"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Toeplitz भार आव्यूह से गुणन]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
इनपुट इमेज को दोहराए गए पैचों में खोलने की जगह हम *कर्नेल* को एक विरल पट्टीदार आव्यूह $\mathbf{W}$ में खोल सकते हैं, जो समतल किए गए इनपुट सदिश $\textrm{vec}(\mathbf{X})$ को गुणा करता है:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **आव्यूह–सदिश रूप:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, जहाँ $\mathbf{W}$ में $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ पंक्तियाँ और $n_\textrm{h} n_\textrm{w}$ स्तंभ हैं।

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} और बैकप्रॉपैगेशन से संबंध:** $\mathbf{W}$ की हर पंक्ति में वही $k_\textrm{h} k_\textrm{w}$ कर्नेल प्रविष्टियाँ स्तंभों में खिसकी हुई होती हैं (पैनल c), इसलिए $\mathbf{X}$ के सापेक्ष बैकवर्ड पास बस $\mathbf{W}^\top$ से गुणन है (एक ट्रांसपोज़्ड कन्वोल्यूशन)।

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="गहराई के साथ फ़ीचर मैप और ग्रहणशील क्षेत्र"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[एक के ऊपर एक रखी लेयरों में ग्रहणशील क्षेत्र की रैखिक वृद्धि]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
किसी इकाई का **ग्रहणशील क्षेत्र** मूल इनपुट इमेज का वह क्षेत्र है जिसके पिक्सेल फ़ॉरवर्ड पास के दौरान उस इकाई के ऐक्टिवेशन को प्रभावित कर सकते हैं:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **एक लेयर ($L=1$):** $k_1 \times k_1$ कर्नेल को $k_1 \times k_1$ का इनपुट पैच दिखता है।

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **स्ट्राइड 1 की $L$ लेयरें एक के ऊपर एक:** लेयर $i$, जिसका कर्नेल आकार $k_i$ है, हर अक्ष पर $k_i - 1$ पिक्सेल आगे तक पहुँचती है, लेयर $i-1$ के एक अकेले अवयव की तुलना में:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **एक के ऊपर एक $L$ लेयरें, सभी $3 \times 3$:** हर लेयर $3 - 1 = 2$ पिक्सेल जोड़ती है, जिससे भुजा की लंबाई $r = 2L + 1$ होती है ($3 \to 5 \to 7$)।

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 कर्नेलों के गहरे स्टैक एक बड़े कर्नेल से बेहतर क्यों हैं"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[पैरामीटर दक्षता और अरैखिक अभिव्यंजकता]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **वही $7 \times 7$ ग्रहणशील क्षेत्र:** एक के ऊपर एक रखी तीन $3 \times 3$ लेयरें ($r = 1 + 2 + 2 + 2 = 7$) ठीक वही $7 \times 7$ पैच देखती हैं जो एक $7 \times 7$ लेयर देखती है।

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% कम पैरामीटर:** $3 \times 9 c^2 = 27 c^2$ भार, न कि $49 c^2$, जब चैनल $c$ हों।

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **अधिक अभिव्यंजकता:** **3 अरैखिक ऐक्टिवेशन** एक के बाद एक लगते हैं, 1 नहीं (§{sec_blocks} में VGG का मूल डिज़ाइन नियम)।

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="सैद्धांतिक बनाम प्रभावी ग्रहणशील क्षेत्र"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[केंद्र से दूर संवेदनशीलता का गाउसियन ह्रास]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **सैद्धांतिक RF ($r = 2L+1$):** उन सभी पिक्सेलों का बाउंडिंग बॉक्स जिनसे आउटपुट इकाई तक *कम से कम एक* पथ है।

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **केंद्रीय सीमा प्रमेय:** केंद्र के पिक्सेल चरघातांकी रूप से अनेक पथों से जुड़ते हैं, जबकि कोने के पिक्सेल केवल **1 पथ** से।

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **प्रभावी RF (Luo et al., 2016):** इनपुट ग्रेडिएंट का परिमाण $\partial y / \partial \mathbf{X}$ गाउसियन घंटी-वक्र की तरह घटता है, जिसका मानक विचलन $\sigma \propto \sqrt{L}$ है।

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
आउटपुट शेप, `im2col` और ग्रहणशील क्षेत्र के अंकगणित की अपनी समझ परखें:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col आव्यूह के आयाम:** आप बिना पैडिंग का $3 \times 3$ एक-चैनल कर्नेल $10 \times 10$ इमेज पर `im2col` के ज़रिए लगाते हैं। आउटपुट का स्थानिक शेप क्या है, और `patches` आव्यूह तथा समतल किए गए कर्नेल सदिश के ठीक-ठीक आयाम क्या हैं?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST को ढकने लायक़ गहराई:** स्ट्राइड 1 पर एक के ऊपर एक कितनी $3 \times 3$ लेयरें चाहिए कि एक अकेली इकाई का सैद्धांतिक ग्रहणशील क्षेत्र पूरी $28 \times 28$ इमेज को ढक ले? और $5 \times 5$ लेयरों के साथ कितनी?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **एज लॉस और सीमा का खिसकना:** $6 \times 8$ के हमारे ऊर्ध्वाधर एज सीखने के प्रयोग में लॉस $\approx 12$ से क्यों शुरू हुआ, और यदि काली पट्टी दाएँ किनारे तक फैली होती (`X[:, 2:8] = 0`) तो आरंभिक लॉस क्या होता?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**उत्तर:**
(1) आउटपुट शेप $(10-3+1) \times (10-3+1) = 8 \times 8$ है ($64$ आउटपुट)। `patches` आव्यूह का शेप $(64, 9)$ है और वह $(9, 1)$ कर्नेल सदिश को गुणा करता है।
(2) $3 \times 3$ के लिए: $2L + 1 \ge 28 \implies L = 14$ लेयरें। $5 \times 5$ के लिए: $4L + 1 \ge 28 \implies L = 7$ लेयरें।
(3) 6 पंक्तियों में से हर एक में दो $\pm 1$ एज थे ($6 \times 2 \times 1^2 = 12$)। काली पट्टी को दाएँ किनारे तक बढ़ाने से दाईं ओर का काले-से-सफ़ेद एज हट जाता है, हर पंक्ति में एक एज बचता है और आरंभिक लॉस आधा होकर $\approx 6$ रह जाता है।

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="सारांश और मुख्य बातें"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[अनुभाग का सार]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2D क्रॉस-सहसंबंध और एज सीखना:** $\mathbf{K}$ को वैध विंडो पर खिसकाता है ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD $[1, -1]$ को 9 चरणों में सीख लेता है।

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col आव्यूह गुणन:** पैचों की $k_\textrm{h} k_\textrm{w}\times$ प्रतियों के बदले GPU GEMM का अधिकतम थ्रूपुट।

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **स्टैक के ग्रहणशील क्षेत्र:** $r = 1 + \sum (k_i - 1) = 2L+1$, जब स्टैक $3 \times 3$ का हो; केंद्र पर गाउसियन संवेदनशीलता।

=== DECISIONS ===
- "to flatten" → समतल करना; "flattened kernel" → समतल किया गया कर्नेल; "patch matrix" → पैच आव्यूह; "matrix product" → आव्यूह गुणनफल (the operation "matrix multiplication/multiply" stays आव्यूह गुणन).
- "to unroll / unfold" (kernel or input into a matrix) → खोलना; "banded matrix" → पट्टीदार आव्यूह, with (banded) on first use; "sparse" → विरल.
- "black band" (of the test image) → काली पट्टी; "cell" (of a figure) → खाना; "bars" (of a figure) → बार.
- "theoretical / effective receptive field" → सैद्धांतिक / प्रभावी ग्रहणशील क्षेत्र; the slide abbreviation "RF" stays RF.
- "path" → पथ; "side length" → भुजा की लंबाई; "magnitude" → परिमाण; "sensitivity" → संवेदनशीलता; "bell curve" → घंटी-वक्र; "Gaussian" (adjective) → गाउसियन.
- "trade-off" → समझौता (slide label) or "इसकी क़ीमत … है" in running prose; "workspace memory" → वर्कस्पेस मेमोरी; "workload" → वर्कलोड; "library" → लाइब्रेरी; "algorithm" → एल्गोरिद्म.
- "forward propagation" → फ़ॉरवर्ड पास (same glossary entry as "forward pass"); "backprop" → बैकप्रॉपैगेशन.
- "expressivity" → अभिव्यंजकता; "exponentially" → चरघातांकी रूप से; "linear growth" → रैखिक वृद्धि; "hierarchy" → पदानुक्रम.
- "Concept Check" → अवधारणा जाँच; "Answers" → उत्तर; "Summary & Key Takeaways" → सारांश और मुख्य बातें.

=== NOTES ===
- e7983df8: "Recall from …" is the verb (याद करें), not the metric "recall" (रीकॉल).
- 16976d18: "spans an entire image" and the title "Depth to span" are the everyday verb (ढकना), not the linear-algebra "span" (रैखिक विस्तार).
- 1fb18ec4: "at the cost of" is the idiom (क़ीमत), not the glossary's "cost" (लागत).
- f47d181e: "std" is written out as मानक विचलन; Hindi has no standard abbreviation.
- 1711f812: "(Image-to-Columns)" is translated (इमेज से स्तंभ); in 5bdffcd7 the English gloss "image to columns" is kept in quotes and followed by its Hindi meaning, since it explains the name im2col.
