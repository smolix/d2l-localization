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
1. [code] **हाथ से बनाए कर्नेल।** कुछ कर्नेल हाथ से डिज़ाइन करें।
    1. दिशा सदिश $\mathbf{v} = (v_1, v_2)$ दिया हो, तो ऐसा
       एज डिटेक्शन कर्नेल व्युत्पन्न करें जो $\mathbf{v}$ के लांबिक एज
       पकड़े, यानी $(v_2, -v_1)$ दिशा वाले एज।
    1. द्वितीय अवकलज के लिए एक परिमित अंतर संकारक (operator) व्युत्पन्न
       करें। इमेज की कौन-सी संरचनाएँ उस पर सबसे तीव्र प्रतिक्रिया देती हैं?
    1. कोटि $d$ का अवकलज पाने के लिए कर्नेल का न्यूनतम आकार क्या है?
    1. आप धुँधला करने वाला (blur) कर्नेल कैसे डिज़ाइन करेंगे? ऐसा कर्नेल
       आप क्यों इस्तेमाल करना चाहेंगे?
    1. एक फ़ोटो लोड करें, उसे एक चैनल में बदलें और घटाकर लगभग
       $100 \times 100$ पिक्सेल का कर दें (`corr2d` Python में लूप चलाता
       है)। उस पर एज डिटेक्टर, ब्लर कर्नेल और शार्पनिंग कर्नेल "पहचान
       (identity) का दोगुना ऋण ब्लर" लगाएँ, और हर परिणाम को मूल के बगल
       में दिखाएँ। फ़ोटो का एज मैप ऐसा क्या दिखाता है जो काली पट्टी का
       एज मैप नहीं दिखा सकता था?

    *Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook से रूपांतरित।*

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
2. [code] **पट्टीदार आव्यूह के रूप में कन्वोल्यूशन।**
   :numref:`subsec_conv_matmul` का पैच आव्यूह इनपुट को खोलता है। इसका
   उलटा निर्माण कर्नेल को खोलता है। मान लें $\mathbf{X}$ एक $4 \times 4$
   इनपुट है, $\mathbf{K}$ एक $3 \times 3$ कर्नेल, और $\textrm{vec}(\cdot)$
   पंक्ति-दर-पंक्ति समतल करने की संक्रिया।
    1. वह $4 \times 16$ आव्यूह $\mathbf{W}$ बनाएँ जिसके लिए
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$
       हो, जहाँ $\mathbf{Y}$ वह क्रॉस-सहसंबंध है जो $\mathbf{X}$ का
       $\mathbf{K}$ के साथ बनता है। हर पंक्ति में कितनी शून्येतर
       प्रविष्टियाँ हैं, और $\mathbf{W}$ में कितने अलग-अलग मान आते हैं?
    1. $\mathbf{W}$ को कोड में `K` से बनाएँ और पुष्टि करें कि गुणनफल
       `corr2d(X, K)` को दोहराता है।
    1. $n \times n$ इनपुट और $k \times k$ कर्नेल के लिए गिनें: डेंस ऐरे
       के रूप में रखे गए $\mathbf{W}$ की प्रविष्टियाँ, $\mathbf{W}$ की
       शून्येतर प्रविष्टियाँ, और पैच आव्यूह की प्रविष्टियाँ। हर गिनती
       $n$ के साथ कैसे बढ़ती है?
    1. $\mathbf{W}$ का विरलता पैटर्न वही रखें, पर हर पंक्ति को उसके अपने
       $k^2$ मान दें। परिणाम एक स्थानीय रूप से जुड़ी (locally connected)
       लेयर है: यह चतुर्थ-कोटि टेंसर $\mathsf{V}$ है, जो
       :numref:`sec_why-conv` के
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$
       में आता है, $k \times k$ विंडो के ऑफ़सेट $(a, b)$ तक सीमित, और
       उस अवस्था में जब स्थानांतरण समचरता ने अभी भारों को आपस में बाँधा
       नहीं है। $n \times n$ इनपुट के लिए इस लेयर के, उन्हीं इनपुट और
       आउटपुट शेप वाली फ़ुली कनेक्टेड लेयर के, और कन्वोल्यूशन के
       पैरामीटर गिनें।

    *CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html)
    से रूपांतरित।*

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
3. [code] **प्रभावी ग्रहणशील क्षेत्र।** ● इस अनुभाग का सैद्धांतिक
   ग्रहणशील क्षेत्र $r$ यह सीमा बाँधता है कि कौन-से इनपुट किसी आउटपुट
   को प्रभावित *कर सकते हैं*; प्रभावी ग्रहणशील क्षेत्र मापता है कि वे
   वास्तव में कितना प्रभावित करते हैं :cite:`Luo.Li.Urtasun.ea.2016`।
   $L$ एक-चैनल $3 \times 3$ कन्वोल्यूशनल लेयरों का स्टैक बनाएँ, जिसमें न
   पैडिंग हो, न बायस, न अरैखिकता; इनपुट $32 \times 32$ हो, और इसके लिए
   :numref:`subsec_learning_kernel` की बनी-बनाई लेयर लें। कर्नेल की हर
   प्रविष्टि मानक गाउसियन वितरण (standard normal distribution) से
   स्वतंत्र रूप से लें।
    1. $L = 3$ और कर्नेलों के एक ही नमूने के लिए, अंतिम लेयर के केंद्र
       के पास की एक आउटपुट इकाई से इनपुट तक इकाई ग्रेडिएंट बैकप्रॉपैगेट
       करें और इनपुट ग्रेडिएंट के परिमाण का प्लॉट बनाएँ। उसके आलंब
       (support) की तुलना $7 \times 7$ सैद्धांतिक ग्रहणशील क्षेत्र से
       करें, और बताएँ कि परिमाण कहाँ सबसे बड़ा है।
    1. कर्नेलों के 500 नमूनों पर इनपुट ग्रेडिएंट के वर्ग का औसत लें,
       $L = 3$ के लिए और $L = 10$ के लिए। हर औसत मैप का उसकी पंक्तियों
       पर योग करें और मिलने वाले स्तंभ प्रोफ़ाइल को इस तरह
       प्रसामान्यीकृत करें कि उसका योग एक हो। प्रोफ़ाइल के मानक विचलन
       और सैद्धांतिक भुजा की लंबाई $2L + 1$ का अनुपात गहराई के साथ कैसे
       बदलता है?
    1. :numref:`fig_conv_rf_paths` एक आयाम में $L = 3$ के लिए पथ गिनता
       है। दिखाएँ कि व्यापक $L$ के लिए इनपुट ग्रेडिएंट के वर्ग का
       प्रत्याशित मान, पंक्तियों पर योग करने के बाद, $L$ बार लिए गए सदिश
       $(1, 1, 1)$ के आपसी कन्वोल्यूशन के समानुपाती है। व्युत्पन्न करें
       कि प्रसामान्यीकृत प्रोफ़ाइल का मानक विचलन $L$ के साथ कैसे बढ़ता
       है।

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
4. [code] **कन्वोल्यूशन का बैकवर्ड पास।** ● मान लें $\mathbf{Y}$ किसी
   इनपुट $\mathbf{X}$ का कर्नेल $\mathbf{K}$ के साथ एक-चैनल
   क्रॉस-सहसंबंध है, जैसा `corr2d` निकालता है, और $\ell$ एक अदिश लॉस है
   जो $\mathbf{X}$ और $\mathbf{K}$ पर केवल $\mathbf{Y}$ के माध्यम से
   निर्भर है।
    1. $\partial \ell/\partial \mathbf{K}$ और
       $\partial \ell/\partial \mathbf{X}$ को
       $\partial \ell/\partial \mathbf{Y}$ के पदों में व्युत्पन्न करें,
       उसी शैली में जो :numref:`sec_backprop` के बैकप्रॉपैगेशन समीकरणों
       की है।
    1. $\partial \ell/\partial \mathbf{K}$ के अपने व्यंजक को स्वचालित
       अवकलन के बिना लॉस $\ell = \sum_{i,j} G_{ij} Y_{ij}$ के लिए लागू
       करें, जहाँ $\mathbf{G}$ एक नियत यादृच्छिक आव्यूह है जिसका शेप
       $\mathbf{Y}$ जैसा है। इसकी तुलना चरण आकार $10^{-2}$ वाले केंद्रित
       परिमित अंतर ग्रेडिएंट से, छोटे यादृच्छिक `X` और `K` पर, करें और
       सबसे बड़ी सापेक्ष त्रुटि बताएँ।
    1. $\partial \ell/\partial \mathbf{K}$ के अपने व्यंजक को उस कर्नेल
       ग्रेडिएंट से मिलाकर जाँचें जिसे स्वचालित अवकलन
       :numref:`subsec_learning_kernel` के प्रयोग में निकालता है।

    *Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    अध्याय 6, और Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook से रूपांतरित।*

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
- Exercises: imperatives in the आप-form (करें, बनाएँ, दिखाएँ); the bold exercise title ends with a danda inside the bold; the `[code]` tag and the ● difficulty mark are copied unchanged.
- Source credits: "Adapted from X" → "X से रूपांतरित।"; titles of courses, assignments, books and notebooks stay in English (Assignment 2, Homework 2 Part 1, Neural Networks and Deep Learning), "Chapter 6" → अध्याय 6.
- "Discussions" (forum link) → चर्चा.
- "blur kernel" → ब्लर कर्नेल, introduced as धुँधला करने वाला (blur) कर्नेल; "sharpening kernel" → शार्पनिंग कर्नेल; "identity" (kernel) → पहचान (identity).
- "nonzero" → शून्येतर; "dense array" → डेंस ऐरे; "sparsity pattern" → विरलता पैटर्न; "locally connected layer" → स्थानीय रूप से जुड़ी (locally connected) लेयर; "fourth-order tensor" → चतुर्थ-कोटि टेंसर; "derivative of order d" → कोटि d का अवकलज.
- "standard normal distribution" → मानक गाउसियन वितरण (the glossary gives one term, गाउसियन वितरण, for Gaussian/normal).
- "a draw" (of random kernels) → नमूना; "expected" → प्रत्याशित; "to normalize" → प्रसामान्यीकृत करना; "proportional to" → के समानुपाती; "support" (of a function) → आलंब (support); "profile" → प्रोफ़ाइल.
- "expression" (formula) → व्यंजक; "operator" → संकारक (operator); "relative error" → सापेक्ष त्रुटि; "centered finite-difference gradient" → केंद्रित परिमित अंतर ग्रेडिएंट; "to backpropagate" → बैकप्रॉपैगेट करना.

=== NOTES ===
- 6f4ddec6: "(`corr2d` loops in Python)" is read as the reason for downsampling (the function runs Python loops, so it is slow) and translated that way.
- 54bcee26: the sentence that sets up the stack is split with a semicolon for Hindi syntax; the protected tokens keep the English order.
- 6f4ddec6, ec8621f8, 6ab5b67c: the link texts "Assignment 2", "Homework 2 Part 1" and "Neural Networks and Deep Learning" are titles and are left in English.
