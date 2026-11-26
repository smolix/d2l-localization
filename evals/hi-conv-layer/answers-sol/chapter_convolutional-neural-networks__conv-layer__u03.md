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
1. [code] **हाथ से बनाए गए कर्नेल।** कुछ कर्नेल हाथ से बनाएँ।
    1. दिए गए दिशा सदिश $\mathbf{v} = (v_1, v_2)$ के लिए ऐसा किनारा संसूचक कर्नेल निकालें जो $\mathbf{v}$ के लांबिक किनारों का पता लगाए, यानी $(v_2, -v_1)$ दिशा वाले किनारों का।
    1. द्वितीय अवकलज के लिए एक परिमित अंतर संकारक निकालें। छवियों की कौन-सी संरचनाएँ इस पर सबसे तीव्र प्रतिक्रिया देती हैं?
    1. कोटि $d$ का अवकलज निकालने के लिए कर्नेल का न्यूनतम आकार क्या है?
    1. छवि को धुंधला करने वाला कर्नेल कैसे बनाएँगे? ऐसे कर्नेल का इस्तेमाल क्यों करना चाहेंगे?
    1. एक फ़ोटो लोड करें, उसे एक चैनल में बदलें और डाउनसैंपल करके लगभग $100 \times 100$ पिक्सेल का बना दें (`corr2d` के लूप Python में चलते हैं)। किनारा संसूचक, धुंधला करने वाला कर्नेल और छवि की तीक्ष्णता बढ़ाने वाला कर्नेल (तत्समक के दोगुने में से धुंधला करने वाला कर्नेल घटाकर बना) लगाएँ। हर परिणाम को मूल फ़ोटो के बगल में दिखाएँ। फ़ोटो के किनारा मानचित्र में ऐसा क्या दिखाई देता है जो काली पट्टी के किनारा मानचित्र में नहीं दिख सकता था?

    *Stanford CS231n के [असाइनमेंट 2](https://cs231n.github.io/assignments2024/assignment2/) की ConvolutionalNetworks नोटबुक से रूपांतरित।*

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
2. [code] **पट्टीदार आव्यूह के रूप में कन्वोल्यूशन।** :numref:`subsec_conv_matmul` का पैच आव्यूह इनपुट को खोलता है। उलटी रचना में कर्नेल को खोलते हैं। मान लें कि $\mathbf{X}$ एक $4 \times 4$ का इनपुट है, $\mathbf{K}$ एक $3 \times 3$ का कर्नेल है और $\textrm{vec}(\cdot)$ पंक्ति दर पंक्ति समतल करने की संक्रिया है।
    1. $4 \times 16$ का ऐसा आव्यूह $\mathbf{W}$ बनाएँ कि $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ हो, जहाँ $\mathbf{Y}$, $\mathbf{X}$ और $\mathbf{K}$ का क्रॉस-सहसंबंध है। हर पंक्ति में कितनी अशून्य प्रविष्टियाँ हैं, और $\mathbf{W}$ में कितने अलग-अलग मान आते हैं?
    1. कोड में $\mathbf{W}$ को `K` से बनाएँ और पुष्टि करें कि गुणनफल से `corr2d(X, K)` वाला परिणाम मिलता है।
    1. $n \times n$ के इनपुट और $k \times k$ के कर्नेल के लिए तीन गिनतियाँ निकालें। सघन ऐरे के रूप में संग्रहीत $\mathbf{W}$ में कुल कितनी प्रविष्टियाँ हैं, $\mathbf{W}$ में कितनी अशून्य प्रविष्टियाँ हैं और पैच आव्यूह में कितनी प्रविष्टियाँ हैं? हर गिनती $n$ के साथ कैसे बढ़ती है?
    1. $\mathbf{W}$ में विरलता का पैटर्न बनाए रखें, लेकिन हर पंक्ति को उसके अपने $k^2$ मान दें। इससे एक स्थानीय रूप से जुड़ी लेयर मिलती है। यह चौथी कोटि का वही टेंसर $\mathsf{V}$ है जो :numref:`sec_why-conv` के समीकरण $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ में है, जब उसे ऑफ़सेट $(a, b)$ की $k \times k$ विंडो तक सीमित कर दें और अभी स्थानांतरण समचरता के अनुसार भार साझा न किए गए हों। $n \times n$ के इनपुट के लिए इस लेयर, उसी इनपुट और आउटपुट शेप वाली फ़ुली कनेक्टेड लेयर और कन्वोल्यूशन, तीनों के पैरामीटर गिनें।

    *CMU 11-785 के [गृहकार्य 2, भाग 1](https://deeplearning.cs.cmu.edu/S21/index.html) से रूपांतरित।*

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
3. [code] **प्रभावी ग्रहणशील क्षेत्र।** ● इस खंड का सैद्धांतिक ग्रहणशील क्षेत्र $r$ तय करता है कि कौन-से इनपुट आउटपुट को प्रभावित कर *सकते हैं*। प्रभावी ग्रहणशील क्षेत्र मापता है कि वे वास्तव में कितना प्रभाव डालते हैं :cite:`Luo.Li.Urtasun.ea.2016`। एक के ऊपर एक रखी $L$ एक-चैनल $3 \times 3$ की कन्वोल्यूशनल लेयरें बनाएँ। पैडिंग, बायस और अरैखिकता के बिना इन्हें $32 \times 32$ के इनपुट पर लगाएँ। इसके लिए :numref:`subsec_learning_kernel` की अंतर्निर्मित लेयर इस्तेमाल करें। कर्नेल की हर प्रविष्टि का मान स्वतंत्र रूप से मानक गाउसियन वितरण से चुनें।
    1. $L = 3$ के लिए कर्नेलों के मान एक बार चुनें। अंतिम लेयर के केंद्र के पास किसी आउटपुट इकाई से मान 1 का ग्रेडिएंट बैकप्रॉपैगेशन से इनपुट तक पहुँचाएँ और इनपुट ग्रेडिएंट के परिमाण का ग्राफ़ बनाएँ। उन स्थानों की तुलना, जहाँ ग्रेडिएंट अशून्य है, $7 \times 7$ के सैद्धांतिक ग्रहणशील क्षेत्र से करें और बताएँ कि परिमाण कहाँ सबसे बड़ा है।
    1. कर्नेलों के मान 500 बार चुनकर वर्गित इनपुट ग्रेडिएंट का माध्य निकालें, $L = 3$ और $L = 10$ दोनों के लिए। हर माध्य मानचित्र की पंक्तियों का योग लें और इस तरह बनी स्तंभ प्रोफ़ाइल को सामान्यीकृत करें ताकि उसके मानों का योग एक हो। गहराई के साथ प्रोफ़ाइल के मानक विचलन और सैद्धांतिक भुजा की लंबाई $2L + 1$ का अनुपात कैसे बदलता है?
    1. :numref:`fig_conv_rf_paths` में $L = 3$ के लिए एक आयाम में पथों की गिनती है। दिखाएँ कि सामान्य $L$ के लिए वर्गित इनपुट ग्रेडिएंट का अपेक्षित मान, पंक्तियों पर योग लेने के बाद, सदिश $(1, 1, 1)$ की $L$ प्रतियों के कन्वोल्यूशन के समानुपाती है। निकालें कि सामान्यीकृत प्रोफ़ाइल का मानक विचलन $L$ के साथ कैसे बढ़ता है।

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
4. [code] **कन्वोल्यूशन का बैकवर्ड पास।** ● मान लें कि $\mathbf{Y}$ इनपुट $\mathbf{X}$ और कर्नेल $\mathbf{K}$ का एक-चैनल क्रॉस-सहसंबंध है, जैसा कि `corr2d` से निकाला जाता है। मान लें कि $\ell$ एक अदिश लॉस है जो $\mathbf{X}$ और $\mathbf{K}$ पर केवल $\mathbf{Y}$ के ज़रिए निर्भर है।
    1. $\partial \ell/\partial \mathbf{K}$ और $\partial \ell/\partial \mathbf{X}$ को $\partial \ell/\partial \mathbf{Y}$ के पदों में निकालें। इसके लिए :numref:`sec_backprop` के बैकप्रॉपैगेशन समीकरणों की शैली अपनाएँ।
    1. $\partial \ell/\partial \mathbf{K}$ के लिए अपनी अभिव्यक्ति को स्वचालित अवकलन के बिना लॉस $\ell = \sum_{i,j} G_{ij} Y_{ij}$ के लिए लागू करें, जहाँ $\mathbf{G}$ एक नियत यादृच्छिक आव्यूह है जिसकी शेप $\mathbf{Y}$ जैसी है। चरण आकार $10^{-2}$ लेकर छोटे यादृच्छिक `X` और `K` पर केंद्रीय परिमित अंतर से ग्रेडिएंट निकालें। दोनों की तुलना करें और सबसे बड़ी सापेक्ष त्रुटि बताएँ।
    1. $\partial \ell/\partial \mathbf{K}$ के लिए अपनी अभिव्यक्ति की तुलना उस कर्नेल ग्रेडिएंट से करें जो :numref:`subsec_learning_kernel` के प्रयोग में स्वचालित अवकलन से मिलता है।

    *Michael Nielsen की [न्यूरल नेटवर्क और डीप लर्निंग](http://neuralnetworksanddeeplearning.com/chap6.html) के अध्याय 6 और Stanford CS231n के [असाइनमेंट 2](https://cs231n.github.io/assignments2024/assignment2/) की ConvolutionalNetworks नोटबुक से रूपांतरित।*

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
- “blur kernel” → “धुंधला करने वाला कर्नेल”; “sharpening kernel” → “छवि की तीक्ष्णता बढ़ाने वाला कर्नेल”; “identity” (operator) → “तत्समक”; “operator” → “संकारक”; “downsample” → “डाउनसैंपल करना”; “locally connected layer” → “स्थानीय रूप से जुड़ी लेयर”; “normalize” → “सामान्यीकृत करना”; “support” (gradient) → “वे स्थान जहाँ ग्रेडिएंट अशून्य है”; “relative error” → “सापेक्ष त्रुटि”.

=== NOTES ===
- ec8621f8: Put the offsets before the window dimensions to keep the Hindi noun phrase together.
- 54bcee26: Put the vector before the number of copies in the corresponding Hindi noun phrase.
