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
1. [code] **हाथ से डिज़ाइन किए गए कर्नेल।** कुछ कर्नेल मैन्युअल रूप से डिज़ाइन करें।
    1. एक दिशात्मक सदिश $\mathbf{v} = (v_1, v_2)$ दिए जाने पर, एक ऐसा एज-डिटेक्शन कर्नेल व्युत्पन्न करें जो $\mathbf{v}$ के लांबिक किनारों, अर्थात् $(v_2, -v_1)$ दिशा में किनारों का पता लगाए।
    1. द्वितीय अवकलज के लिए एक परिमित अंतर ऑपरेटर व्युत्पन्न करें। छवियों में कौन-सी संरचनाएँ इसके प्रति सबसे प्रबल प्रतिक्रिया करती हैं?
    1. कोटि $d$ का अवकलज प्राप्त करने के लिए कर्नेल का न्यूनतम आकार क्या है?
    1. आप एक ब्लर (blur) कर्नेल कैसे डिज़ाइन करेंगे? आप ऐसे कर्नेल का उपयोग क्यों करना चाहेंगे?
    1. एक फ़ोटो लोड करें, उसे एक चैनल में बदलें, और लगभग $100 \times 100$ पिक्सेल तक डाउनसैंपल करें (Python में `corr2d` लूप)। एज संसूचक, ब्लर कर्नेल, और शार्पनिंग कर्नेल "पहचान (identity) का दोगुना घटा ब्लर" लागू करें, और प्रत्येक परिणाम को मूल छवि के बगल में प्रदर्शित करें। किसी फ़ोटो का एज मैप ऐसा क्या दिखाता है जो काली पट्टी का एज मैप नहीं दिखा सका?

    *Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
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
2. [code] **बैंडेड आव्यूह के रूप में कन्वोल्यूशन।** :numref:`subsec_conv_matmul` का पैच आव्यूह इनपुट को खोलता है। इसके विपरीत निर्माण कर्नेल को खोलता है। मान लें कि $\mathbf{X}$ एक $4 \times 4$ इनपुट है, $\mathbf{K}$ एक $3 \times 3$ कर्नेल है, और $\textrm{vec}(\cdot)$ पंक्ति-दर-पंक्ति समतलन (flattening) है।
    1. $4 \times 16$ आव्यूह $\mathbf{W}$ का निर्माण करें जहाँ $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ हो, और $\mathbf{Y}$, $\mathbf{X}$ का $\mathbf{K}$ के साथ क्रॉस-सहसंबंध है। प्रत्येक पंक्ति में कितनी शून्येतर (nonzero) प्रविष्टियाँ हैं, और $\mathbf{W}$ में कितने भिन्न मान आते हैं?
    1. कोड में `K` से $\mathbf{W}$ बनाएँ और पुष्टि करें कि गुणनफल `corr2d(X, K)` को पुनरुत्पादित करता है।
    1. $n \times n$ इनपुट और $k \times k$ कर्नेल के लिए, डेंस ऐरे के रूप में संग्रहीत $\mathbf{W}$ की प्रविष्टियों, $\mathbf{W}$ की शून्येतर प्रविष्टियों, और पैच आव्यूह की प्रविष्टियों की गिनती करें। प्रत्येक गिनती $n$ के साथ कैसे बढ़ती है?
    1. $\mathbf{W}$ के विरलता पैटर्न को बनाए रखें लेकिन प्रत्येक पंक्ति को उसके अपने $k^2$ मान दें। इसका परिणाम एक स्थानीय रूप से संयोजित (locally connected) लेयर है: स्थानांतरण समचरता द्वारा भारों को बाँधने से पहले ऑफ़सेट $(a, b)$ की $k \times k$ विंडो तक सीमित, $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ में :numref:`sec_why-conv` का चतुर्थ-कोटि टेंसर $\mathsf{V}$। $n \times n$ इनपुट के लिए, इस लेयर के, समान इनपुट और आउटपुट शेप वाली फ़ुली कनेक्टेड लेयर के, और कन्वोल्यूशन के पैरामीटरों की गिनती करें।

    *CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html) से अनुकूलित।*

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
3. [code] **प्रभावी ग्रहणशील क्षेत्र।** ● इस अनुभाग का सैद्धांतिक ग्रहणशील क्षेत्र $r$ यह सीमा तय करता है कि कौन-से इनपुट किसी आउटपुट को प्रभावित *कर सकते हैं*; प्रभावी ग्रहणशील क्षेत्र यह मापता है कि वे वास्तव में कितना प्रभाव डालते हैं :cite:`Luo.Li.Urtasun.ea.2016`। :numref:`subsec_learning_kernel` की अंतर्निहित लेयर का उपयोग करते हुए, $32 \times 32$ इनपुट पर बिना पैडिंग, बिना बायस, और बिना अरैखिकता के $L$ एकल-चैनल $3 \times 3$ कन्वोल्यूशनल लेयरों का स्टैक बनाएँ। प्रत्येक कर्नेल प्रविष्टि को एक मानक गाउसियन वितरण से स्वतंत्र रूप से चुनें।
    1. $L = 3$ और कर्नेल के एकल नमूने के लिए, अंतिम लेयर के केंद्र के पास की एक आउटपुट इकाई से इनपुट तक एक इकाई ग्रेडिएंट को बैकप्रॉपैगेट करें और इनपुट ग्रेडिएंट का परिमाण प्लॉट करें। इसके आधार (support) की तुलना $7 \times 7$ सैद्धांतिक ग्रहणशील क्षेत्र से करें, और बताएँ कि परिमाण कहाँ सबसे बड़ा है।
    1. $L = 3$ और $L = 10$ के लिए, कर्नेल के 500 नमूनों पर वर्ग इनपुट ग्रेडिएंट का औसत निकालें। प्रत्येक औसत मैप का उसकी पंक्तियों पर योग करें और परिणामी स्तंभ प्रोफ़ाइल को 1 के योग तक सामान्यीकृत (normalize) करें। गहराई के साथ प्रोफ़ाइल के मानक विचलन का सैद्धांतिक भुजा लंबाई $2L + 1$ से अनुपात कैसे बदलता है?
    1. :numref:`fig_conv_rf_paths` एक विमा में $L = 3$ के लिए पथों की गिनती करता है। दिखाएँ कि सामान्य $L$ के लिए, पंक्तियों पर योग किया गया प्रत्याशित वर्ग इनपुट ग्रेडिएंट सदिश $(1, 1, 1)$ की $L$ प्रतियों के कन्वोल्यूशन के समानुपाती होता है। व्युत्पन्न करें कि सामान्यीकृत प्रोफ़ाइल का मानक विचलन $L$ के साथ कैसे बढ़ता है।

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
4. [code] **कन्वोल्यूशन का बैकवर्ड पास।** ● मान लें कि $\mathbf{Y}$ एक इनपुट $\mathbf{X}$ का कर्नेल $\mathbf{K}$ के साथ एकल-चैनल क्रॉस-सहसंबंध है, जैसा कि `corr2d` द्वारा संगणित किया गया है, और मान लें कि $\ell$ एक अदिश लॉस है जो $\mathbf{X}$ और $\mathbf{K}$ पर केवल $\mathbf{Y}$ के माध्यम से निर्भर करता है।
    1. :numref:`sec_backprop` के बैकप्रॉपैगेशन समीकरणों की शैली में, $\partial \ell/\partial \mathbf{Y}$ के पदों में $\partial \ell/\partial \mathbf{K}$ और $\partial \ell/\partial \mathbf{X}$ व्युत्पन्न करें।
    1. लॉस $\ell = \sum_{i,j} G_{ij} Y_{ij}$ के लिए स्वचालित अवकलन के बिना $\partial \ell/\partial \mathbf{K}$ के लिए अपना व्यंजक लागू करें, जहाँ $\mathbf{G}$, $\mathbf{Y}$ के शेप का एक निश्चित यादृच्छिक आव्यूह है। एक छोटे यादृच्छिक `X` और `K` पर $10^{-2}$ चरण आकार वाले केंद्रित परिमित-अंतर ग्रेडिएंट से इसकी तुलना करें, और सबसे बड़ी सापेक्ष त्रुटि रिपोर्ट करें।
    1. :numref:`subsec_learning_kernel` के प्रयोग में स्वचालित अवकलन द्वारा संगणित कर्नेल ग्रेडिएंट के विरुद्ध $\partial \ell/\partial \mathbf{K}$ के लिए अपने व्यंजक की जाँच करें।

    *Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    अध्याय 6, और Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
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
- Carried forward all u01 and u02 decisions.
- "Discussions" (tab_note) → "चर्चा"
- "standard normal distribution" → "मानक गाउसियन वितरण" (aligned with glossary "गाउसियन वितरण")
- "support" (of gradient) → "आधार" (support)
- "relative error" → "सापेक्ष त्रुटि"
- "nonzero" → "शून्येतर"

=== NOTES ===
- ec8621f8, 54bcee26, 6ab5b67c: math tokens reordered to match Hindi SOV clause order.
