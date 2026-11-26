<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## التمارين

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
1. [code] **نوى مصمّمة يدويًا.** صمّم بعض النوى بنفسك.
    1. إذا أُعطيت متجه اتجاه $\mathbf{v} = (v_1, v_2)$، فاشتق نواة لكشف الحواف تكشف الحواف المتعامدة مع $\mathbf{v}$، أي الحواف التي تمتد في الاتجاه $(v_2, -v_1)$.
    1. اشتق مؤثر الفروق المحدودة للمشتق الثاني. ما البنى في الصور التي تستجيب له بأقوى درجة؟
    1. ما أصغر حجم للنواة يسمح بالحصول على مشتق من الرتبة $d$؟
    1. كيف تصمّم نواة تمويه؟ ولماذا قد ترغب في استخدام نواة كهذه؟
    1. حمّل صورة فوتوغرافية، وحوّلها إلى قناة واحدة، وخفّض دقتها إلى نحو $100 \times 100$ بكسل (لأن `corr2d` يدور في حلقات Python). طبّق كاشف الحواف ونواة التمويه ونواة الشحذ «ضعف الهوية ناقص التمويه»، واعرض كل نتيجة بجانب الأصل. ما الذي تُظهره خريطة حواف الصورة الفوتوغرافية ولم يكن بوسع خريطة حواف الشريط الأسود أن تُظهره؟

    *مقتبس بتصرّف من Stanford CS231n،
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)،
    دفتر ConvolutionalNetworks.*

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
2. [code] **الالتفاف بوصفه مصفوفة شريطية.** مصفوفة الرقع في :numref:`subsec_conv_matmul` تبسط الدخل، أما البناء المقابل فيفرد النواة. ليكن $\mathbf{X}$ دخلًا $4 \times 4$، و $\mathbf{K}$ نواة $3 \times 3$، و $\textrm{vec}(\cdot)$ التسطيح صفًا بعد صف.
    1. أنشئ مصفوفة $4 \times 16$ هي $\mathbf{W}$ تحقق $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$، حيث $\mathbf{Y}$ هو الارتباط المتقاطع بين $\mathbf{X}$ و $\mathbf{K}$. كم عنصرًا غير صفري في كل صف، وكم قيمة مختلفة تظهر في $\mathbf{W}$؟
    1. ابنِ $\mathbf{W}$ برمجيًا من `K`، وتأكّد من أن الجداء يعيد إنتاج `corr2d(X, K)`.
    1. لدخل $n \times n$ ونواة $k \times k$، احسب عدد عناصر $\mathbf{W}$ إذا خُزّنت مصفوفةً كثيفة، وعدد عناصر $\mathbf{W}$ غير الصفرية، وعدد عناصر مصفوفة الرقع. كيف ينمو كل عدد من هذه الأعداد مع $n$؟
    1. احتفظ بنمط التناثر في $\mathbf{W}$، لكن أعطِ كل صف قيمه الخاصة البالغ عددها $k^2$. النتيجة طبقة متصلة محليًا: إنها الموتر رباعي الرتبة $\mathsf{V}$ من :numref:`sec_why-conv` في $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$، مقصورًا على نافذة $k \times k$ من الإزاحات $(a, b)$، قبل أن يربط تساوق الإزاحة الأوزان ببعضها. لدخل $n \times n$، احسب عدد معاملات هذه الطبقة، وعدد معاملات طبقة كاملة الاتصال لها أشكال الدخل والخرج نفسها، وعدد معاملات الالتفاف.

    *مقتبس بتصرّف من CMU 11-785،
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **مجال الاستقبال الفعّال.** ● يحدّد مجال الاستقبال النظري $r$ في هذا القسم المدخلاتِ التي *يمكن* أن تؤثر في خرج ما؛ أما مجال الاستقبال الفعّال فيقيس مقدار تأثيرها الفعلي :cite:`Luo.Li.Urtasun.ea.2016`. ابنِ كومة من $L$ طبقة التفافية $3 \times 3$ أحادية القناة بلا حشو وبلا انحياز ومن دون أي لاخطية، على دخل $32 \times 32$، مستخدمًا الطبقة المدمجة من :numref:`subsec_learning_kernel`. اسحب كل عنصر من عناصر النوى مستقلًا من التوزيع الغاوسي المعياري.
    1. لـ $L = 3$ وسحبة واحدة للنوى، انشر عكسيًا تدرجًا واحديًا من وحدة خرج قرب مركز الطبقة الأخيرة إلى الدخل، وارسم مقدار تدرج الدخل. قارن حامله (support) بمجال الاستقبال النظري $7 \times 7$، وبيّن أين يكون المقدار أكبر ما يمكن.
    1. احسب متوسط مربع تدرج الدخل على 500 سحبة للنوى، لـ $L = 3$ ولـ $L = 10$. اجمع كل خريطة متوسطة على صفوفها، وطبّع ملف الأعمدة الناتج (column profile) بحيث يساوي مجموعه واحدًا. كيف تتغير نسبة الانحراف المعياري للملف إلى طول الضلع النظري $2L + 1$ مع العمق؟
    1. يعدّ :numref:`fig_conv_rf_paths` المسارات لـ $L = 3$ في بعد واحد. بيّن أن مربع تدرج الدخل المتوقع، مجموعًا على الصفوف، يتناسب لأي $L$ مع التفاف $L$ نسخة من المتجه $(1, 1, 1)$. ثم اشتق كيف ينمو الانحراف المعياري للملف المطبَّع مع $L$.

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
4. [code] **التمرير الخلفي للالتفاف.** ● ليكن $\mathbf{Y}$ الارتباط المتقاطع أحادي القناة بين دخل $\mathbf{X}$ ونواة $\mathbf{K}$، كما تحسبه `corr2d`، وليكن $\ell$ خسارة عددية لا تعتمد على $\mathbf{X}$ و $\mathbf{K}$ إلا من خلال $\mathbf{Y}$.
    1. اشتق $\partial \ell/\partial \mathbf{K}$ و $\partial \ell/\partial \mathbf{X}$ بدلالة $\partial \ell/\partial \mathbf{Y}$، على غرار معادلات الانتشار العكسي في :numref:`sec_backprop`.
    1. نفّذ تعبيرك لـ $\partial \ell/\partial \mathbf{K}$ من دون الاشتقاق التلقائي للخسارة $\ell = \sum_{i,j} G_{ij} Y_{ij}$، حيث $\mathbf{G}$ مصفوفة عشوائية ثابتة لها شكل $\mathbf{Y}$. قارنه بتدرج محسوب بالفروق المحدودة المركزية بحجم الخطوة $10^{-2}$ على `X` و `K` عشوائيتين صغيرتين، وأبلغ عن أكبر خطأ نسبي.
    1. تحقّق من تعبيرك لـ $\partial \ell/\partial \mathbf{K}$ بمقارنته بتدرج النواة الذي يحسبه الاشتقاق التلقائي في تجربة :numref:`subsec_learning_kernel`.

    *مقتبس بتصرّف من Michael Nielsen،
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)،
    الفصل 6، ومن Stanford CS231n،
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)،
    دفتر ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[المناقشات](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[المناقشات](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[المناقشات](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[المناقشات](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Exercises: the `[code]` tag and the ● marker are copied unchanged; the bold title keeps its full stop inside the bold; sub-items keep the English list structure and indentation.
- "Adapted from X, [title](url), …" → *مقتبس بتصرّف من X، [title](url)، …*; course names, assignment titles and notebook names stay in English (Stanford CS231n, Assignment 2, ConvolutionalNetworks, CMU 11-785); "notebook" → دفتر; "Chapter 6" → الفصل 6.
- "Discussions" (link text) → المناقشات.
- scalar (adjective: scalar loss, scalar bias) → عددي/عددية.
- standard normal distribution → التوزيع الغاوسي المعياري (glossary: Gaussian distribution).
- draw (a random sample) → سحبة / اسحب; unit gradient → تدرج واحدي; support (of a function) → الحامل (support); column profile → ملف الأعمدة.
- blur kernel → نواة التمويه; sharpening kernel → نواة الشحذ; locally connected layer → طبقة متصلة محليًا.

=== NOTES ===
- Glossary warnings "kernels -> نواة الالتفاف" are false matches: the plural of نواة is النوى / نوى, which the check does not recognise as the glossary term.
