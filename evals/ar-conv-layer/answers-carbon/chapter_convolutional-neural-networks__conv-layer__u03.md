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
1. [code] **نوى مصمَّمة يدويًا.** صمِّم بعض النوى يدويًا.
    1. بإعطاء متجه اتجاهي $\mathbf{v} = (v_1, v_2)$، اشتق نواة لكشف الحواف تكشف الحواف المتعامدة على $\mathbf{v}$، أي الحواف الممتدة في الاتجاه $(v_2, -v_1)$.
    1. اشتق مؤثر الفروق المحدودة للمشتق الثاني. ما البنى في الصور التي تستجيب له بأكبر قوة؟
    1. ما الحجم الأدنى للنواة للحصول على مشتق من الرتبة $d$؟
    1. كيف تصمِّم نواة تمويه؟ ولماذا قد ترغب في استخدام مثل هذه النواة؟
    1. حمِّل صورة فوتوغرافية، وحوِّلها إلى قناة واحدة، واخفض دقتها إلى نحو $100 \times 100$ من البكسلات (حلقات `corr2d` مكتوبة بلغة Python). طبِّق كاشف الحواف، ونواة التمويه، ونواة زيادة الحدة "ضعف المحايد ناقص التمويه"، واعرض كل نتيجة بجانب الصورة الأصلية. ما الذي تُظهره خريطة الحواف للصورة الفوتوغرافية ولم تكن خريطة الحواف للشريط الأسود قادرةً على إظهاره؟

    *مقتبس من مقرر Stanford CS231n، [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/)، دفتر ConvolutionalNetworks.*

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
2. [code] **الالتفاف بوصفه مصفوفة شريطية.** تبسط مصفوفة الرقع في :numref:`subsec_conv_matmul` المدخل. أما البناء المعاكس فيبسط النواة. ليكن $\mathbf{X}$ مدخلًا بحجم $4 \times 4$، و $\mathbf{K}$ نواة بحجم $3 \times 3$، وليكن $\textrm{vec}(\cdot)$ التسطيح صفًا بصف.
    1. أنشئ المصفوفة ذات الحجم $4 \times 16$ وهي $\mathbf{W}$ التي تحقق $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$، حيث $\mathbf{Y}$ هو الارتباط المتقاطع للمدخل $\mathbf{X}$ مع $\mathbf{K}$. كم عنصرًا غير صفري يحوي كل صف، وكم قيمة متمايزة تظهر في $\mathbf{W}$؟
    1. ابنِ $\mathbf{W}$ برمجيًا انطلاقًا من `K` وتأكد من أن الجداء يعيد إنتاج `corr2d(X, K)`.
    1. لمدخل بحجم $n \times n$ ونواة بحجم $k \times k$، احسب عدد عناصر $\mathbf{W}$ عند تخزينها كمصفوفة كثيفة، وعدد العناصر غير الصفرية في $\mathbf{W}$، وعدد عناصر مصفوفة الرقع. كيف ينمو كل عدد منها مع $n$؟
    1. احتفظ بنمط التناثر للمصفوفة $\mathbf{W}$ لكن امنح كل صف قيمه الخاصة البالغ عددها $k^2$. النتيجة هي طبقة متصلة محليًا: أي الموتر ذو الرتبة الرابعة $\mathsf{V}$ من :numref:`sec_why-conv` في المعادلة $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$، مقيَّدًا بنافذة بحجم $k \times k$ من الإزاحات $(a, b)$ قبل أن يربط تساوق الإزاحة الأوزان ببعضها. لمدخل بحجم $n \times n$، احسب عدد معاملات هذه الطبقة، ومعاملات طبقة كاملة الاتصال لها نفس شكلي المدخل والمخرج، ومعاملات الالتفاف.

    *مقتبس من مقرر CMU 11-785، [الواجب المنزلي 2 الجزء 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **مجال الاستقبال الفعّال.** ● يحدُّ مجال الاستقبال النظري $r$ في هذا القسم المدخلات التي *يمكن* أن تؤثر في المخرج؛ بينما يقيس مجال الاستقبال الفعّال مقدار تأثيرها الفعلي :cite:`Luo.Li.Urtasun.ea.2016`. ابنِ كدسة من $L$ من الطبقات الالتفافية أحادية القناة ذات الحجم $3 \times 3$ دون حشو، ودون انحياز، ودون لاخطية على مدخل بحجم $32 \times 32$، باستخدام الطبقة المدمجة في :numref:`subsec_learning_kernel`. اسحب كل عنصر من عناصر النوى بصورة مستقلة من التوزيع الغاوسي المعياري (التوزيع الطبيعي المعياري).
    1. لقيمة $L = 3$ وسحبة واحدة للنوى، انشر عبر الانتشار العكسي تدرجًا واحديًا من وحدة مخرج قريبة من مركز الطبقة الأخيرة إلى المدخل وارسم مقدار تدرج المدخل. قارن حامله بمجال الاستقبال النظري ذي الحجم $7 \times 7$، واذكر أين يكون المقدار أكبر ما يمكن.
    1. احسب متوسط مربع تدرج المدخل على مدى 500 سحبة للنوى، لكلٍّ من $L = 3$ و $L = 10$. اجمع كل خريطة متوسطة عبر صفوفها وطبِّع المقطع العمودي الناتج بحيث يصبح مجموعه مساويًا للواحد. كيف تتغير نسبة الانحراف المعياري للمقطع إلى طول الضلع النظري $2L + 1$ مع العمق؟
    1. يحصي :numref:`fig_conv_rf_paths` المسارات لحالة $L = 3$ في بُعد واحد. أثبت أنه لقيمة عامة $L$ تتناسب القيمة المتوقعة لمربع تدرج المدخل، مجموعًا عبر الصفوف، مع التفاف $L$ من النسخ من المتجه $(1, 1, 1)$. اشتق كيف ينمو الانحراف المعياري للمقطع المطبَّع مع $L$.

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
4. [code] **التمرير الخلفي للالتفاف.** ● ليكن $\mathbf{Y}$ هو الارتباط المتقاطع أحادي القناة لمدخل $\mathbf{X}$ مع نواة $\mathbf{K}$، كما تحسبه الدالة `corr2d`، ولتكن $\ell$ خسارة من عدد قياسي لا تعتمد على $\mathbf{X}$ و $\mathbf{K}$ إلا من خلال $\mathbf{Y}$.
    1. اشتق $\partial \ell/\partial \mathbf{K}$ و $\partial \ell/\partial \mathbf{X}$ بدلالة $\partial \ell/\partial \mathbf{Y}$، على غرار معادلات الانتشار العكسي في :numref:`sec_backprop`.
    1. نفِّذ صيغتك للمقدار $\partial \ell/\partial \mathbf{K}$ دون استخدام الاشتقاق التلقائي للخسارة $\ell = \sum_{i,j} G_{ij} Y_{ij}$، حيث $\mathbf{G}$ مصفوفة عشوائية ثابتة لها نفس شكل $\mathbf{Y}$. قارنها بتدرج الفروق المحدودة المركزية بحجم خطوة $10^{-2}$ على مدخل عشوائي صغير `X` ونواة `K`، وأورِد أكبر خطأ نسبي.
    1. تحقَّق من صيغتك للمقدار $\partial \ell/\partial \mathbf{K}$ بمقارنتها بتدرج النواة الذي يحسبه الاشتقاق التلقائي في تجربة :numref:`subsec_learning_kernel`.

    *مقتبس من كتاب Michael Nielsen، [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)، الفصل 6، ومقرر Stanford CS231n، [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/)، دفتر ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[النقاشات](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[النقاشات](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[النقاشات](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[النقاشات](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Style (register): Modern Standard Arabic technical prose addressing the reader in the collegial first-person plural ("ننفّذ"، "نختبر"، "نلاحظ"), with second-person singular masculine imperative in exercises and concept checks.
- Style (punctuation & numerals): Arabic punctuation (`،`، `؛`، `؟`) with Western Arabic numerals (`0`–`9`) matching math and code; zero invisible Unicode bidi marks.
- Style (acronyms): Retain standard English acronyms (`MLP`, `CNN`, `SGD`, `GPU`, `GEMM`, `2D`, `4D`, `im2col`) alongside Arabic terms where helpful on first mention.
- "locality" → "المحلية"
- "edge detection" / "edge detector" / "edge map" → "كشف الحواف" / "كاشف الحواف" / "خريطة الحواف"
- "sliding window" → "نافذة منزلقة"
- "patch" / "patch matrix" → "رقعة" (plural: "رقع") / "مصفوفة الرقع"
- "dense layer" → "الطبقة الكثيفة"
- "filter bank" → "مجموعة مرشحات" (plural: "مجموعات المرشحات")
- "flatten" / "flattened" → "تسطيح" / "مسطَّح"
- "unroll" / "unfold" → "بَسْط"
- "banded matrix" → "مصفوفة شريطية"
- "theoretical receptive field" / "effective receptive field" → "مجال الاستقبال النظري" / "مجال الاستقبال الفعّال"
- "stacked layers" → "الطبقات المكدَّسة"
- "Exercises" (section heading) → "التمارين"
- "Discussions" (tab link) → "النقاشات"
- "blur kernel" / "sharpening kernel" → "نواة تمويه" / "نواة زيادة الحدة"
- "downsample" → "خفض الدقة"
- "locally connected layer" → "طبقة متصلة محليًا"
- "support" (of a function/gradient) → "حامل"
