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
1. [code] **نوى مصمَّمة يدويًا.** صمّم بعض النوى يدويًا.
    1. بالنظر إلى متجه اتجاه $\mathbf{v} = (v_1, v_2)$، اشتق نواة لكشف الحواف تكشف الحواف العمودية على $\mathbf{v}$،
       أي الحواف في الاتجاه $(v_2, -v_1)$.
    1. اشتق مؤثر فرق محدود للمشتق الثاني.
       أي البنى في الصور تستجيب له بأشد قوة؟
    1. ما أصغر حجم لنواة يتيح الحصول على مشتق
       من الرتبة $d$؟
    1. كيف تصمّم نواة تمويه؟ ولماذا قد ترغب في استخدام نواة كهذه؟
    1. حمّل صورة فوتوغرافية، وحوّلها إلى قناة واحدة، وقلّل دقتها إلى
       نحو $100 \times 100$ بكسل (حلقات `corr2d` في Python). طبّق
       كاشف الحواف ونواة التمويه ونواة التحديد "ضعف الوحدة ناقص
       التمويه"، واعرض كل نتيجة بجانب الصورة
       الأصلية. ما الذي تُظهره خريطة حواف الصورة الفوتوغرافية ولم تستطع خريطة حواف
       الشريط الأسود إظهاره؟

    *مقتبس من Stanford CS231n،
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
2. [code] **الالتفاف بوصفه مصفوفة شريطية.** مصفوفة الرقع في
   :numref:`subsec_conv_matmul` تفرد المدخل. أما البناء المعاكس
   فيفكّ النواة. ليكن $\mathbf{X}$ مدخلًا $4 \times 4$،
   و$\mathbf{K}$ نواة $3 \times 3$، و$\textrm{vec}(\cdot)$
   التسطيح صفًا بصف.
    1. ابنِ المصفوفة $\mathbf{W}$ ذات الحجم $4 \times 16$ التي تحقق
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$،
       حيث $\mathbf{Y}$ هو الارتباط المتقاطع لـ $\mathbf{X}$ مع
       $\mathbf{K}$. كم عنصرًا غير صفري في كل صف، وكم قيمة
       متمايزة تظهر في $\mathbf{W}$؟
    1. ابنِ $\mathbf{W}$ في الشيفرة انطلاقًا من `K` وتأكد من أن الجداء
       يعيد إنتاج `corr2d(X, K)`.
    1. لمدخل $n \times n$ ونواة $k \times k$، عُدّ
       عناصر $\mathbf{W}$ المخزّنة بوصفها مصفوفة كثيفة، والعناصر غير الصفرية
       في $\mathbf{W}$، وعناصر مصفوفة الرقع. كيف
       ينمو كل عدد مع $n$؟
    1. احتفظ بنمط التناثر في $\mathbf{W}$ لكن أعطِ كل صف
       $k^2$ قيمة خاصة به. والنتيجة طبقة متصلة محليًا:
       الموتر الرباعي الرتبة $\mathsf{V}$ في :numref:`sec_why-conv` ضمن
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$،
       مقيّدًا بنافذة $k \times k$ من الإزاحات $(a, b)$ قبل أن
       يربط تساوق الإزاحة الأوزان. لمدخل $n \times n$،
       عُدّ معاملات هذه الطبقة، ومعاملات طبقة متصلة اتصالًا كاملًا
       بشكلي المدخل والخرج نفسيهما، ومعاملات
       الالتفاف.

    *مقتبس من CMU 11-785،
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
3. [code] **مجال الاستقبال الفعّال.** ● إن مجال الاستقبال
   النظري $r$ في هذا القسم يحدّ المدخلات التي *يمكنها* التأثير في
   خرج ما؛ أما مجال الاستقبال الفعّال فيقيس مقدار تأثيرها
   الفعلي :cite:`Luo.Li.Urtasun.ea.2016`. ابنِ مكدّسًا من $L$ طبقة
   التفافية $3 \times 3$ أحادية القناة بلا
   حشو ولا انحياز ولا دالة لاخطية على مدخل $32 \times 32$، مستخدمًا
   الطبقة المدمجة في :numref:`subsec_learning_kernel`. اسحب كل
   عنصر من عناصر النوى بشكل مستقل من توزيع طبيعي معياري.
    1. لـ $L = 3$ وسحبة واحدة للنوى، مرّر تدرجًا وحدويًا بالانتشار العكسي من وحدة خرج قرب مركز
       الطبقة الأخيرة إلى المدخل وارسم مقدار تدرج المدخل. قارن سنده
       بمجال الاستقبال النظري $7 \times 7$، وبيّن
       أين يبلغ المقدار أكبر قيمة.
    1. احسب متوسط مربع تدرج المدخل على 500 سحبة للنوى،
       لـ $L = 3$ ولـ $L = 10$. اجمع كل خريطة متوسطة على صفوفها
       وطبّع منحنى الأعمدة الناتج ليكون مجموعه واحدًا. كيف
       تتغير نسبة الانحراف المعياري للمنحنى إلى
       طول الضلع النظري $2L + 1$ مع العمق؟
    1. يعدّ :numref:`fig_conv_rf_paths` المسارات لـ $L = 3$ في بعد
       واحد. بيّن أنه لـ $L$ عامة يتناسب المتوسط المتوقع لمربع تدرج المدخل
       مجموعًا على الصفوف مع التفاف
       $L$ نسخة من المتجه $(1, 1, 1)$. اشتق كيف ينمو الانحراف
       المعياري للمنحنى المطبّع مع $L$.

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
4. [code] **التمرير الخلفي للالتفاف.** ● ليكن $\mathbf{Y}$ الارتباط المتقاطع
   أحادي القناة لمدخل $\mathbf{X}$ مع نواة
   $\mathbf{K}$، كما تحسبه `corr2d`، وليكن $\ell$ خسارة عددية
   تعتمد على $\mathbf{X}$ و$\mathbf{K}$ عبر $\mathbf{Y}$ فقط.
    1. اشتق $\partial \ell/\partial \mathbf{K}$ و
       $\partial \ell/\partial \mathbf{X}$ بدلالة
       $\partial \ell/\partial \mathbf{Y}$، على نمط معادلات
       الانتشار العكسي في :numref:`sec_backprop`.
    1. نفّذ تعبيرك لـ $\partial \ell/\partial \mathbf{K}$
       دون اشتقاق تلقائي للخسارة
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$، حيث $\mathbf{G}$ مصفوفة
       عشوائية ثابتة بشكل $\mathbf{Y}$. قارنه بتدرج فرق محدود مركزي
       بحجم خطوة $10^{-2}$ على `X` و`K` عشوائيين صغيرين، وأبلغ عن أكبر خطأ نسبي.
    1. تحقق من تعبيرك لـ $\partial \ell/\partial \mathbf{K}$
       مقابل تدرج النواة الذي يحسبه الاشتقاق التلقائي
       في تجربة :numref:`subsec_learning_kernel`.

    *مقتبس من Michael Nielsen،
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)،
    الفصل 6، ومن Stanford CS231n،
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)،
    دفتر ConvolutionalNetworks.*

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
- Exercise blocks: imperative singular masculine ("اشتق", "ابنِ", "نفّذ"); the tags `[code]` and `●` stay unchanged.
- "locally connected layer" → طبقة متصلة محليًا; "fully connected layer" → طبقة متصلة اتصالًا كاملًا; "dense array" → مصفوفة كثيفة; "nonzero" → غير صفري; "unit gradient" → تدرج وحدوي.
- "finite-difference gradient" → تدرج فرق محدود; "relative error" → خطأ نسبي; "automatic differentiation" → الاشتقاق التلقائي; "blur/sharpening kernel" → نواة التمويه / التحديد; "downsample" → تقليل الدقة.
- Tensor of order four ($\mathsf{V}$) → موتر رباعي الرتبة.
- "Discussions" link text → النقاشات.

=== NOTES ===
- 54bcee26: the English sentence "the expected squared input gradient, summed over rows, is proportional to the convolution ..." was kept with the same meaning; word order in Arabic differs slightly.
- ec8621f8, 8d45aae8, 24a25ec4: "math tokens reordered" is Arabic word order (e.g. "$4 \times 4$ ... $\mathbf{X}$"); no math changed.
- "numbers not found" warnings (12, 4, 5, 1, 2, 6) are false matches: those numbers are written as Arabic words (اثنا عشر, أربعة, خمسة, واحد, بكسلين, ستة) or "واحد" in prose; the "glossary term not used" warnings are inflections or deliberate short forms (النواة for نواة الالتفاف, definite/plural forms of طبقة, الوزن, التدرج).
