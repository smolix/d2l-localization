<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## تمارين

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
1. [code] **أنوية مصمَّمة يدويًا.** صمّم بعض أنوية الالتفاف يدويًا.
    1. إذا أُعطيت متجه اتجاه $\mathbf{v} = (v_1, v_2)$، فاشتق نواة لكشف الحواف
       تكشف الحواف المتعامدة مع $\mathbf{v}$،
       أي الحواف في الاتجاه $(v_2, -v_1)$.
    1. اشتق مؤثر فروق محدودة للمشتق الثاني.
       أي البنى في الصور تستجيب له بأقوى ما يكون؟
    1. ما أصغر حجم لنواة تعطي مشتقًا من
       الرتبة $d$؟
    1. كيف تصمّم نواة تمويه؟ ولماذا قد ترغب في استخدام
       نواة كهذه؟
    1. حمّل صورة فوتوغرافية، وحوّلها إلى قناة واحدة، واخفض دقتها إلى
       نحو $100 \times 100$ بكسل (فالدالة `corr2d` تدور حلقاتها في Python). طبّق
       كاشف الحواف ونواة التمويه ونواة الشحذ «ضعف النواة المحايدة
       ناقص نواة التمويه»، واعرض كل نتيجة بجانب الصورة
       الأصلية. ما الذي تظهره خريطة حواف صورة فوتوغرافية
       ولم تكن خريطة حواف الشريط الأسود لتظهره؟

    *مقتبس بتصرف من Stanford CS231n،
    [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/)،
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
   :numref:`subsec_conv_matmul` تبسط الدخل. والبناء المعاكس
   يفرد النواة. ليكن $\mathbf{X}$ دخلًا $4 \times 4$،
   و $\mathbf{K}$ نواة $3 \times 3$، و $\textrm{vec}(\cdot)$
   التسطيح صفًا بصف.
    1. أنشئ مصفوفة $4 \times 16$ هي $\mathbf{W}$ تحقق
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$،
       حيث $\mathbf{Y}$ هو الارتباط المتقاطع بين $\mathbf{X}$ و
       $\mathbf{K}$. كم عنصرًا غير صفري في كل صف، وكم
       قيمة مختلفة تظهر في $\mathbf{W}$؟
    1. ابنِ $\mathbf{W}$ في الشيفرة انطلاقًا من `K` وتأكد من أن الجداء
       يعيد إنتاج `corr2d(X, K)`.
    1. لدخل $n \times n$ ونواة $k \times k$، احسب عدد
       عناصر $\mathbf{W}$ إذا خُزّنت مصفوفة كثيفة، وعدد العناصر غير
       الصفرية في $\mathbf{W}$، وعدد عناصر مصفوفة الرقع. كيف
       ينمو كل عدد مع $n$؟
    1. أبقِ نمط التناثر في $\mathbf{W}$ لكن أعطِ كل صف قيمه
       الخاصة وعددها $k^2$. الناتج طبقة موصولة محليًا (locally connected layer): وهي
       الموتر من الرتبة الرابعة $\mathsf{V}$ الوارد في :numref:`sec_why-conv` في
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$،
       مقصورًا على نافذة $k \times k$ من الإزاحات $(a, b)$، قبل
       أن يربط تساوق الإزاحة الأوزان بعضها ببعض. لدخل $n \times n$،
       احسب عدد معاملات هذه الطبقة، وعدد معاملات طبقة كاملة الاتصال
       لها شكلا الدخل والخرج نفساهما، وعدد معاملات
       الالتفاف.

    *مقتبس بتصرف من CMU 11-785،
    [الواجب المنزلي 2، الجزء 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **مجال الاستقبال الفعّال.** ● مجال الاستقبال النظري
   $r$ في هذا القسم يحدّ المدخلات التي *يمكن* أن تؤثر في
   خرج ما؛ أما مجال الاستقبال الفعّال فيقيس مقدار
   تأثيرها الفعلي :cite:`Luo.Li.Urtasun.ea.2016`. ابنِ كدسة من $L$
   طبقة التفافية $3 \times 3$ أحادية القناة، بلا
   حشو ولا انحياز ولا لاخطية، على دخل $32 \times 32$، مستخدمًا
   الطبقة المدمجة من :numref:`subsec_learning_kernel`. اسحب كل
   عنصر من عناصر الأنوية بصورة مستقلة من التوزيع الغاوسي المعياري.
    1. عند $L = 3$ ولسحبة واحدة للأنوية، انشر عكسيًا تدرجًا
       مقداره واحد من وحدة خرج قريبة من مركز الطبقة الأخيرة إلى
       الدخل، وارسم مقدار تدرج الدخل. قارن
       حامله (support) بمجال الاستقبال النظري $7 \times 7$، وبيّن
       أين يبلغ المقدار أقصاه.
    1. احسب متوسط مربع تدرج الدخل على 500 سحبة للأنوية،
       عند $L = 3$ وعند $L = 10$. اجمع كل خريطة متوسطة على صفوفها
       وطبّع مقطع الأعمدة الناتج ليكون مجموعه واحدًا. كيف
       تتغير مع العمق نسبة الانحراف المعياري للمقطع إلى
       طول الضلع النظري $2L + 1$؟
    1. يعدّ :numref:`fig_conv_rf_paths` المسارات عند $L = 3$ في بعد
       واحد. بيّن أن القيمة المتوقعة لمربع تدرج الدخل،
       مجموعةً على الصفوف، تتناسب عند $L$ عام مع التفاف
       $L$ نسخة من المتجه $(1, 1, 1)$. واشتق كيف ينمو الانحراف
       المعياري للمقطع المطبَّع مع $L$.

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
4. [code] **التمرير الخلفي للالتفاف.** ● ليكن $\mathbf{Y}$
   الارتباط المتقاطع أحادي القناة بين دخل $\mathbf{X}$ ونواة
   $\mathbf{K}$، كما تحسبه `corr2d`، ولتكن $\ell$ خسارة قياسية
   لا تعتمد على $\mathbf{X}$ و $\mathbf{K}$ إلا عبر $\mathbf{Y}$.
    1. اشتق $\partial \ell/\partial \mathbf{K}$ و
       $\partial \ell/\partial \mathbf{X}$ بدلالة
       $\partial \ell/\partial \mathbf{Y}$، على نسق
       معادلات الانتشار العكسي في :numref:`sec_backprop`.
    1. نفّذ عبارتك للتدرج $\partial \ell/\partial \mathbf{K}$
       دون اشتقاق تلقائي للخسارة
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$، حيث $\mathbf{G}$ مصفوفة
       عشوائية ثابتة لها شكل $\mathbf{Y}$. قارنها
       بتدرج محسوب بالفروق المحدودة المركزية بحجم خطوة $10^{-2}$ على
       `X` و `K` عشوائيين صغيرين، واذكر أكبر خطأ نسبي.
    1. تحقق من عبارتك للتدرج $\partial \ell/\partial \mathbf{K}$
       مقابل تدرج النواة الذي يحسبه الاشتقاق التلقائي
       في تجربة :numref:`subsec_learning_kernel`.

    *مقتبس بتصرف من Michael Nielsen،
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)،
    الفصل 6، ومن Stanford CS231n،
    [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/)،
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
- "Exercises" (heading) → «تمارين»; "Discussions" (link text) → «النقاشات».
- The exercise tag `[code]` and the difficulty mark ● are kept as in the English.
- "Adapted from …" → «مقتبس بتصرف من …»; course names and book titles stay in English, "Assignment 2" → «الواجب 2», "Homework 2 Part 1" → «الواجب المنزلي 2، الجزء 1», "notebook" → «دفتر».
- "blur kernel" → «نواة التمويه»; "sharpening kernel" → «نواة الشحذ»; "identity (kernel)" → «النواة المحايدة».
- "downsample" → «يخفض الدقة».
- "finite difference operator" → «مؤثر فروق محدودة»; "centered finite difference" → «الفروق المحدودة المركزية».
- "derivative of order d" → «مشتق من الرتبة d»; "fourth-order tensor" → «موتر من الرتبة الرابعة».
- "nonzero" → «غير صفري».
- "sparsity pattern" → «نمط التناثر».
- "locally connected layer" → «طبقة موصولة محليًا».
- "draw" (a random sample) → «يسحب», noun «سحبة».
- "standard normal distribution" → «التوزيع الغاوسي المعياري» (after the glossary's «التوزيع الغاوسي»).
- "support" (of a function) → «حامل».
- "normalize" → «يطبّع»; "profile" → «مقطع»; "column profile" → «مقطع الأعمدة».
- "expected (value)" → «القيمة المتوقعة».
- "expression" (mathematical) → «عبارة».
- "relative error" → «خطأ نسبي».
- "backpropagate" (verb) → «ينشر عكسيًا».

=== NOTES ===
- Glossary warnings "term not used" for 'kernel' → «نواة الالتفاف» and 'weight' → «الوزن» are false matches throughout this unit: after the first mention the kernel is «النواة» (plural «أنوية»), and weights are the plural «أوزان» or indefinite «وزن» (see DECISIONS in u01).
- 6f4ddec6: "(`corr2d` loops in Python)" gives the reason for downsampling; translated as a clause, «فالدالة `corr2d` تدور حلقاتها في Python».
- 54bcee26: "a unit gradient" is a gradient of value one, «تدرجًا مقداره واحد»; glossary check may flag "value" or "unit" there.
- 6ab5b67c: "your expression for $\partial \ell/\partial \mathbf{K}$" is «عبارتك للتدرج …», naming the quantity so that no preposition is attached to the math.
