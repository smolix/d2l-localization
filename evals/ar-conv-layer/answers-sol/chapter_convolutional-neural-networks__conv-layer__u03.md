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
1. [code] **تصميم النوى يدويًا.** صمّم بعض نوى الالتفاف يدويًا.
    1. إذا كان متجه الاتجاه هو $\mathbf{v} = (v_1, v_2)$، فاشتقّ نواةً تكشف الحواف المتعامدة مع $\mathbf{v}$، أي الحواف الممتدّة في الاتجاه $(v_2, -v_1)$.
    1. اشتقّ مؤثّرًا بالفروق المحدودة لحساب المشتق الثاني. ما البنى في الصور التي تستجيب له بأكبر شدّة؟
    1. ما أصغر حجم لنواة يمكن بها الحصول على مشتقّ من الرتبة $d$؟
    1. كيف تصمّم نواة تمويه؟ ولماذا قد ترغب في استخدام مثل هذه النواة؟
    1. حمّل صورة فوتوغرافية، وحوّلها إلى قناة واحدة، وقلّل عدد عيناتها إلى نحو $100 \times 100$ بكسل (تستخدم `corr2d` حلقات بلغة Python). طبّق كاشف الحواف ونواة التمويه ونواة زيادة الحِدّة «ضعف نواة الهوية ناقص نواة التمويه»، واعرض كل نتيجة إلى جانب الصورة الأصلية. ما الذي تُظهره خريطة حواف الصورة الفوتوغرافية وتعجز خريطة حواف الشريط الأسود عن إظهاره؟

    *بتصرّف عن دفتر ConvolutionalNetworks في [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/) لمقرر CS231n في Stanford.*

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
2. [code] **تمثيل الالتفاف بمصفوفة شريطية.** تفرد مصفوفة الرقع في :numref:`subsec_conv_matmul` المدخلة. وفي الصياغة المقابلة نفرد النواة. لتكن $\mathbf{X}$ مدخلةً أبعادها $4 \times 4$، و$\mathbf{K}$ نواةً أبعادها $3 \times 3$، و$\textrm{vec}(\cdot)$ عملية التسطيح صفًا تلو الآخر.
    1. أنشئ مصفوفة أبعادها $4 \times 16$ هي $\mathbf{W}$ بحيث يكون $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$، حيث $\mathbf{Y}$ هو الارتباط المتقاطع بين $\mathbf{X}$ و$\mathbf{K}$. كم عنصرًا غير صفري في كل صفّ؟ وكم قيمةً مختلفةً تظهر في $\mathbf{W}$؟
    1. أنشئ $\mathbf{W}$ برمجيًا انطلاقًا من `K`، وتحقّق من أن حاصل الضرب يعطي نتيجة `corr2d(X, K)` نفسها.
    1. لمدخلة أبعادها $n \times n$ ونواة أبعادها $k \times k$، احسب عدد عناصر $\mathbf{W}$ عند تخزينها في مصفوفة كثيفة، وعدد عناصر $\mathbf{W}$ غير الصفرية، وعدد عناصر مصفوفة الرقع. كيف يزداد كلٌّ من هذه الأعداد مع ازدياد $n$؟
    1. أبقِ على نمط التناثر في $\mathbf{W}$، لكن خصّص لكل صفّ $k^2$ قيمةً تخصّه. والنتيجة طبقة متّصلة محليًا، يمثّلها الموتر من الرتبة الرابعة $\mathsf{V}$ الوارد في :numref:`sec_why-conv` في المعادلة $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$، بعد حصر الإزاحات في نافذة أبعادها $k \times k$ من الإزاحات $(a, b)$ وقبل تطبيق تساوق الإزاحة الذي يُلزم الصفوف بتشارك الأوزان. ولمدخلة أبعادها $n \times n$، احسب عدد معلمات هذه الطبقة، وعدد معلمات طبقة كاملة الاتصال لها شكل المدخلة وشكل المخرج نفسهما، وعدد معلمات الالتفاف.

    *بتصرّف عن [الجزء 1 من الواجب 2](https://deeplearning.cs.cmu.edu/S21/index.html) لمقرر CMU 11-785.*

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
3. [code] **مجال الاستقبال الفعلي.** ● يحدّد مجال الاستقبال النظري $r$ في هذا القسم المدخلات التي *يمكنها* التأثير في أحد عناصر المخرج. ويقيس مجال الاستقبال الفعلي مقدار تأثيرها بالفعل :cite:`Luo.Li.Urtasun.ea.2016`. كوّن سلسلة من $L$ طبقةً التفافية أحادية القناة بأبعاد $3 \times 3$، بلا حشو ولا انحياز ولا تحويل لاخطي، على مدخلة أبعادها $32 \times 32$، مستخدمًا الطبقة المدمجة في :numref:`subsec_learning_kernel`. اسحب قيمة كل عنصر من عناصر النوى بصورة مستقلّة من توزيع طبيعي معياري.
    1. عند $L = 3$ ولعيّنة واحدة من النوى، مرّر تدرجًا قيمته واحد عكسيًا، بدءًا من وحدة مخرج قرب مركز الطبقة الأخيرة وصولًا إلى المدخلة، وارسم مقدار التدرج بالنسبة إلى المدخلة. قارن نطاق قيمه غير الصفرية بمجال الاستقبال النظري ذي الأبعاد $7 \times 7$، واذكر أين يبلغ المقدار أقصاه.
    1. احسب متوسط مربع التدرج بالنسبة إلى المدخلة على مدى 500 سحب للنوى، لكلٍّ من $L = 3$ و$L = 10$. اجمع كل خريطة متوسطة على امتداد الصفوف، وطبّع التوزيع الناتج على الأعمدة بحيث يساوي مجموع قيمه واحدًا. كيف تتغيّر نسبة الانحراف المعياري لهذا التوزيع إلى طول الضلع النظري $2L + 1$ مع ازدياد العمق؟
    1. يحصي :numref:`fig_conv_rf_paths` المسارات عندما يكون $L = 3$ في بُعدٍ واحد. بيّن أنه لأيّ $L$، يتناسب مجموع القيم المتوقّعة لمربّع التدرج بالنسبة إلى المدخلة، على امتداد الصفوف، مع التفاف $L$ نسخةً من المتجه $(1, 1, 1)$. واشتقّ كيف يزداد الانحراف المعياري للتوزيع المطبّع مع ازدياد $L$.

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
4. [code] **التمرير الخلفي لعملية الالتفاف.** ● لتكن $\mathbf{Y}$ ناتج الارتباط المتقاطع أحادي القناة بين مدخلة $\mathbf{X}$ ونواة $\mathbf{K}$، كما تحسبه `corr2d`. ولتكن $\ell$ خسارة قيمتها عدد قياسي، لا تعتمد على $\mathbf{X}$ ولا على $\mathbf{K}$ إلا من خلال $\mathbf{Y}$.
    1. اشتقّ $\partial \ell/\partial \mathbf{K}$ و$\partial \ell/\partial \mathbf{X}$ بدلالة $\partial \ell/\partial \mathbf{Y}$، على غرار معادلات الانتشار العكسي في :numref:`sec_backprop`.
    1. نفّذ الصيغة التي اشتققتها لـ $\partial \ell/\partial \mathbf{K}$ من دون استعمال الاشتقاق التلقائي، وذلك للخسارة $\ell = \sum_{i,j} G_{ij} Y_{ij}$، حيث $\mathbf{G}$ مصفوفة عشوائية ثابتة لها شكل $\mathbf{Y}$ نفسه. قارن النتيجة بتدرج محسوب بالفروق المحدودة المركزية بحجم خطوة $10^{-2}$، باستخدام `X` و`K` صغيرين عشوائيين، وأبلغ عن أكبر خطأ نسبي.
    1. تحقّق من الصيغة التي اشتققتها لـ $\partial \ell/\partial \mathbf{K}$ بمقارنتها بتدرج النواة الذي يحسبه الاشتقاق التلقائي في تجربة :numref:`subsec_learning_kernel`.

    *بتصرّف عن الفصل 6 من [الشبكات العصبية والتعلّم العميق](http://neuralnetworksanddeeplearning.com/chap6.html) لـ Michael Nielsen، وعن دفتر ConvolutionalNetworks في [الواجب 2](https://cs231n.github.io/assignments2024/assignment2/) لمقرر CS231n في Stanford.*

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
- Blur kernel → «نواة تمويه»; sharpening kernel → «نواة زيادة الحِدّة»; locally connected layer → «طبقة متّصلة محليًا»; standard normal distribution → «توزيع طبيعي معياري».
- Use «نطاق القيم غير الصفرية» for support in the gradient exercise and «التوزيع على الأعمدة» for the column profile being normalized to sum to one.
