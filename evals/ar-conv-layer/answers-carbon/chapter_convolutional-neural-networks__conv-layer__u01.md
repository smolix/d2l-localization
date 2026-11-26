<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# الالتفافات للصور

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
في :numref:`sec_why-conv`، اشتققنا الطبقة الالتفافية على الورق من مبدأين هندسيين: تساوق الإزاحة والمحلية. والآن نرى ما تحسبه تلك الصيغة فعليًا على الصورة، وما إذا كان بإمكان النزول التدرجي اكتشاف نوى الالتفاف المفيدة تلقائيًا من البيانات.

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->
لنفكر في تحديد مواقع الحواف الرأسية حيث تلتقي منطقة ساطعة بأخرى معتمة. إن طرح قيمة الجار الأيمن لكل بكسل ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) يعطي صفرًا داخل المناطق المتجانسة، ويعطي $+1$ حيث ينتقل الأبيض إلى الأسود، ويعطي $-1$ حيث يعود الأسود إلى الأبيض. يؤدي مسح الكاشف ذي الوزنين $(1, -1)$ عبر كل موضع صالح في الصورة إلى إنتاج *خريطة حواف*. في هذا القسم، ننفّذ الارتباط المتقاطع ثنائي الأبعاد، ونختبر كاشف الحواف المصمَّم يدويًا، وندع النزول التدرجي *يتعلم* $(1, -1)$ مباشرةً من البيانات، ونبيّن كيف يتحول الالتفاف إلى عملية ضرب مصفوفات واحدة (*im2col*)، ونشتق كيف يؤدي تكديس الطبقات إلى توسيع مجال استقبال الوحدة.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## الارتباط المتقاطع

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### العملية

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
بترك القنوات جانبًا في الوقت الحالي، ليكن المدخل $\mathbf{X}$ مصفوفة بحجم $n_\textrm{h} \times n_\textrm{w}$ ولتكن نواة الالتفاف $\mathbf{K}$ مصفوفة بحجم $k_\textrm{h} \times k_\textrm{w}$. تحدد النواة *نافذة* منزلقة شكلها $k_\textrm{h} \times k_\textrm{w}$. بوضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$، نضرب رقعة المدخل المغطاة عنصرًا بعنصر في $\mathbf{K}$ ونجمع حواصل الضرب لإنتاج عنصر المخرج عند $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
هذا هو *الارتباط المتقاطع ثنائي الأبعاد* للمصفوفة $\mathbf{X}$ مع $\mathbf{K}$، ويُكتب $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ بفهارس تبدأ من الصفر مطابقةً للبرمجة. وبينما قام :numref:`sec_why-conv` بمَرْكَزة الإزاحات تناظريًا من $-\Delta$ إلى $\Delta$، فإن الإرساء عند الزاوية العلوية اليسرى لا يختلف إلا بإزاحة ثابتة مقدارها $\Delta$ مع التعامل بصورة طبيعية مع نوى الالتفاف زوجية الأبعاد (مثل نواة بحجم $2 \times 2$ أو كاشف الحواف لدينا ذي الحجم $1 \times 2$) التي لا تملك بكسلًا مركزيًا.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![الارتباط المتقاطع ثنائي الأبعاد. الأجزاء المظللة هي أول عنصر في المخرج وعناصر المدخل ونواة الالتفاف المستخدمة لحسابه: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
في :numref:`fig_correlation`، نُزلِق نواة التفاف بحجم $2 \times 2$ عبر مدخل بحجم $3 \times 3$ بمقدار بكسل واحد في كل مرة، من اليسار إلى اليمين ومن الأعلى إلى الأسفل. تستقر النافذة تمامًا في أربعة مواضع، منتجةً المخرج ذا الحجم $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
لاحظ أن المخرج يتقلص قليلًا على طول كل محور لأن نواة الالتفاف يجب أن تبقى بالكامل داخل المدخل. إن احتواء نافذة عرضها $k$ داخل طول قدره $n$ يترك $n - k + 1$ من المواضع الصالحة، ولذلك فإن شكل المخرج هو

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
وبذلك فإن مدخلًا بحجم $3 \times 3$ مع نواة التفاف بحجم $2 \times 2$ ينتج $2 \times 2$، وصورة بحجم $6 \times 8$ مع نواة بحجم $1 \times 2$ تنتج $6 \times 7$، وصورة Fashion-MNIST بحجم $28 \times 28$ مع نواة بحجم $5 \times 5$ تنتج $24 \times 24$. وكلما أردنا الحفاظ على الأبعاد المكانية، نحيط المدخل بإطار من الأصفار (*الحشو*)، وهو ما سندرسه إلى جانب خطوات الانزلاق والتوسيع في :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### التنفيذ

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
ننفّذ :eqref:`eq_corr2d` مباشرةً في الدالة `corr2d` باستخدام زوج من الحلقات التكرارية على كل إحداثي في المخرج.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
للتحقق من `corr2d` مقارنةً بحسابنا اليدوي في :numref:`fig_correlation`، ننشئ موتر المدخل ذا الحجم $3 \times 3$ المسمى `X` ونواة الالتفاف ذات الحجم $2 \times 2$ المسماة `K` ثم نحسب `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
يعيد المخرج إنتاج :numref:`fig_correlation` عنصرًا بعنصر (وكما هو مبيّن في :numref:`sec_why-conv`، فإن الالتفاف الرياضي الصارم بنواة التفاف مقلوبة $\mathbf{K}'$ يتعلم نفس عائلة الطبقات تمامًا، لذا نتبع مصطلحات التعلم العميق القياسية ونطلق على الارتباط المتقاطع اسم "الالتفاف").

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## الطبقات الالتفافية

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### الطبقة

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
تُجري الطبقة الالتفافية ثنائية الأبعاد ارتباطًا متقاطعًا لمدخلها مع نواة التفاف متعلَّمة وتضيف انحيازًا من عدد قياسي. تمامًا مثل أوزان البيرسبترون متعدد الطبقات (MLP) في :numref:`chap_perceptrons`، نهيّئ النواة من التوزيع $\mathcal{N}(0, 0.01^2)$ والانحياز بالقيمة صفر. أدناه، يسجّل الصنف `Conv2D` كلا المعاملين ويستدعي `corr2d` في التمرير الأمامي؛ وتُسمى الطبقة ذات النواة التي ارتفاعها $h$ وعرضها $w$ طبقة التفافية بحجم $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
على عكس الطبقة الكثيفة، لا تتقيد الطبقة الالتفافية بدقة مدخل ثابتة لأن معاملاتها تعتمد فقط على حجم النواة. فنفس الطبقة ذات الحجم $2 \times 2$ تمامًا (5 معاملات إجمالًا) تعالج كلًا من الموتر ذي الحجم $3 \times 3$ المسمى `X` وصورة بحجم $6 \times 8$، مُرجِعةً مخرجَين شكلاهما $2 \times 2$ و $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### كشف الحواف

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
نختبر الآن كاشف الحواف الرأسية المذكور في المقدمة على صورة اصطناعية بحجم $6 \times 8$ أعمدتها الأربعة الوسطى سوداء (`0`) وأعمدتها الخارجية بيضاء (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
نعرّف نواة الالتفاف ذات الحجم $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
عند كل موضع $(i, j)$، يحسب هذا *الفرق المحدود* الأفقي $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، وهو تقريب متقطع (حتى الإشارة) للمشتق الأفقي للسطوع، يتلاشى عبر المناطق المسطحة ويقفز بحدّة أينما قفزت الشدة الضوئية.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
إن إجراء الارتباط المتقاطع لصورة الشريط ذات الحجم $6 \times 8$ المسماة `X` مع نواة الالتفاف للفروق الأفقية ذات الحجم $1 \times 2$ المسماة `K` عبر `corr2d(X, K)` ينتج خريطة الحواف ذات الحجم $6 \times 7$ المسماة `Y`:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
وبالفعل، تُطلق `Y` القيمة `+1` على طول الحد الفاصل من الأبيض إلى الأسود، والقيمة `-1` على طول الحد الفاصل من الأسود إلى الأبيض، والقيمة `0` في كل موضع آخر. أما تطبيق نفس نواة الالتفاف تمامًا على الصورة المنقولة (مما يحوّل الشريط الرأسي إلى أفقي) فيعطي أصفارًا بالكامل:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
من الواضح أن `[1, -1]` تستجيب حصرًا للحواف الرأسية. ويتطلب كشف الحواف الأفقية أو الأقطار أو الزوايا أو الأنسجة نوى التفاف مختلفة تمامًا.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### تعلّم نواة الالتفاف

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->
إن التصميم اليدوي للنواة `[1, -1]` سهل عندما نعرف الميزة المطلوبة بالضبط، غير أنه لا يستطيع أي مهندس تحديد آلاف المرشحات الوسيطة عبر شبكة عميقة يدويًا. فهل يستطيع النزول التدرجي استعادة نواة الالتفاف `[1, -1]` من زوج المدخل والمخرج `(X, Y)` وحده؟ نختبر ذلك باستخدام الطبقة الالتفافية ثنائية الأبعاد (2D) المدمجة في كل إطار عمل (مع حذف الانحياز وتهيئة الأوزان من $\mathcal{N}(0, 0.01^2)$)، بتصغير الخطأ التربيعي المجموع عبر جميع عناصر المخرج الـ 42 على مدى 10 خطوات من التدرج عند معدّل التعلّم $0.03$. ولأن الطبقات المدمجة تتوقع موترات رباعية الأبعاد (4D) تشمل محورَي الدفعة والقناة (:numref:`sec_channels`)، نعيد تشكيل `X` و `Y` تبعًا لذلك.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
في البداية تستقر الخسارة بالقرب من 12 (إذ تخطئ نواة الالتفاف العشوائية القريبة من الصفر جميع بكسلات الحواف الـ 12 ذات القيم $\pm 1$ بمقدار يقارب 1)، وفي غضون تسع خطوات فقط تهبط إلى ما دون 0.01. ويؤكد فحص نواة الالتفاف المتعلَّمة ما توصلت إليه:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
يستقر كلا العنصرين ضمن نحو 0.02 من `[1, -1]`. لم يكن في هذه الحلقة ما يعرف شيئًا عن الحواف: فبمجرد توفر إشارة خسارة، اكتشف النزول التدرجي المرشح من تلقاء نفسه. هنا أشرفنا على الطبقة المفردة مباشرةً باستخدام `Y`؛ أما في الشبكة العصبية الالتفافية (CNN) العميقة، فيتعلم الانتشار العكسي نوى الالتفاف لكل طبقة من البداية إلى النهاية انطلاقًا من خسارة المهمة النهائية وحدها.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
الالتفافات للصور<br>**الارتباط المتقاطع ثنائي الأبعاد · كشف الحواف · النوى المتعلَّمة · im2col · مجالات الاستقبال**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="كشف الحواف المحلية عبر الصورة"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[الدافع والنموذج الذهني الأساسي]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **صورة اختبار اصطناعية بحجم $6 \times 8$:** أعمدة خارجية بيضاء (`1`) تحيط بشريط رأسي أسود من 4 أعمدة (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **الفرق المحدود الأفقي:** مسح $\mathbf{K} = [1, -1]$ يحسب $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، مُطلِقًا `+1` عند الحواف من الأبيض إلى الأسود و `-1` عند الحواف من الأسود إلى الأبيض.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **التعلّم مقابل التصميم اليدوي:** تكتشف الشبكات العصبية الالتفافية (CNN) العميقة آلافًا من هذه النوى تلقائيًا عبر النزول التدرجي.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="عملية الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[الجداءات الداخلية بالنافذة المنزلقة]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
بإعطاء مصفوفة مدخل بحجم $n_\textrm{h} \times n_\textrm{w}$ هي $\mathbf{X}$ ونواة التفاف بحجم $k_\textrm{h} \times k_\textrm{w}$ هي $\mathbf{K}$، يضع الارتباط المتقاطع ثنائي الأبعاد $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ الزاوية العلوية اليسرى للنافذة عند $(i, j)$ ويجمع حواصل الضرب عنصرًا بعنصر:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **الإرساء العلوي الأيسر:** الفهارس الصفرية $a, b \ge 0$ تطابق فهرسة المصفوفات برمجيًا وتتعامل مع النوى زوجية الأبعاد (مثل $2 \times 2$ أو $1 \times 2$) بسلاسة.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **المخرج العلوي الأيسر المظلل $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="مواضع النافذة الصالحة والشكل المكاني للمخرج"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[لماذا تقلّص الالتفافات غير المحشوة الشبكة]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **مواضع النافذة الصالحة:** تتسع نافذة بحجم $k_\textrm{h} \times k_\textrm{w}$ داخل شبكة بحجم $n_\textrm{h} \times n_\textrm{w}$ في $n - k + 1$ من المواضع لكل محور.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **أمثلة على الشكل:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)؛ و $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="تنفيذ الارتباط المتقاطع ثنائي الأبعاد برمجيًا"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[تنفيذ مباشر بحلقات متداخلة]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
تخصص الدالة `corr2d` موتر المخرج ذا الشكل $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ وتحسب كل عنصر بوصفه مجموع `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
يقتطع كل تكرار شريحة نافذة فرعية بحجم $k_\textrm{h} \times k_\textrm{w}$ من `X`، ويضربها عنصرًا بعنصر في `K`، ثم يختزلها عبر `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="التحقق من corr2d وتكافؤ النواة المقلوبة"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[الارتباط المتقاطع مقابل الالتفاف الرياضي]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **نواة مقلوبة بزاوية 180° هي $\mathbf{K}'$:** الارتباط المتقاطع مع $\mathbf{K}$ يساوي الالتفاف مع $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **نفس الفئة المتعلَّمة:** تحسب أطر العمل الارتباط المتقاطع وتسميه التفافًا.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="بناء وحدة الطبقة الالتفافية ثنائية الأبعاد"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[أوزان نواة قابلة للتعلّم مع انحياز من عدد قياسي]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
تغلّف الطبقة الالتفافية الدالة `corr2d` بمعامل وزن قابل للتعلّم بحجم $k_\textrm{h} \times k_\textrm{w}$ (مهيّأ من $\mathcal{N}(0, 0.01^2)$) وانحياز من عدد قياسي (مهيّأ بالقيمة $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="استقلالية الطبقات الالتفافية عن دقة الصورة"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[معاملات ثابتة عبر أحجام مدخلات كيفية]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **مستقلة عن $H \times W$:** نفس الـ 5 معاملات تمامًا (نواة التفاف $2 \times 2$ + انحياز قدره 1) تعالج مدخلات بحجم $3 \times 3$ أو $6 \times 8$ أو بالميغابكسل دون إعادة تشكيل الأوزان.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="إنشاء صورة اصطناعية ذات حواف رأسية"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[نمط اختبار ثنائي بحجم 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
ننشئ صورة الاختبار ذات الحجم $6 \times 8$ المسماة `X` بقيم واحد على الحافتين اليسرى واليمنى وأصفار عبر الأعمدة الأربعة الوسطى (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **أعمدة ثابتة مقطعيًا:** الشدة ثابتة (`1` أو `0`) داخل الأعمدة $0\text{--}1$ و $2\text{--}5$ و $6\text{--}7$، ولا تقفز إلا بين العمودين $1 \to 2$ والعمودين $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="تطبيق نواة الالتفاف للفروق المحدودة الأفقية"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[كشف الحواف الرأسية باستخدام K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
إن إجراء الارتباط المتقاطع للمدخل `X` مع نواة الالتفاف ذات الحجم $1 \times 2$ وهي $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ يحسب $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ عند كل زوج أفقي:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **خريطة حواف بإشارة:** العمود 1 يُطلق `+1` (انتقال من الأبيض إلى الأسود)، والعمود 5 يُطلق `-1` (انتقال من الأسود إلى الأبيض)، وجميع الأعمدة الداخلية الثابتة تساوي `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="انتقائية الاتجاه لنوى الحواف"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[لماذا لا تستطيع نواة التفاف واحدة كشف جميع الاتجاهات]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **عمياء عن الحواف الأفقية:** أخذ منقول `X` (`X.T`) يجعل الجيران الأفقيين متطابقين، لذا تُخرج $[1, -1]$ أصفارًا بالكامل.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **مجموعات المرشحات (§{sec_channels}):** تتعلم طبقات الرؤية قنوات متوازية للاتجاهات الأفقية والرأسية والقطرية.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="تعلّم كاشف حواف عبر النزول التدرجي"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[الإشراف على الطبقة الالتفافية المدمجة باستخدام (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **موترات رباعية الأبعاد (4D) في أطر العمل:** نعيد تشكيل `X` و `Y` إلى موترات رباعية الأبعاد (مثال واحد، قناة واحدة) ونصغّر $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ عند $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="فحص أوزان النواة المتعلَّمة"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[لماذا تبدأ الخسارة عند 12 وتتقارب إلى [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **لماذا تبدأ الخسارة عند $12.0$:** الأوزان القريبة من الصفر تتنبأ بـ $\hat{Y} \approx 0$، فتخطئ بكسلات الحواف الاثني عشر ذات القيم $\pm 1$ في `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **التقارب في 9 خطوات:** يستعيد النزول التدرجي العشوائي (SGD) القيم $[0.99, -0.99]$ تلقائيًا من البيانات.

=== DECISIONS ===
- Style (register): Modern Standard Arabic technical prose addressing the reader in the collegial first-person plural ("ننفّذ"، "نختبر"، "نلاحظ"), with second-person singular masculine imperative in exercises and concept checks.
- Style (punctuation & numerals): Arabic punctuation (`،`، `؛`، `؟`) with Western Arabic numerals (`0`–`9`) matching math and code; zero invisible Unicode bidi marks.
- Style (acronyms): Retain standard English acronyms (`MLP`, `CNN`, `SGD`, `GPU`, `GEMM`, `2D`, `4D`, `im2col`) alongside Arabic terms where helpful on first mention.
- "locality" → "المحلية"
- "edge detection" / "edge detector" / "edge map" → "كشف الحواف" / "كاشف الحواف" / "خريطة الحواف"
- "sliding window" → "نافذة منزلقة"
- "patch" → "رقعة" (plural: "رقع")
- "dense layer" → "الطبقة الكثيفة"
- "filter bank" → "مجموعة مرشحات" (plural: "مجموعات المرشحات")

=== NOTES ===
- 55f0a41b: English "returns" ("where black returns to white") is an ordinary verb ("يعود"), not the RL glossary term *return* ("عائد").
- 29cd9061: English "returning" ("returning outputs") is an ordinary participle ("مُرجِعةً"), not the RL glossary term *return* ("عائد").
