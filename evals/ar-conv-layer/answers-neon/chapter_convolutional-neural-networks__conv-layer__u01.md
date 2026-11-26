<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# الالتفاف للصور

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
في :numref:`sec_why-conv`، اشتققنا الطبقة الالتفافية نظريًا انطلاقًا من مبدأين هندسيين: تساوق الإزاحة والمحلية. والآن نرى ما تحسبه تلك الصيغة فعليًا على صورة، وما إذا كان النزول التدرجي قادرًا على اكتشاف نوى التفاف مفيدة تلقائيًا من البيانات.

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
لنفكر في تحديد الحواف الرأسية حيث تلتقي منطقة ساطعة بأخرى داكنة. إن طرح قيمة البكسل المجاور من جهة اليمين من كل بكسل ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) يعطي صفرًا داخل المناطق المتجانسة، و$+1$ حيث ينتقل الأبيض إلى الأسود، و$-1$ حيث يعود الأسود إلى الأبيض. ويؤدي تمرير الكاشف ذي الوزنين $(1, -1)$ عبر كل موضع صالح في الصورة إلى إنتاج *خريطة حواف* (edge map). في هذا القسم، ننفّذ عملية الارتباط المتقاطع ثنائي الأبعاد، ونختبر كاشف الحواف المصمَّم يدويًا، وندع النزول التدرجي *يتعلّم* $(1, -1)$ مباشرةً من البيانات، ونبيّن كيف يتحول الالتفاف إلى عملية ضرب مصفوفات واحدة (*im2col*)، كما نشتق كيف يؤدي تكديس الطبقات إلى توسيع مجال الاستقبال للوحدة.

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
مع تنحية القنوات جانبًا في الوقت الراهن، لنفترض أن الدخل $\mathbf{X}$ مصفوفة بأبعاد $n_\textrm{h} \times n_\textrm{w}$ وأن نواة الالتفاف $\mathbf{K}$ مصفوفة بأبعاد $k_\textrm{h} \times k_\textrm{w}$. تُعرّف النواة *نافذة* منزلقة بشكل $k_\textrm{h} \times k_\textrm{w}$. وعند وضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$، نضرب رقعة الدخل المغطّاة عنصرًا بعنصر في $\mathbf{K}$ ونجمع حواصل الضرب لإنتاج عنصر الخرج عند $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
هذه هي عملية *الارتباط المتقاطع ثنائي الأبعاد* بين $\mathbf{X}$ و$\mathbf{K}$، وتُكتب $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ باستخدام فهارس تبدأ من الصفر بما يتوافق مع الشيفرة البرمجية. وبينما ركّز :numref:`sec_why-conv` الإزاحات بشكل متناظر من $-\Delta$ إلى $\Delta$، فإن التثبيت عند الزاوية العلوية اليسرى لا يختلف إلا بإزاحة ثابتة مقدارها $\Delta$، مع تعامله طبيعيًا مع النوى ذات الأبعاد الزوجية (مثل نواة $2 \times 2$ أو كاشف الحواف $1 \times 2$) التي لا تملك بكسلًا مركزيًا.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![الارتباط المتقاطع ثنائي الأبعاد. تمثّل الأجزاء المظللة عنصر الخرج الأول وعناصر الدخل والنواة المستخدَمة في حسابه: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
في :numref:`fig_correlation`، نُمرّر نواة بأبعاد $2 \times 2$ عبر دخل بأبعاد $3 \times 3$ بمقدار بكسل واحد في كل مرة، من اليسار إلى اليمين ومن الأعلى إلى الأسفل. وتنطبق النافذة تمامًا في أربعة مواضع، مما ينتج خرجًا بأبعاد $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
نلاحظ أن الخرج يتقلص قليلًا على امتداد كل محور لأن النواة يجب أن تبقى بالكامل داخل الدخل. فإسكان نافذة بعرض $k$ داخل طول $n$ يترك $n - k + 1$ موضعًا صالحًا، ولذلك يكون شكل الخرج

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
وهكذا فإن دخلًا بأبعاد $3 \times 3$ مع نواة $2 \times 2$ يعطي $2 \times 2$، وصورة بأبعاد $6 \times 8$ مع نواة $1 \times 2$ تعطي $6 \times 7$، وصورة من Fashion-MNIST بأبعاد $28 \times 28$ مع نواة $5 \times 5$ تعطي $24 \times 24$. ومتى أردنا الحفاظ على الأبعاد المكانية، نحيط الدخل بإطار من الأصفار (*الحشو*)، وهو ما ندرسه إلى جانب خطوة الانزلاق والتوسيع في :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### التنفيذ

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
ننفّذ :eqref:`eq_corr2d` مباشرةً في الدالة `corr2d` باستخدام حلقتين متداخلتين تمرّان على كل إحداثي من إحداثيات الخرج.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
للتحقق من `corr2d` مقارنةً بحسابنا اليدوي في :numref:`fig_correlation`، ننشئ موتر الدخل `X` بأبعاد $3 \times 3$ والنواة `K` بأبعاد $2 \times 2$ ثم نحسب `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
يطابق الخرج :numref:`fig_correlation` عنصرًا بعنصر (وكما بيّنا في :numref:`sec_why-conv`، فإن الالتفاف الرياضي الدقيق باستخدام نواة مقلوبة $\mathbf{K}'$ يتعلّم عائلة الطبقات نفسها تمامًا، لذا نتبع الاصطلاح الشائع في التعلم العميق ونطلق على الارتباط المتقاطع اسم «الالتفاف»).

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
تُجري الطبقة الالتفافية ثنائية الأبعاد ارتباطًا متقاطعًا بين دخلها ونواة التفاف مُتعلَّمة، ثم تضيف انحيازًا عدديًا قياسيًا. ومثل أوزان البيرسبترون متعدد الطبقات في :numref:`chap_perceptrons`، نهيّئ النواة من التوزيع $\mathcal{N}(0, 0.01^2)$ ونهيّئ الانحياز بالصفر. أدناه، يسجّل الصنف `Conv2D` كلتا المعلمتين ويستدعي `corr2d` في التمرير الأمامي؛ وتُسمّى الطبقة ذات النواة التي ارتفاعها $h$ وعرضها $w$ طبقة التفافية بأبعاد $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
على خلاف الطبقة الكثيفة، لا ترتبط الطبقة الالتفافية بدقة دخل ثابتة لأن معلماتها تعتمد على حجم النواة فحسب. فالطبقة نفسها بأبعاد $2 \times 2$ (5 معلمات إجمالًا) تعالج كلًا من موترنا `X` بأبعاد $3 \times 3$ وصورة بأبعاد $6 \times 8$، وتُخرج نواتج بأشكال $2 \times 2$ و$5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### كشف الحواف

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
نختبر الآن كاشف الحواف الرأسية الوارد في المقدمة على صورة اصطناعية بأبعاد $6 \times 8$، أعمدتها الأربعة الوسطى سوداء (`0`) وأعمدتها الخارجية بيضاء (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
نعرّف النواة بأبعاد $1 \times 2$ على النحو الآتي:

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
عند كل موضع $(i, j)$، تحسب هذه النواة *الفرق المحدود* الأفقي $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، وهو تقريب متقطع (بغضّ النظر عن الإشارة) للمشتق الأفقي للسطوع، إذ يتلاشى في المناطق المستوية ويقفز كلما تغيّرت شدة الإضاءة فجأةً.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
يُنتج إجراء الارتباط المتقاطع لصورة الشريط `X` بأبعاد $6 \times 8$ مع نواة الفرق الأفقي `K` بأبعاد $1 \times 2$ عبر `corr2d(X, K)` خريطة الحواف `Y` بأبعاد $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
وكما هو متوقع، تعطي `Y` القيمة `+1` على طول الحد الفاصل بين الأبيض والأسود، و`-1` على طول الحد الفاصل بين الأسود والأبيض، و`0` في ما عدا ذلك. أما تطبيق النواة نفسها تمامًا على منقول الصورة (مما يحوّل الشريط الرأسي إلى شريط أفقي) فيعطي أصفارًا بالكامل:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
من الواضح أن `[1, -1]` تستجيب للحواف الرأسية حصرًا. أما كشف الحواف الأفقية أو القطرية أو الزوايا أو الأنسجة فيتطلب نوى مختلفة تمامًا.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### تعلّم نواة التفاف

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
إن تصميم `[1, -1]` يدويًا أمر يسير عندما نعرف الميزة التي نريدها بدقة، غير أنه لا يمكن لأي مهندس تحديد آلاف المرشحات الوسيطة عبر شبكة عميقة يدويًا. فهل يستطيع النزول التدرجي استعادة النواة `[1, -1]` من زوج الدخل والخرج `(X, Y)` وحده؟ نختبر ذلك باستخدام الطبقة الالتفافية ثنائية الأبعاد المدمجة في كل إطار عمل (مع حذف الانحياز وتهيئة الأوزان من $\mathcal{N}(0, 0.01^2)$)، وذلك بتقليل مجموع الخطأ التربيعي عبر جميع عناصر الخرج الـ 42 على مدى 10 خطوات تدرج بمعدّل التعلّم $0.03$. ولأن الطبقات المدمجة تتوقع موترات رباعية الأبعاد تتضمن محوري الدفعة والقناة (:numref:`sec_channels`)، نعيد تشكيل `X` و`Y` وفقًا لذلك.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
في البداية تكون الخسارة قريبة من 12 (إذ تخطئ النواة العشوائية القريبة من الصفر جميع بكسلات الحواف الـ 12 ذات القيمة $\pm 1$ بمقدار يقارب 1)، وخلال تسع خطوات فقط تهبط إلى ما دون 0.01. ويؤكد فحص النواة المُتعلَّمة ما توصلت إليه:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
تستقر كلتا القيمتين ضمن مسافة تقارب 0.02 من `[1, -1]`. لم يكن في هذه الحلقة أي علم مسبق بالحواف: فبمجرد تزويده بإشارة خسارة، اكتشف النزول التدرجي المرشح من تلقاء نفسه. هنا أشرفنا على الطبقة الواحدة مباشرةً باستخدام `Y`؛ أما في الشبكة العصبية الالتفافية العميقة، فيتعلّم الانتشار العكسي نوى كل طبقة من الطرف إلى الطرف انطلاقًا من خسارة المهمة النهائية وحدها.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
الالتفاف للصور<br>**الارتباط المتقاطع ثنائي الأبعاد · كشف الحواف · النوى المُتعلَّمة · im2col · مجالات الاستقبال**

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
- **صورة اختبار اصطناعية $6 \times 8$:** أعمدة خارجية بيضاء (`1`) تحيط بشريط رأسي أسود من 4 أعمدة (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **الفروق المحدودة الأفقية:** تمرير $\mathbf{K} = [1, -1]$ يحسب $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، فيعطي `+1` عند حافة الأبيض إلى الأسود و`-1` عند حافة الأسود إلى الأبيض.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **التعلّم مقابل التصميم اليدوي:** تكتشف الشبكات العصبية الالتفافية العميقة آلاف النوى المماثلة تلقائيًا عبر النزول التدرجي.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="عملية الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[جداءات داخلية بنافذة منزلقة]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
بمعلومية مصفوفة دخل $\mathbf{X}$ بأبعاد $n_\textrm{h} \times n_\textrm{w}$ ونواة التفاف $\mathbf{K}$ بأبعاد $k_\textrm{h} \times k_\textrm{w}$، يضع الارتباط المتقاطع ثنائي الأبعاد $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ الزاوية العلوية اليسرى للنافذة عند $(i, j)$ ويجمع حواصل الضرب عنصرًا بعنصر:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **التثبيت العلوي الأيسر:** فهارس تبدأ من الصفر $a, b \ge 0$ تطابق فهرسة المصفوفات في الشيفرة وتتعامل بسلاسة مع النوى الزوجية الأبعاد (مثل $2 \times 2$ أو $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **الخرج العلوي الأيسر المظلّل $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="مواضع النافذة الصالحة والشكل المكاني للخرج"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[لماذا يقلّص الالتفاف دون حشو أبعاد الشبكة]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **مواضع النافذة الصالحة:** تنطبق نافذة $k_\textrm{h} \times k_\textrm{w}$ داخل شبكة $n_\textrm{h} \times n_\textrm{w}$ في $n - k + 1$ موضعًا على كل محور.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **أمثلة على الشكل:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)؛ و$28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="تنفيذ الارتباط المتقاطع ثنائي الأبعاد برمجيًا"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[التنفيذ المباشر بحلقات متداخلة]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
تحجز الدالة `corr2d` موتر الخرج بأبعاد $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ وتحسب كل عنصر كمجموع `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
في كل تكرار، تقتطع شريحة نافذة فرعية بأبعاد $k_\textrm{h} \times k_\textrm{w}$ من `X`، وتضربها عنصرًا بعنصر في `K`، ثم تختزلها عبر `sum`.

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
- **نواة مقلوبة بزاوية 180° $\mathbf{K}'$:** الارتباط المتقاطع مع $\mathbf{K}$ يكافئ الالتفاف مع $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **فئة مُتعلَّمة متطابقة:** تحسب أطر العمل الارتباط المتقاطع وتسميه التفافًا.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="بناء وحدة طبقة التفافية ثنائية الأبعاد"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[أوزان نواة قابلة للتعلّم مع انحياز عددي قياسي]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
تغلّف الطبقة الالتفافية الدالة `corr2d` بمعلمة وزن قابلة للتعلّم بأبعاد $k_\textrm{h} \times k_\textrm{w}$ (مُهيّأة من $\mathcal{N}(0, 0.01^2)$) وانحياز عددي قياسي (مُهيّأ بالصفر $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="استقلال الطبقات الالتفافية عن دقة الدخل"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[معلمات ثابتة عبر أي أحجام دخل]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **غير مقيدة بـ $H \times W$:** تعالج المعلمات الخمس نفسها (نواة $2 \times 2$ + انحياز واحد) مدخلات بأبعاد $3 \times 3$ أو $6 \times 8$ أو صورًا بملايين البكسلات دون إعادة تشكيل الأوزان.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="إنشاء صورة اصطناعية بحواف رأسية"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[نمط اختبار ثنائي بأبعاد 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
ننشئ صورة الاختبار `X` بأبعاد $6 \times 8$ بوحدات على الحافتين اليسرى واليمنى وأصفار عبر الأعمدة الأربعة الوسطى (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **أعمدة ثابتة جزئيًا:** تكون الشدة ثابتة (`1` أو `0`) داخل الأعمدة $0\text{--}1$ و$2\text{--}5$ و$6\text{--}7$، وتقفز فقط بين العمودين $1 \to 2$ و$5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="تطبيق نواة الفروق المحدودة الأفقية"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[كشف الحواف الرأسية باستخدام K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
يحسب الارتباط المتقاطع لـ `X` مع النواة $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ بأبعاد $1 \times 2$ القيمة $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ عند كل زوج أفقي:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **خريطة حواف ذات إشارة:** يعطي العمود 1 القيمة `+1` (انتقال من الأبيض إلى الأسود)، والعمود 5 القيمة `-1` (انتقال من الأسود إلى الأبيض)، وتُقيَّم جميع الأعمدة الداخلية الثابتة بـ `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="انتقائية الاتجاه في نوى الحواف"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[لماذا تعجز نواة واحدة عن كشف جميع الاتجاهات]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **عمياء عن الحواف الأفقية:** يجعل منقول `X` (`X.T`) الجيران الأفقيين متطابقين، فتُخرج $[1, -1]$ أصفارًا بالكامل.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **بنوك المرشحات (§{sec_channels}):** تتعلّم طبقات الرؤية قنوات متوازية للاتجاهات الأفقية والرأسية والقطرية.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="تعلّم كاشف حواف عبر النزول التدرجي"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[تدريب طبقة التفافية مدمجة بإشراف على (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **موترات أطر العمل رباعية الأبعاد:** أعد تشكيل `X` و`Y` إلى موترات رباعية الأبعاد (مثال واحد، قناة واحدة) وقلّل $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ بمعدّل التعلّم $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="فحص أوزان النواة المُتعلَّمة"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[لماذا تبدأ الخسارة عند 12 وتتقارب نحو [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **سبب بدء الخسارة عند $12.0$:** تتنبأ الأوزان القريبة من الصفر بـ $\hat{Y} \approx 0$، فتخطئ بكسلات الحواف الـ 12 ذات القيمة $\pm 1$ في `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **التقارب في 9 خطوات:** يستعيد النزول التدرجي العشوائي $[0.99, -0.99]$ تلقائيًا من البيانات.

=== DECISIONS ===
- Style conventions (no `ar/style.md` yet): Modern Standard Arabic (الفصحى العلمية الحديثة), inclusive first-person plural («نحن») in body prose, imperative in exercises, Arabic punctuation (`،` `؛` `؟`), Western Arabic numerals (`0–9`), Latin identifiers/acronyms (`CNN`, `SGD`, `GPU`, `GEMM`, `im2col`, `Fashion-MNIST`, `Python`) retained in Latin.
- "edge map" → «خريطة الحواف» / «خريطة حواف»
- "edge detector" → «كاشف الحواف»
- "locality" → «المحلية»
- "window" (sliding window) → «نافذة» (نافذة منزلقة)
- "patch" → «رقعة»
- "dense layer" → «طبقة كثيفة»
- "kernel" → «نواة الالتفاف» on first/explicit use, shortened naturally to «النواة» / «نواة» (plural «نوى») in subsequent occurrences.
- "filter banks" → «بنوك المرشحات»

=== NOTES ===
- 55f0a41b, 29cd9061: "returns" / "returning" are ordinary verbs (يعود / وتُخرج), not the reinforcement-learning glossary term *return* (عائد).
- 8d45aae8, 9be85019: math tokens reordered because Arabic places dimensions after the named matrix/kernel (e.g., `مصفوفة دخل $\mathbf{X}$ بأبعاد $n_\textrm{h} \times n_\textrm{w}$`).
- 53a2bce7: Arabic expresses small counts grammatically as words («المعلمات الخمس»، «انحياز واحد») rather than bare digits `5` and `1` (false-match warning).
