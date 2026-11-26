<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# الالتفافات على الصور

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
في :numref:`sec_why-conv` اشتققنا الطبقة الالتفافية (convolutional layer) نظريًا من مبدأين هندسيين: تساوق الإزاحة (translation equivariance) والمحلية. والآن سنرى ما الذي تحسبه تلك الصيغة فعليًا على صورة، وهل يستطيع النزول التدرجي (gradient descent) أن يكتشف نوى مفيدة من البيانات تلقائيًا.

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
لنفترض أننا نريد تحديد الحواف العمودية، أي المواضع التي تلتقي فيها منطقة ساطعة بمنطقة معتمة. طرح الجار الأيمن من كل بكسل ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) يعطي صفرًا داخل المناطق المتجانسة، و$+1$ حيث ينتقل اللون من الأبيض إلى الأسود، و$-1$ حيث يعود من الأسود إلى الأبيض. وبتمرير الكاشف ذي الوزنين $(1, -1)$ على كل موضع صالح في الصورة نحصل على *خريطة الحواف* (edge map). سنطبّق في هذا القسم الارتباط المتقاطع (cross-correlation) ثنائي الأبعاد، ونختبر كاشف الحواف الذي صممناه يدويًا، ثم ندع النزول التدرجي *يتعلم* $(1, -1)$ مباشرة من البيانات، ونبيّن كيف يصبح الالتفاف ضربَ مصفوفات واحدًا (*im2col*)، ونشتق كيف يؤدي تكديس الطبقات إلى توسيع مجال الاستقبال (receptive field) لوحدة ما.

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
لنُرجئ القنوات (channels) إلى حين، ولنفترض أن المدخل $\mathbf{X}$ مصفوفة (matrix) حجمها $n_\textrm{h} \times n_\textrm{w}$ وأن النواة $\mathbf{K}$ مصفوفة حجمها $k_\textrm{h} \times k_\textrm{w}$. تعرّف النواة *نافذة* منزلقة بشكل $k_\textrm{h} \times k_\textrm{w}$. نضع الزاوية العليا اليسرى للنافذة عند $(i, j)$، ونضرب رقعة المدخل التي تغطيها نافذةً عنصرًا بعنصر في $\mathbf{K}$، ثم نجمع حواصل الضرب لنحصل على عنصر الخرج عند $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
هذا هو *الارتباط المتقاطع ثنائي الأبعاد* لـ $\mathbf{X}$ مع $\mathbf{K}$، ونكتبه $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ بفهارس تبدأ من الصفر بما يوافق الشيفرة. وبينما كان :numref:`sec_why-conv` يوزّع الإزاحات تناظريًا حول المركز من $-\Delta$ إلى $\Delta$، فإن التثبيت عند الزاوية العليا اليسرى لا يختلف عنه إلا بإزاحة ثابتة مقدارها $\Delta$، ويعالج بشكل طبيعي النوى ذات الأحجام الزوجية (مثل نواة $2 \times 2$ أو كاشف الحواف $1 \times 2$ لدينا) التي لا يوجد لها بكسل مركزي.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![الارتباط المتقاطع ثنائي الأبعاد. الأجزاء المظللة هي عنصر الخرج الأول وعناصر المدخل والنواة المستخدمة في حسابه: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
في :numref:`fig_correlation` نمرّر نواة $2 \times 2$ على مدخل $3 \times 3$ بمقدار بكسل واحد في كل مرة، من اليسار إلى اليمين ومن الأعلى إلى الأسفل. تتسع النافذة بشكل كامل في أربعة مواضع، فينتج خرج $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
لاحظ أن الخرج ينكمش قليلًا على طول كل محور، لأن النواة يجب أن تبقى داخل المدخل بالكامل. فإدخال نافذة عرضها $k$ في طول $n$ يترك $n - k + 1$ موضعًا صالحًا، ومن ثَمّ يكون شكل (shape) الخرج

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
وهكذا يعطي مدخل $3 \times 3$ مع نواة $2 \times 2$ خرجًا $2 \times 2$، وصورة $6 \times 8$ مع نواة $1 \times 2$ خرجًا $6 \times 7$، وصورة Fashion-MNIST بحجم $28 \times 28$ مع نواة $5 \times 5$ خرجًا $24 \times 24$. وحين نريد الحفاظ على الأبعاد المكانية نحيط المدخل بإطار من الأصفار (*الحشو*، padding)، وسندرسه مع خطوة الانزلاق والتوسيع في :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### التنفيذ

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
ننفّذ :eqref:`eq_corr2d` مباشرةً في الدالة `corr2d` باستخدام حلقتين تمران على كل إحداثي من إحداثيات الخرج.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
وللتحقق من `corr2d` مقابل حسابنا اليدوي في :numref:`fig_correlation`، ننشئ موتر (tensor) المدخل `X` بحجم $3 \times 3$ والنواة `K` بحجم $2 \times 2$، ثم نقيّم `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
يعيد الخرج إنتاج :numref:`fig_correlation` عنصرًا بعنصر (وكما رأينا في :numref:`sec_why-conv`، فإن الالتفاف الرياضي الصارم مع نواة مقلوبة $\mathbf{K}'$ يتعلم الأسرة نفسها تمامًا من الطبقات، ولذلك نتبع المصطلح المعتاد في التعلم العميق ونسمّي الارتباط المتقاطع "التفافًا").

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
تطبّق الطبقة الالتفافية ثنائية الأبعاد الارتباط المتقاطع بين مدخلها ونواة متعلَّمة، ثم تضيف انحيازًا (bias) عدديًا. وكما فعلنا مع أوزان بيرسبترون متعدد الطبقات (MLP) في :numref:`chap_perceptrons`، نهيّئ النواة من $\mathcal{N}(0, 0.01^2)$ والانحياز بالصفر. في ما يلي تسجّل `Conv2D` المعاملين كليهما وتستدعي `corr2d` في تمريرها الأمامي (forward pass)؛ وتسمى الطبقة التي ارتفاع نواتها $h$ وعرضها $w$ طبقة التفافية $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
وعلى خلاف الطبقة الكثيفة، لا ترتبط الطبقة الالتفافية بدقة مدخل ثابتة، لأن معاملاتها تعتمد على حجم النواة (kernel size) فقط. فالطبقة $2 \times 2$ نفسها (5 معاملات في المجموع) تعالج موتر `X` بحجم $3 \times 3$ وصورة $6 \times 8$ على حد سواء، وتعيد خرجين بشكلين $2 \times 2$ و$5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### كشف الحواف

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
نختبر الآن كاشف الحواف العمودية الوارد في المقدمة على صورة اصطناعية بحجم $6 \times 8$ أعمدتها الأربعة الوسطى سوداء (`0`) وأعمدتها الخارجية بيضاء (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
نعرّف النواة $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
عند كل موضع $(i, j)$ تحسب هذه النواة *الفرق المحدود* (finite difference) الأفقي $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، وهو تقريب متقطع (حتى الإشارة) للمشتق الأفقي للسطوع، ينعدم عبر المناطق المسطحة ويقفز حيثما قفزت الشدة.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
يؤدي الارتباط المتقاطع لصورة الأشرطة `X` بحجم $6 \times 8$ مع نواة الفرق الأفقي `K` بحجم $1 \times 2$ عبر `corr2d(X, K)` إلى خريطة الحواف `Y` بحجم $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
وكما هو متوقع، تعطي `Y` القيمة `+1` على طول الحد بين الأبيض والأسود، والقيمة `-1` على طول الحد بين الأسود والأبيض، و`0` في كل موضع آخر. أما تطبيق النواة نفسها على الصورة المنقولة (فيتحول الشريط العمودي إلى شريط أفقي) فيعطي أصفارًا كلها:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
من الواضح أن `[1, -1]` تستجيب للحواف العمودية فقط. أما كشف الحواف الأفقية والأقطار والزوايا والأنسجة فيتطلب نوى مختلفة تمامًا.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### تعلّم نواة

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
من السهل أن نصمم `[1, -1]` يدويًا حين نعرف السمة المطلوبة بدقة، لكن لا يستطيع أي مهندس أن يحدد يدويًا آلاف المرشحات (filters) الوسيطة عبر شبكة عميقة. فهل يستطيع النزول التدرجي استرداد النواة `[1, -1]` من زوج المدخل والخرج `(X, Y)` وحدهما؟ نختبر ذلك باستخدام الطبقة الالتفافية ثنائية الأبعاد المدمجة في كل إطار عمل (من دون انحياز، وبأوزان مهيأة من $\mathcal{N}(0, 0.01^2)$)، فنقلّل الخطأ التربيعي (squared error) المجموع على عناصر الخرج الـ 42 كلها خلال 10 خطوات تدرج بمعدّل تعلّم (learning rate) $0.03$. ولأن الطبقات المدمجة تتوقع موترات رباعية الأبعاد تتضمن محوري الدفعة والقناة (:numref:`sec_channels`)، نعيد تشكيل `X` و`Y` وفقًا لذلك.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
في البداية تكون الخسارة (loss) قريبة من 12 (فالنواة العشوائية القريبة من الصفر تخطئ جميع بكسلات الحواف الاثني عشر البالغة $\pm 1$ بنحو 1 لكل منها)، وفي تسع خطوات فقط تنخفض إلى ما دون 0.01. ويؤكد فحص النواة المتعلَّمة ما وجدته:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
يقع كلا العنصرين ضمن نحو 0.02 من `[1, -1]`. لم يكن في هذه الحلقة ما يعرف شيئًا عن الحواف: فمن إشارة الخسارة وحدها اكتشف النزول التدرجي المرشح بنفسه. وقد أشرفنا هنا على الطبقة الوحيدة مباشرةً باستخدام `Y`؛ أما في شبكة عصبية التفافية (CNN) عميقة فيتعلم الانتشار العكسي (backpropagation) نوى كل طبقة من البداية إلى النهاية انطلاقًا من خسارة المهمة النهائية وحدها.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
الالتفافات على الصور<br>**الارتباط المتقاطع ثنائي الأبعاد · كشف الحواف · النوى المتعلَّمة · im2col · مجالات الاستقبال**

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
- **صورة اختبار اصطناعية $6 \times 8$:** أعمدة خارجية بيضاء (`1`) تحيط بشريط عمودي أسود (`0`) من أربعة أعمدة.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **الفرق المحدود الأفقي:** تمرير $\mathbf{K} = [1, -1]$ يحسب $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، فيعطي `+1` عند حواف الانتقال من الأبيض إلى الأسود و`-1` عند الانتقال من الأسود إلى الأبيض.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **التعلّم مقابل التصميم اليدوي:** تكتشف الشبكات الالتفافية العميقة آلاف هذه النوى تلقائيًا بالنزول التدرجي.

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
بالنظر إلى مصفوفة مدخل $\mathbf{X}$ بحجم $n_\textrm{h} \times n_\textrm{w}$ ونواة $\mathbf{K}$ بحجم $k_\textrm{h} \times k_\textrm{w}$، يضع الارتباط المتقاطع ثنائي الأبعاد $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ الزاوية العليا اليسرى للنافذة عند $(i, j)$ ويجمع حواصل الضرب عنصرًا بعنصر:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **التثبيت عند الزاوية العليا اليسرى:** الفهارس التي تبدأ من الصفر $a, b \ge 0$ توافق فهرسة المصفوفات في الشيفرة وتعالج النوى ذات الأحجام الزوجية (مثل $2 \times 2$ أو $1 \times 2$) بسلاسة.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **خرج الزاوية العليا اليسرى المظلل $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="مواضع النافذة الصالحة وشكل الخرج المكاني"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[لماذا تُنقص الالتفافات غير المحشوة الشبكة]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **مواضع النافذة الصالحة:** تتسع نافذة $k_\textrm{h} \times k_\textrm{w}$ داخل شبكة $n_\textrm{h} \times n_\textrm{w}$ في $n - k + 1$ موضعًا على كل محور.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **أمثلة على الأشكال:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)؛ $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="تنفيذ الارتباط المتقاطع ثنائي الأبعاد في الشيفرة"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[تنفيذ مباشر بحلقات متداخلة]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
تخصص الدالة `corr2d` موتر الخرج بحجم $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ وتحسب كل عنصر فيه بوصفه مجموع `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
تقتطع كل دورة نافذة فرعية بحجم $k_\textrm{h} \times k_\textrm{w}$ من `X`، وتضربها عنصرًا بعنصر في `K`، ثم تختزلها بالدالة `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="التحقق من corr2d ومكافئة النواة المقلوبة"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[الارتباط المتقاطع مقابل الالتفاف الرياضي]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **النواة $\mathbf{K}'$ المقلوبة بزاوية 180°:** الارتباط المتقاطع مع $\mathbf{K}$ يساوي الالتفاف مع $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **الصنف المتعلَّم نفسه:** تحسب أطر العمل الارتباط المتقاطع وتسميه التفافًا.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="بناء وحدة طبقة التفافية ثنائية الأبعاد"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[أوزان نواة قابلة للتعلم مع انحياز عددي]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
تغلّف الطبقة الالتفافية الدالة `corr2d` بمعامل وزن قابل للتعلم بحجم $k_\textrm{h} \times k_\textrm{w}$ (مهيأ من $\mathcal{N}(0, 0.01^2)$) وانحياز عددي (مهيأ بالقيمة $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="استقلال الطبقات الالتفافية عن الدقة"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[معاملات ثابتة عبر أحجام مدخلات اعتباطية]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **منفصلة عن $H \times W$:** المعاملات الخمسة نفسها (نواة $2 \times 2$ + انحياز واحد) تعالج مدخلات $3 \times 3$ أو $6 \times 8$ أو بحجم ملايين البكسلات دون إعادة تشكيل الأوزان.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="بناء صورة اصطناعية للحواف العمودية"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[نمط اختبار ثنائي بحجم 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
ننشئ صورة الاختبار `X` بحجم $6 \times 8$ بوضع الواحدات على الحدين الأيسر والأيمن والأصفار في الأعمدة الأربعة الوسطى (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **أعمدة ثابتة قطعيًا:** الشدة ثابتة (`1` أو `0`) داخل الأعمدة $0\text{--}1$ و$2\text{--}5$ و$6\text{--}7$، ولا تقفز إلا بين العمودين $1 \to 2$ و$5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="تطبيق نواة الفرق المحدود الأفقي"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[كشف الحواف العمودية بالنواة K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
يحسب الارتباط المتقاطع لـ `X` مع النواة $1 \times 2$ ذات الصيغة $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ القيمة $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ عند كل زوج أفقي:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **خريطة حواف ذات إشارة:** يعطي العمود 1 القيمة `+1` (انتقال من الأبيض إلى الأسود)، ويعطي العمود 5 القيمة `-1` (انتقال من الأسود إلى الأبيض)، وتساوي جميع الأعمدة الداخلية الثابتة `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="انتقائية الاتجاه لنوى الحواف"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[لماذا لا تكشف نواة واحدة جميع الاتجاهات]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **عمياء عن الحواف الأفقية:** نقل `X` (`X.T`) يجعل الجيران الأفقيين متطابقين، فتعطي $[1, -1]$ أصفارًا كلها.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **بنوك المرشحات (§{sec_channels}):** تتعلم طبقات الرؤية قنوات متوازية للاتجاهات الأفقية والعمودية والقطرية.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="تعلّم كاشف حواف بالنزول التدرجي"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[الإشراف على طبقة التفافية مدمجة باستخدام (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **موترات رباعية الأبعاد في أطر العمل:** أعد تشكيل `X` و`Y` إلى موترين رباعيي الأبعاد (مثال واحد وقناة واحدة) وقلّل $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ عند $\eta = 0.03$.

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
- **لماذا تبدأ الخسارة عند $12.0$:** الأوزان القريبة من الصفر تتنبأ بـ $\hat{Y} \approx 0$، فتخطئ بكسلات الحواف الاثني عشر البالغة $\pm 1$ في `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **التقارب في 9 خطوات:** يسترد النزول التدرجي العشوائي (SGD) القيمتين $[0.99, -0.99]$ تلقائيًا من البيانات.

=== DECISIONS ===
- Register: formal Modern Standard Arabic, "نحن" voice, Western digits (0, 1, 2.5), Arabic comma "،" and "؛" in prose; math, code and Latin names unchanged. Key terms get the English in parentheses at first use in the page (once), then Arabic only.
- "kernel" → النواة (short for glossary "نواة الالتفاف"); "kernel size" → حجم النواة.
- "cross-correlation" → الارتباط المتقاطع; "edge map" → خريطة الحواف; "edge detector" → كاشف الحواف; "feature map" → خريطة الميزات; "window" → نافذة; "patch" → رقعة; "unit" (a neuron/entry of a layer) → وحدة; "receptive field" → مجال الاستقبال, "effective receptive field" → مجال الاستقبال الفعّال, "theoretical" → النظري.
- "two-dimensional" → ثنائي الأبعاد; "elementwise" → عنصرًا بعنصر; "output/input" → الخرج/المدخل; "framework" → إطار العمل.
- "padding" → الحشو; "stride" → خطوة الانزلاق; "dilation" → التوسيع; "im2col" stays Latin.
- "im2col"-style English names, `corr2d`, framework names stay as in English; slide kickers/titles are translated, code names inside titles are kept Latin.
- Layer "bias" → الانحياز; "filter bank" → بنك المرشحات; "workspace memory" → ذاكرة عمل مؤقتة.
- Unit "u01" does not translate the equation that follows "نافذة ... :" paragraphs; those blocks end with a colon by design.

=== NOTES ===
- 29cd9061 (English): says the $2\times2$ layer returns $2\times2$ and $5\times7$ for a $6\times8$ image; $6\times8$ with $2\times2$ gives $5\times7$, correct.
- f0f30f76 and similar slide blocks keep `§{sec_channels}` unchanged inside parentheses.
