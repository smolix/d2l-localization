<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# الالتفاف على الصور

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
اشتققنا في :numref:`sec_why-conv` صيغة الطبقة الالتفافية على الورق انطلاقًا من مبدأين هندسيين هما تساوق الإزاحة والمحلية. لنرَ الآن ما تحسبه هذه الصيغة فعليًا عند تطبيقها على صورة، وهل يستطيع النزول التدرجي اكتشاف نوى التفاف مفيدة تلقائيًا من البيانات.

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
لنفكر في كيفية تحديد مواضع الحواف الرأسية حيث تلتقي منطقة ساطعة بأخرى داكنة. إذا طرحنا من قيمة كل بكسل قيمة جاره الأيمن ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)، حصلنا على صفر داخل المناطق المتجانسة، وعلى $+1$ حيث يتحول الأبيض إلى الأسود، وعلى $-1$ حيث يعود الأسود إلى الأبيض. وبإمرار الكاشف ذي الوزنين $(1, -1)$ على جميع المواضع الصالحة في الصورة نحصل على *خريطة حواف*. في هذا القسم ننفّذ الارتباط المتقاطع ثنائي الأبعاد، ونختبر كاشف الحواف الذي صمّمناه يدويًا، ثم ندع خوارزمية النزول التدرجي *تتعلّم* $(1, -1)$ مباشرة من البيانات. ونبيّن كيف نعبّر عن الالتفاف بعملية واحدة من ضرب المصفوفات (*im2col*)، ونشتق مقدار اتساع مجال استقبال الوحدة عند تكديس الطبقات.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## الارتباط المتقاطع

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### طريقة الحساب

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
لندع القنوات جانبًا الآن. لنفترض أن المدخل $\mathbf{X}$ مصفوفة أبعادها $n_\textrm{h} \times n_\textrm{w}$، وأن نواة الالتفاف $\mathbf{K}$ مصفوفة أبعادها $k_\textrm{h} \times k_\textrm{w}$. تحدّد النواة *نافذة* منزلقة شكلها $k_\textrm{h} \times k_\textrm{w}$. نضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$، ونضرب كل عنصر من رقعة المدخل التي تغطّيها النافذة في العنصر المقابل له من $\mathbf{K}$، ثم نجمع حواصل الضرب لنحصل على عنصر المخرج عند $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
هذا هو *الارتباط المتقاطع ثنائي الأبعاد* بين $\mathbf{X}$ و$\mathbf{K}$، ونكتبه $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ بفهارس تبدأ من الصفر كما في الشيفرة. في :numref:`sec_why-conv` قسنا الإزاحات حول المركز بصورة متناظرة من $-\Delta$ إلى $\Delta$. ولا يختلف اتخاذ الزاوية العلوية اليسرى مرجعًا إلا بإزاحة ثابتة مقدارها $\Delta$، وهو يتيح لنا مباشرةً استخدام نوى ذات أبعاد زوجية (مثل نواة أبعادها $2 \times 2$ أو كاشف الحواف ذي الأبعاد $1 \times 2$) لا يتوسطها بكسل.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![الارتباط المتقاطع ثنائي الأبعاد. يبيّن التظليل أول عنصر من المخرج وعناصر المدخل والنواة المستخدمة في حسابه: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
في :numref:`fig_correlation`، نحرّك نواة أبعادها $2 \times 2$ على مدخل أبعاده $3 \times 3$ بمقدار بكسل واحد في كل مرة، من اليسار إلى اليمين ومن الأعلى إلى الأسفل. وتقع النافذة بأكملها داخل المدخل في أربعة مواضع، فنحصل على مخرج أبعاده $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
لاحظ أن المخرج ينكمش قليلًا على كل محور، إذ يجب أن تبقى النواة بكاملها داخل المدخل. ولنافذة عرضها $k$ ضمن امتداد طوله $n$ عدد من المواضع الصالحة يساوي $n - k + 1$، ومن ثم يكون شكل المخرج

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
وهكذا يعطي مدخل أبعاده $3 \times 3$ مع نواة أبعادها $2 \times 2$ مخرجًا أبعاده $2 \times 2$. وتعطي صورة أبعادها $6 \times 8$ مع نواة أبعادها $1 \times 2$ مخرجًا أبعاده $6 \times 7$، بينما تعطي صورة أبعادها $28 \times 28$ من Fashion-MNIST مع نواة أبعادها $5 \times 5$ مخرجًا أبعاده $24 \times 24$. وحين نريد الحفاظ على الأبعاد المكانية، نحيط المدخل بإطار من الأصفار (*الحشو*)، وسندرس الحشو مع خطوة الانزلاق والتوسيع في :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### التنفيذ

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
ننفّذ :eqref:`eq_corr2d` مباشرة في الدالة `corr2d` بحلقتين متداخلتين تمرّان بجميع إحداثيات المخرج.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
للتحقق من أن نتائج `corr2d` توافق حسابنا اليدوي في :numref:`fig_correlation`، ننشئ موتر المدخل ذا الأبعاد $3 \times 3$ ونسميه `X`، ونواةً أبعادها $2 \times 2$ ونسميها `K`، ثم نحسب `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
يطابق المخرج ما في :numref:`fig_correlation` عنصرًا بعنصر. وكما رأينا في :numref:`sec_why-conv`، يتيح الالتفاف بمعناه الرياضي الدقيق، بنواة مقلوبة $\mathbf{K}'$، تعلّم عائلة الطبقات نفسها تمامًا. لذا نتبع الاصطلاح الشائع في التعلّم العميق ونسمّي الارتباط المتقاطع «التفافًا».

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## الطبقات الالتفافية

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### بنية الطبقة

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
تحسب الطبقة الالتفافية ثنائية الأبعاد الارتباط المتقاطع بين مدخلها ونواة تتعلّمها، ثم تضيف انحيازًا قياسيًا. وكما فعلنا مع أوزان البيرسبترون متعدد الطبقات في :numref:`chap_perceptrons`، نهيّئ أوزان النواة بسحبها من $\mathcal{N}(0, 0.01^2)$، ونهيّئ الانحياز بالصفر. في الشيفرة التالية يسجّل الصنف `Conv2D` هاتين المعلمتين ويستدعي `corr2d` في التمرير الأمامي. ونسمّي الطبقة التي يبلغ ارتفاع نواتها $h$ وعرضها $w$ طبقةً التفافيةً بأبعاد $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
بخلاف الطبقة الكثيفة، تستطيع الطبقة الالتفافية معالجة مدخلات متفاوتة الدقة لأن معلماتها تعتمد على حجم النواة وحده. فالطبقة نفسها ذات الأبعاد $2 \times 2$ (بما مجموعه 5 معلمات) تعالج الموتر ذا الأبعاد $3 \times 3$ المسمّى `X`، وكذلك صورة أبعادها $6 \times 8$، وتعطي مخرجين شكلهما $2 \times 2$ و$5 \times 7$ على الترتيب:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### كشف الحواف

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
نختبر الآن كاشف الحواف الرأسية الذي عرضناه في المقدمة على صورة اصطناعية أبعادها $6 \times 8$، أعمدتها الأربعة الوسطى سوداء (`0`) وأعمدتها الخارجية بيضاء (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
نعرّف النواة ذات الأبعاد $1 \times 2$ كما يلي:

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
عند كل موضع $(i, j)$، تحسب هذه النواة *الفرق المحدود* الأفقي $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. وهو تقريب متقطع للمشتق الأفقي للسطوع، مع احتمال اختلاف الإشارة. ينعدم المشتق في المناطق المتجانسة ويبلغ قيمًا كبيرة بالمقدار حيثما تتغير شدة الإضاءة فجأة.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
نحسب الارتباط المتقاطع بين صورة الشريط ذات الأبعاد $6 \times 8$، أي `X`، ونواة الفروق الأفقية ذات الأبعاد $1 \times 2$، أي `K`، باستدعاء `corr2d(X, K)`، فنحصل على خريطة الحواف ذات الأبعاد $6 \times 7$ المسمّاة `Y`:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
كما توقعنا، تأخذ `Y` القيمة `+1` على الحد الفاصل من الأبيض إلى الأسود، و`-1` على الحد الفاصل من الأسود إلى الأبيض، و`0` في سائر المواضع. وإذا طبّقنا النواة نفسها على منقول الصورة (فيصبح الشريط الرأسي أفقيًا)، حصلنا على أصفار فقط:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
يتّضح أن `[1, -1]` لا يستجيب إلا للحواف الرأسية. أمّا كشف الحواف الأفقية أو القطرية أو الزوايا أو أنماط النسيج فيتطلب نوى مختلفة تمامًا.

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
يسهل تصميم `[1, -1]` يدويًا حين نعرف على وجه الدقة الميزة التي نريدها، لكن لا يسع أي مهندس أن يحدّد يدويًا آلاف المرشحات الوسيطة الموزعة على طبقات شبكة عميقة. فهل يستطيع النزول التدرجي استعادة النواة `[1, -1]` من زوج المدخل والمخرج `(X, Y)` وحده؟ نختبر ذلك باستخدام الطبقة الالتفافية ثنائية الأبعاد المضمّنة في كل إطار عمل، من دون انحياز وبأوزان نهيّئها من $\mathcal{N}(0, 0.01^2)$. ونقلّل مجموع الخطأ التربيعي على عناصر المخرج البالغ عددها 42، في 10 خطوات من النزول التدرجي مع ضبط معدّل التعلّم على $0.03$. ولأن هذه الطبقات تستقبل موترات رباعية الأبعاد تتضمّن محورًا للدفعة وآخر للقناة (:numref:`sec_channels`)، نغيّر شكل كل من `X` و`Y` وفقًا لذلك.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
تبدأ الخسارة بقيمة تقارب 12، إذ تعطي النواة العشوائية التي تقارب أوزانها الصفر تنبؤًا يخطئ بنحو 1 عند كل بكسل من بكسلات الحواف الاثني عشر ذات القيم $\pm 1$. وفي غضون تسع خطوات فقط تهبط الخسارة إلى أقل من 0.01. ولنتفحّص النواة لنتأكد مما تعلّمته:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
لا تبعد قيمة أيٍّ من عنصري النواة عن القيمة المناظرة لها في `[1, -1]` بأكثر من نحو 0.02. لم نزوّد حلقة التدريب بأي معرفة عن الحواف، بل اكتشف النزول التدرجي المرشح بنفسه مستعينًا بإشارة الخسارة. هنا أشرفنا على تدريب الطبقة الوحيدة مباشرةً باستخدام `Y`. أمّا في الشبكة العصبية الالتفافية العميقة، فيتيح الانتشار العكسي تعلّم نوى جميع الطبقات معًا من أول الشبكة إلى آخرها اعتمادًا على خسارة المهمة النهائية وحدها.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
الالتفاف على الصور<br>**الارتباط المتقاطع ثنائي الأبعاد · كشف الحواف · النوى المتعلَّمة · im2col · مجالات الاستقبال**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="كشف الحواف المحلية في أنحاء الصورة"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[الدافع والفكرة الأساسية]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **صورة اختبار اصطناعية بأبعاد $6 \times 8$:** أعمدة بيضاء خارجية (`1`) تحيط بشريط رأسي أسود عرضه 4 أعمدة (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **الفرق المحدود الأفقي:** نحسب بإمرار $\mathbf{K} = [1, -1]$ الفرق $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، فنحصل على `+1` عند الانتقال من الأبيض إلى الأسود و`-1` عند الانتقال من الأسود إلى الأبيض.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **التعلّم والتصميم اليدوي:** تكتشف الشبكات العصبية الالتفافية العميقة آلاف النوى من هذا النوع تلقائيًا بالنزول التدرجي.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="عملية الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[الجداء الداخلي في نافذة منزلقة]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
إذا كانت أبعاد مصفوفة المدخل $n_\textrm{h} \times n_\textrm{w}$ ورمزها $\mathbf{X}$، وأبعاد النواة $k_\textrm{h} \times k_\textrm{w}$ ورمزها $\mathbf{K}$، نحسب الارتباط المتقاطع ثنائي الأبعاد $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ بوضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$ وجمع حواصل ضرب العناصر المتناظرة:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **الزاوية العلوية اليسرى مرجعًا:** توافق الفهارس $a, b \ge 0$ التي تبدأ من الصفر فهرسة المصفوفات في الشيفرة وتسهّل التعامل مع النوى ذات الأبعاد الزوجية (مثل $2 \times 2$ أو $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **عنصر المخرج المظلّل في أعلى اليسار $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="مواضع النافذة الصالحة والشكل المكاني للمخرج"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[لماذا تنكمش الشبكة بعد الالتفاف من دون حشو؟]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **مواضع النافذة الصالحة:** لنافذة أبعادها $k_\textrm{h} \times k_\textrm{w}$ داخل شبكة أبعادها $n_\textrm{h} \times n_\textrm{w}$ ما مجموعه $n - k + 1$ موضعًا صالحًا على كل محور.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **أمثلة على شكل المخرج:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)، و$28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="برمجة الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[التنفيذ المباشر بحلقتين متداخلتين]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
تخصّص الدالة `corr2d` موتر المخرج ذا الأبعاد $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$، وتحسب كل عنصر فيه بجمع عناصر `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
نأخذ في كل تكرار نافذة فرعية أبعادها $k_\textrm{h} \times k_\textrm{w}$ من `X`، ونضرب عناصرها في العناصر المناظرة من `K`، ثم نجمعها باستخدام `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="التحقق من corr2d والتكافؤ عند قلب النواة"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[الارتباط المتقاطع والالتفاف بمعناه الرياضي]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **النواة $\mathbf{K}'$ بعد تدويرها 180°:** الارتباط المتقاطع مع $\mathbf{K}$ يكافئ الالتفاف مع $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **الفئة نفسها من الطبقات المتعلَّمة:** تحسب أطر العمل الارتباط المتقاطع وتسمّيه التفافًا.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="بناء وحدة برمجية لطبقة التفافية ثنائية الأبعاد"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[أوزان نواة قابلة للتعلّم وانحياز قياسي]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
تجمع الطبقة الالتفافية بين `corr2d` ومعلمة أوزان قابلة للتعلّم أبعادها $k_\textrm{h} \times k_\textrm{w}$ (تُهيّأ من $\mathcal{N}(0, 0.01^2)$) وانحياز قياسي (يُهيّأ بالقيمة $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="استقلال الطبقات الالتفافية عن دقة المدخل"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[معلمات ثابتة مهما تغيّر حجم المدخل]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **الاستقلال عن $H \times W$:** تعالج المعلمات الخمس نفسها (نواة أبعادها $2 \times 2$ وانحياز واحد) مدخلات بأبعاد $3 \times 3$ أو $6 \times 8$ أو بدقة ميغابكسل من دون تغيير شكل الأوزان.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="إنشاء صورة اصطناعية ذات حواف رأسية"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[نمط اختبار ثنائي بأبعاد 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
ننشئ صورة الاختبار ذات الأبعاد $6 \times 8$ ونسمّيها `X`. ونضع فيها آحادًا عند الحدَّين الأيسر والأيمن وأصفارًا في الأعمدة الأربعة الوسطى (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **ثبات الشدة ضمن كل مجموعة أعمدة:** تبقى الشدة ثابتة (`1` أو `0`) داخل كلٍّ من مجموعات الأعمدة $0\text{--}1$ و$2\text{--}5$ و$6\text{--}7$، ولا تتغيّر إلا عند الانتقال $1 \to 2$ أو $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="تطبيق نواة الفرق المحدود الأفقي"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[كشف الحواف الرأسية باستخدام K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
يحسب الارتباط المتقاطع بين `X` والنواة ذات الأبعاد $1 \times 2$، أي $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$، الفرق $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ لكل زوج من البكسلات المتجاورة أفقيًا:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **خريطة حواف بإشاراتها:** نقرأ في العمود 1 القيمة `+1` (من الأبيض إلى الأسود)، وفي العمود 5 القيمة `-1` (من الأسود إلى الأبيض)، وفي سائر الأعمدة الداخلية الثابتة القيمة `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="انتقائية نوى كشف الحواف للاتجاه"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[لماذا لا تكشف نواة واحدة الحواف في جميع الاتجاهات؟]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **لا استجابة للحواف الأفقية:** عند أخذ منقول `X` (`X.T`) تتساوى قيم البكسلات المتجاورة أفقيًا، فتكون مخرجات $[1, -1]$ أصفارًا كلها.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **مجموعات المرشحات (§{sec_channels}):** تتعلّم طبقات الرؤية قنوات متوازية للاتجاهات الأفقية والرأسية والقطرية.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="تعلّم كاشف للحواف بالنزول التدرجي"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[التدريب الموجّه لطبقة التفافية مضمّنة باستخدام (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **الموترات رباعية الأبعاد في أطر العمل:** نغيّر شكل `X` و`Y` ليصبحا موترين رباعيي الأبعاد (مثال واحد وقناة واحدة)، ونقلّل $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ باستخدام $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="فحص أوزان النواة بعد التعلّم"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[لماذا تبدأ الخسارة عند 12 وتتقارب الأوزان نحو [1, -1]؟]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **لماذا تبدأ الخسارة عند $12.0$؟** تعطي الأوزان القريبة من الصفر تنبؤات $\hat{Y} \approx 0$، فتخطئ في بكسلات الحواف الاثني عشر ذات القيم $\pm 1$ في `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **التقارب في 9 خطوات:** يستعيد النزول التدرجي العشوائي $[0.99, -0.99]$ تلقائيًا من البيانات.

=== DECISIONS ===
- Use contemporary standard Arabic, first-person plural for shared derivations, Arabic punctuation, Western digits and unchanged mathematical notation. Retain framework names and identifiers in Latin script.
- “locality” → «المحلية», “edge map” → «خريطة الحواف», “patch” → «رقعة», “parameter” → «معلمة», “dense layer” → «الطبقة الكثيفة».
- Use «نواة الالتفاف» for the convolutional kernel and shorten to «النواة» when the referent is clear. Render an additive scalar bias as «انحياز قياسي».

=== NOTES ===
- 7e4ef44b: the loss starts at 12, but the weights are what converge to [1, -1]. The Arabic names both subjects.
- c29c5d61: the second occurrence of 12 is written out as «الاثني عشر», agreeing with the edge pixels.
- 53a2bce7: «المعلمات الخمس» and «انحياز واحد» express the counts 5 and 1.
