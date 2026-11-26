<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# الالتفاف في الصور

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
في :numref:`sec_why-conv` اشتققنا صيغة الطبقة الالتفافية على الورق انطلاقًا من مبدأين هندسيين، هما تساوق الإزاحة والمحلية. والآن نرى ما تحسبه هذه الصيغة فعليًا عند تطبيقها على صورة، وهل يستطيع النزول التدرجي اكتشاف نوى التفاف مفيدة تلقائيًا من البيانات.

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
لنفكّر في كيفية تحديد الحواف الرأسية عند التقاء منطقة فاتحة بأخرى داكنة. إذا طرحنا من قيمة كل بكسل قيمة جاره على اليمين ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)، حصلنا على صفر داخل المناطق المتجانسة، وعلى $+1$ عند الانتقال من الأبيض إلى الأسود، وعلى $-1$ عند الانتقال من الأسود إلى الأبيض. وبتمرير الكاشف ذي الوزنين $(1, -1)$ على كل موضع صالح في الصورة نحصل على *خريطة حواف*. في هذا القسم ننفّذ الارتباط المتقاطع ثنائي الأبعاد، ونختبر كاشف الحواف الذي صمّمناه يدويًا، وندع النزول التدرجي *يتعلّم* $(1, -1)$ مباشرةً من البيانات. ثم نبيّن كيف يمكن حساب الالتفاف بعملية واحدة من ضرب المصفوفات (*im2col*)، ونشتقّ مقدار اتساع مجال استقبال الوحدة عند تكديس الطبقات.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## الارتباط المتقاطع

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### حساب الارتباط المتقاطع

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
لنضع القنوات جانبًا في الوقت الحالي. لتكن المدخلة $\mathbf{X}$ مصفوفة أبعادها $n_\textrm{h} \times n_\textrm{w}$، ولتكن نواة الالتفاف $\mathbf{K}$ مصفوفة أبعادها $k_\textrm{h} \times k_\textrm{w}$. تحدّد النواة *نافذة* منزلقة شكلها $k_\textrm{h} \times k_\textrm{w}$. نضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$، ونضرب عناصر الرقعة التي تغطيها النافذة في العناصر المناظرة لها من $\mathbf{K}$، ثم نجمع نواتج الضرب لنحصل على عنصر المخرج عند $(i, j)$.

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
هذا هو *الارتباط المتقاطع ثنائي الأبعاد* بين $\mathbf{X}$ و$\mathbf{K}$، ونكتبه $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ بفهرسة تبدأ من الصفر كما في الشيفرة. في :numref:`sec_why-conv` كانت الإزاحات موزّعة تناظريًا من $-\Delta$ إلى $\Delta$. أما اتخاذ الزاوية العلوية اليسرى نقطة مرجعية فلا يغيّر سوى إزاحة ثابتة قدرها $\Delta$، ويسهّل التعامل مع النوى ذات الأبعاد الزوجية التي لا بكسل في مركزها، مثل نواة أبعادها $2 \times 2$ أو كاشف الحواف لدينا بأبعاد $1 \times 2$.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![الارتباط المتقاطع ثنائي الأبعاد. يبيّن التظليل أول عنصر من المخرج وعناصر المدخلة والنواة المستخدمة في حسابه، وهو $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
في :numref:`fig_correlation` نُزلق نواة أبعادها $2 \times 2$ على مدخلة أبعادها $3 \times 3$، بكسلًا واحدًا في كل مرة، من اليسار إلى اليمين ومن الأعلى إلى الأسفل. تقع النافذة بكاملها داخل المدخلة في أربعة مواضع، فنحصل على المخرج التالي بأبعاد $2 \times 2$.

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
لاحظ أن حجم المخرج يتقلّص قليلًا على امتداد كل محور، إذ يجب أن تبقى النواة بكاملها داخل المدخلة. فإذا كان عرض النافذة $k$ وطول المحور $n$، أمكن وضعها في $n - k + 1$ موضعًا. وعليه، يكون شكل المخرج كما يلي.

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
وهكذا، تعطي مدخلة أبعادها $3 \times 3$ مع نواة أبعادها $2 \times 2$ مخرجًا أبعاده $2 \times 2$، وتعطي صورة أبعادها $6 \times 8$ مع نواة أبعادها $1 \times 2$ مخرجًا أبعاده $6 \times 7$، وتعطي صورة من Fashion-MNIST أبعادها $28 \times 28$ مع نواة أبعادها $5 \times 5$ مخرجًا أبعاده $24 \times 24$. وحين نريد الحفاظ على الأبعاد المكانية، نحيط المدخلة بإطار من الأصفار. وهذا هو *الحشو* الذي نتناوله مع خطوات الانزلاق والتوسيع في :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### التنفيذ

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
ننفّذ :eqref:`eq_corr2d` مباشرةً في `corr2d` باستخدام حلقتين متداخلتين تمرّان على جميع إحداثيات المخرج.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
للتحقّق من أن `corr2d` توافق حسابنا اليدوي في :numref:`fig_correlation`، ننشئ موتر المدخلة بأبعاد $3 \times 3$، `X`، ونواة الالتفاف بأبعاد $2 \times 2$، `K`، ثم نحسب `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
يطابق المخرج ما في :numref:`fig_correlation` عنصرًا بعنصر. وكما بيّنّا في :numref:`sec_why-conv`، يتيح الالتفاف بمعناه الرياضي الدقيق، باستخدام النواة المقلوبة $\mathbf{K}'$، تعلّم مجموعة الطبقات نفسها تمامًا. لذا نتّبع الاصطلاح الشائع في التعلّم العميق ونسمّي الارتباط المتقاطع «التفافًا».

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
تحسب الطبقة الالتفافية ثنائية الأبعاد الارتباط المتقاطع بين مدخلتها ونواة متعلَّمة، ثم تضيف انحيازًا هو عدد قياسي. وكما فعلنا مع أوزان البيرسبترون متعدد الطبقات في :numref:`chap_perceptrons`، نسحب القيم الابتدائية للنواة من التوزيع $\mathcal{N}(0, 0.01^2)$، ونهيّئ الانحياز بالصفر. في المثال التالي، تسجّل `Conv2D` المعلمتين وتستدعي `corr2d` في التمرير الأمامي. وتُسمّى الطبقة التي ارتفاع نواتها $h$ وعرضها $w$ طبقةً التفافية بأبعاد $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
بخلاف الطبقة الكثيفة، تقبل الطبقة الالتفافية مدخلات متفاوتة الدقة، لأن معلماتها تعتمد على حجم النواة وحده. فالطبقة نفسها، بأبعاد $2 \times 2$ وبإجمالي 5 معلمات، تعالج الموتر ذا الأبعاد $3 \times 3$، `X`، وصورة أبعادها $6 \times 8$، وتنتج مخرجين شكلاهما $2 \times 2$ و$5 \times 7$ على الترتيب.

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### كشف الحواف

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
نختبر الآن كاشف الحواف الرأسية الذي عرضناه في المقدّمة على صورة اصطناعية أبعادها $6 \times 8$، أعمدتها الأربعة الوسطى سوداء (`0`) وأعمدتها الخارجية بيضاء (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
نعرّف نواة الالتفاف ذات الأبعاد $1 \times 2$ كما يلي.

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
تحسب هذه النواة عند كل موضع $(i, j)$ *الفرق المحدود* في الاتجاه الأفقي، $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. وهذا تقريب متقطّع لمشتق السطوع في الاتجاه الأفقي، مع إغفال اختلاف الإشارة. وينعدم الفرق في المناطق المتجانسة، ويكبر مقداره حيثما تتغيّر شدّة الإضاءة فجأة.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
نحسب الارتباط المتقاطع للصورة المخطّطة ذات الأبعاد $6 \times 8$، `X`، مع نواة الفرق الأفقي ذات الأبعاد $1 \times 2$، `K`، باستدعاء `corr2d(X, K)`. فنحصل على خريطة حواف أبعادها $6 \times 7$ هي `Y`.

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
وكما توقّعنا، تسجّل `Y` القيمة `+1` على امتداد الحدّ الفاصل من الأبيض إلى الأسود، والقيمة `-1` على امتداد الحدّ الفاصل من الأسود إلى الأبيض، والقيمة `0` في سائر المواضع. وإذا طبّقنا النواة نفسها على الصورة بعد تبديل صفوفها بأعمدتها، فصار الشريط الرأسي أفقيًا، حصلنا على مخرج عناصره كلها أصفار.

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
يتّضح أن `[1, -1]` تستجيب للحواف الرأسية وحدها. أما كشف الحواف الأفقية أو المائلة، أو كشف الزوايا أو الأنسجة، فيتطلّب نوى مختلفة تمامًا.

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
يسهل تصميم `[1, -1]` يدويًا عندما نعرف بالضبط الميزة التي نبحث عنها، لكن لا يستطيع أي مهندس أن يحدّد يدويًا آلاف المرشحات الوسيطة في شبكة عميقة. فهل يستطيع النزول التدرجي استعادة النواة `[1, -1]` من زوج المدخلة والمخرج `(X, Y)` وحده؟ لاختبار ذلك نستخدم الطبقة الالتفافية ثنائية الأبعاد المدمجة في كل إطار عمل، من دون انحياز، وبأوزان ابتدائية مسحوبة من $\mathcal{N}(0, 0.01^2)$. ونُصغّر مجموع الأخطاء التربيعية على جميع عناصر المخرج الـ42 في 10 خطوات من النزول التدرجي، عند ضبط معدّل التعلّم على $0.03$. ولأن الطبقات المدمجة تتطلّب موترات رباعية الأبعاد فيها محور للدفعة وآخر للقنوات (:numref:`sec_channels`)، نعيد تشكيل `X` و`Y` بما يلائمها.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
تبدأ الخسارة قرب 12، لأن النواة العشوائية ذات القيم القريبة من الصفر تخطئ بنحو 1 في كل بكسل من بكسلات الحواف الـ12 ذات القيم $\pm 1$. وفي غضون تسع خطوات فقط تهبط الخسارة إلى أقل من 0.01. ويؤكّد فحص النواة المتعلَّمة ما توصّل إليه النزول التدرجي.

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
يختلف كل عنصر عن نظيره في `[1, -1]` بنحو 0.02 على الأكثر. لم تتضمّن الحلقة أي معرفة مسبقة بالحواف، فقد اكتشف النزول التدرجي المرشح بنفسه انطلاقًا من إشارة الخسارة. هنا أشرفنا على تدريب طبقة واحدة مباشرةً باستخدام `Y`. أما في الشبكة العصبية الالتفافية العميقة، فيتعلّم الانتشار العكسي نوى جميع الطبقات من طرف إلى طرف بالاعتماد على خسارة المهمة النهائية وحدها.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
الالتفاف في الصور<br>**الارتباط المتقاطع ثنائي الأبعاد · كشف الحواف · النوى المتعلَّمة · im2col · مجالات الاستقبال**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="كشف الحواف المحلية عبر الصورة"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[الدافع والفكرة الأساسية]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **صورة اختبار اصطناعية بأبعاد $6 \times 8$**. تحيط الأعمدة الخارجية البيضاء (`1`) بشريط رأسي أسود عرضه 4 أعمدة (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **الفرق المحدود الأفقي**. يحسب تمرير $\mathbf{K} = [1, -1]$ الفرق $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$، فيعطي `+1` عند الانتقال من الأبيض إلى الأسود، و`-1` في الاتجاه المعاكس.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **التعلّم والتصميم اليدوي**. تكتشف الشبكات العصبية الالتفافية العميقة آلاف النوى المماثلة تلقائيًا بالنزول التدرجي.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[الجداءات الداخلية في نافذة منزلقة]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
إذا كانت لدينا مصفوفة مدخلة أبعادها $n_\textrm{h} \times n_\textrm{w}$ هي $\mathbf{X}$، ونواة أبعادها $k_\textrm{h} \times k_\textrm{w}$ هي $\mathbf{K}$، فإن الارتباط المتقاطع ثنائي الأبعاد $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ يضع الزاوية العلوية اليسرى للنافذة عند $(i, j)$ ويجمع نواتج ضرب العناصر المتناظرة.

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **الزاوية العلوية اليسرى نقطة مرجعية**. تبدأ الفهارس $a, b \ge 0$ من الصفر كما في فهرسة المصفوفات في الشيفرة، وتلائم النوى ذات الأبعاد الزوجية، مثل $2 \times 2$ أو $1 \times 2$.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **العنصر المظلّل أعلى يسار المخرج $(0,0)$**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="مواضع النافذة الصالحة والأبعاد المكانية للمخرج"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[لماذا تتقلّص الشبكة عند الالتفاف بلا حشو؟]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **مواضع النافذة الصالحة**. تقع نافذة أبعادها $k_\textrm{h} \times k_\textrm{w}$ بكاملها داخل شبكة أبعادها $n_\textrm{h} \times n_\textrm{w}$ في $n - k + 1$ موضعًا لكل محور.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **أمثلة على الأشكال**. $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)، و$28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="تنفيذ الارتباط المتقاطع ثنائي الأبعاد"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[تنفيذ مباشر بحلقتين متداخلتين]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
تخصّص الدالة `corr2d` موتر المخرج بأبعاد $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$، وتحسب كل عنصر فيه بجمع عناصر `X[i:i+h, j:j+w] * K`.

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
في كل تكرار نقتطع نافذة فرعية بأبعاد $k_\textrm{h} \times k_\textrm{w}$ من `X`، ونضرب عناصرها في نظائرها من `K`، ثم نجمع النواتج بواسطة `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="التحقّق من corr2d والتكافؤ عند قلب النواة"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[الارتباط المتقاطع والالتفاف الرياضي]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **النواة $\mathbf{K}'$ بعد تدويرها 180°**. الارتباط المتقاطع مع $\mathbf{K}$ يكافئ الالتفاف مع $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **مجموعة الطبقات المتعلَّمة هي نفسها**. تحسب أطر العمل الارتباط المتقاطع وتسمّيه التفافًا.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="بناء وحدة لطبقة التفافية ثنائية الأبعاد"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[أوزان نواة قابلة للتعلّم وانحياز قياسي]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
تستخدم الطبقة الالتفافية `corr2d` مع معلمة أوزان قابلة للتعلّم شكلها $k_\textrm{h} \times k_\textrm{w}$ (تُهيّأ من $\mathcal{N}(0, 0.01^2)$) وانحياز قياسي (يُهيّأ بالقيمة $0$).

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="استقلال الطبقات الالتفافية عن دقّة المدخلة"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[المعلمات نفسها لأي حجم للمدخلة]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **استقلال عن $H \times W$**. تعالج 5 معلمات ثابتة (نواة بأبعاد $2 \times 2$ + 1 للانحياز) مدخلات بأبعاد $3 \times 3$ أو $6 \times 8$ أو بحجم ميغابكسل، من دون إعادة تشكيل الأوزان.

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
ننشئ صورة الاختبار ذات الأبعاد $6 \times 8$، `X`، فنملأ طرفيها الأيسر والأيمن بالآحاد، وأعمدتها الأربعة الوسطى بالأصفار (`X[:, 2:6] = 0`).

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **أعمدة ثابتة الشدّة على أجزاء**. تبقى الشدّة ثابتة (`1` أو `0`) داخل كل مجموعة من الأعمدة $0\text{--}1$، و$2\text{--}5$، و$6\text{--}7$، ولا تتغيّر إلا عند الانتقال $1 \to 2$ و$5 \to 6$.

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
عند حساب الارتباط المتقاطع بين `X` ونواة أبعادها $1 \times 2$ هي $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$، نحصل على $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ لكل زوج من البكسلات المتجاورة أفقيًا.

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **خريطة حواف بإشارة**. يعطي العمود 1 القيمة `+1` (من الأبيض إلى الأسود)، والعمود 5 القيمة `-1` (من الأسود إلى الأبيض)، وتعطي الأعمدة الداخلية ثابتة الشدّة القيمة `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="انتقائية نوى الحواف بحسب الاتجاه"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[لماذا لا تكشف نواة واحدة الحواف في كل الاتجاهات؟]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **انعدام الاستجابة للحواف الأفقية**. بعد تبديل صفوف `X` بأعمدتها (`X.T`)، تتساوى قيم البكسلات المتجاورة أفقيًا، فتعطي $[1, -1]$ أصفارًا فقط.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **مجموعات المرشحات (§{sec_channels})**. تتعلّم طبقات الرؤية قنوات متوازية للاتجاهات الأفقية والرأسية والمائلة.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="تعلّم كاشف الحواف بالنزول التدرجي"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[تدريب طبقة التفافية مدمجة بإشراف الزوج (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **موترات رباعية الأبعاد**. نعيد تشكيل `X` و`Y` في أربعة أبعاد (مثال واحد وقناة واحدة) ونُصغّر $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ بمعدّل تعلّم $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="فحص أوزان النواة المتعلَّمة"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[لماذا تبدأ الخسارة عند 12 وتتقارب الأوزان نحو [1, -1]؟]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **سبب بدء الخسارة عند $12.0$**. تعطي الأوزان القريبة من الصفر التنبؤ $\hat{Y} \approx 0$، فتخطئ في بكسلات الحواف الاثني عشر ذات القيم $\pm 1$ في `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **التقارب في 9 خطوات**. يستعيد النزول التدرجي العشوائي (SGD) الأوزان $[0.99, -0.99]$ تلقائيًا من البيانات.

=== DECISIONS ===
- Use Modern Standard Arabic, first-person plural in explanations and singular imperatives in exercises. Retain Western digits, use Arabic prose punctuation, and keep the source's names, code and identifiers. Use Arabic terminology directly, retaining English acronyms and algorithm names such as SGD and im2col.
- locality → «المحلية»; edge detection / edge detector / edge map → «كشف الحواف» / «كاشف الحواف» / «خريطة الحواف»; vertical edge → «حافة رأسية»; horizontal edge → «حافة أفقية».
- input / output → «مدخلة» / «مخرج» for a structured object; patch → «رقعة»; sliding window → «نافذة منزلقة»; parameter → «معلمة»; dense layer → «طبقة كثيفة»; initialize → «هيّأ».
- Use the singular «فرق محدود» for an individual finite difference, «الفروق المحدودة» for the method. Introduce «نواة الالتفاف» and use «النواة» where the referent is clear.
- transpose (image) → «تبديل الصفوف بالأعمدة»; filter bank → «مجموعة المرشحات»; end to end → «من طرف إلى طرف».

=== NOTES ===
- 7e4ef44b: The heading says the loss “converges to [1, -1]”. It is the weights that converge toward those values.
- 456670c5: The slide uses SGD, though the shown loop computes full-batch gradient descent on the single input–output pair. The source terminology is retained here.
