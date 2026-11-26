<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## الالتفاف بوصفه ضرب مصفوفات

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
مع أن الحلقات المتداخلة في `corr2d` تجعل الرياضيات شفافة، فإن زيارة بكسل خرج واحد في كل مرة داخل Python بطيئة جدًا للأعمال الحقيقية. وثمة علاج كلاسيكي، لا يزال من الخوارزميات التي تختار منها المكتبات، هو إعادة صياغة الارتباط المتقاطع على هيئة ضرب مصفوفات واحد كبير، وهي العملية نفسها التي تنفّذها وحدات معالجة الرسومات (GPU) بأقصى سرعة. لاحظ أن كل عنصر خرج في :numref:`fig_correlation` ليس إلا الجداء النقطي بين النواة المسطّحة ذات $k_\textrm{h} k_\textrm{w}$ عنصرًا والنافذة المقابلة من المدخل بعد تسطيحها. فاستخراج جميع النوافذ البالغ عددها $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ وتكديسها صفوفًا يعطي مصفوفة رقع؛ وضربها في النواة المسطّحة يحسب الارتباط المتقاطع كله دفعة واحدة. ويسمى هذا التحويل القياسي للتخطيط *im2col* ("الصورة إلى أعمدة"، أو إلى صفوف هنا). ويجريه :numref:`fig_conv_im2col` على مثال :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![الارتباط المتقاطع في :numref:`fig_correlation` بوصفه جداء مصفوفات واحدًا. تصبح كل نافذة من النوافذ الأربع $2 \times 2$ للمدخل صفًا في مصفوفة الرقع، ويعطي الضرب في النواة المسطّحة الخرجَ المسطّح. تتبع الخلايا المظللة النافذة الأولى إلى صفها وإلى الخرج 19. البكسل المركزي 4 يقع في النوافذ الأربع كلها، ولذلك يظهر في كل صف.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
والمقايضة هي ذاكرة عمل إضافية، لأن كل بكسل داخلي يُكرَّر في ما يصل إلى $k_\textrm{h} k_\textrm{w}$ رقعة. ونتحقق في ما يلي، على :numref:`fig_correlation`، من أن ضرب المصفوفات الواحد هذا يطابق `corr2d` تمامًا:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
الفرق صفر تمامًا: جداء مصفوفتين واحد يعيد إنتاج `corr2d` بإتقان.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
وبالمثل يمكننا أن نفكّ *النواة* بدلًا من المدخل إلى مصفوفة شريطية متناثرة $\mathbf{W}$ تُضرب في الصورة المسطّحة، وهي النظير ثنائي الأبعاد تمامًا للوحة (c) في :numref:`fig_why_conv_weight_sharing`، حيث يكرر كل صف مجموعة الأوزان القليلة نفسها من النواة. إن النظر إلى الالتفاف بوصفه طبقة كثيفة مهيكلة يعيدنا إلى حيث بدأنا، إلى اشتقاق :numref:`sec_why-conv` (وتبني التمارين $\mathbf{W}$ صراحةً).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## خرائط الميزات ومجالات الاستقبال

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
ترى الطبقة الالتفافية الواحدة رقعة بحجم نواتها فقط، بينما يتطلب التعرف على أشياء كاملة دمج السياق عبر مساحات أوسع بكثير. تذكّر من :numref:`subsec_why-conv-channels` أن خرج الطبقة الالتفافية يسمى *خريطة ميزات* (feature map)، مثل خريطة الحواف العمودية `Y` لدينا. ولأي عنصر $x$ في خريطة ميزات، فإن *مجال استقباله* هو منطقة صورة المدخل (أو طبقة سابقة) التي يمكن لقيمها أن تؤثر في $x$ أثناء الانتشار الأمامي، وهو مصطلح مستعار من علم الفيزيولوجيا العصبية في :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### الطبقات المكدّسة

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
في :numref:`fig_correlation` يكون لمجال استقبال الخرج المظلل `19` حجم $2 \times 2$ يغطي البكسلات الأربعة العليا اليسرى من $\mathbf{X}$. لنكدّس الآن طبقة التفافية *ثانية* بحجم $2 \times 2$ فوق الخرج $\mathbf{Y}$ بحجم $2 \times 2$ لننتج عددًا قياسيًا (scalar) واحدًا $z$: يعتمد $z$ على عناصر $\mathbf{Y}$ الأربعة كلها، وهي تعتمد معًا على عناصر $\mathbf{X}$ التسعة كلها. وهكذا يتسع مجال الاستقبال بسلاسة مع العمق بتكديس الطبقات.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
فما سرعة نموه؟ كل طبقة التفافية $k \times k$ تصل إلى مسافة $k - 1$ بكسلًا أبعد من عنصر واحد من مدخلها، ولذلك فإن تكديس $L$ طبقة ذات أحجام نوى $k_1, \ldots, k_L$ يعطي مجال استقبال طول ضلعه

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
وفي مكدّس من $L$ طبقة متتالية $3 \times 3$ تضيف كل طبقة بكسلين، فتغطي $(2L + 1) \times (2L + 1)$ مدخلًا: $5 \times 5$ بعد طبقتين و$7 \times 7$ بعد ثلاث (:numref:`fig_conv_receptive_field`). وهذا النمو الخطي نعمة في الواقع وليس سببًا للجوء إلى نوى ضخمة: فثلاث طبقات $3 \times 3$ مكدّسة تحقق مجال الاستقبال $7 \times 7$ نفسه الذي تحققه طبقة واحدة $7 \times 7$ بمعاملات أقل بكثير ($3 \times 9 = 27$ وزنًا مقابل 49 لكل زوج من القنوات)، مع تداخل ثلاث دوال لاخطية بدلًا من واحدة.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![مجال استقبال وحدة واحدة بعد ثلاث طبقات أحادية البعد بحجم نواة 3. الخلايا المظللة هي الوحدات التي تعتمد عليها الوحدة العليا، وتصل الخطوط كلًا منها بمدخلاتها. تُوسّع كل طبقة مجال الاستقبال بمقدار 2، حتى 7 مدخلات.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### مجالات الاستقبال الفعّالة

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
وليكن في بالك أن $r$ يقيس مجال الاستقبال *النظري* (كل بكسل مدخل له *أي* مسار إلى وحدة الخرج) دون أن يقول بأي قوة يسهم كل بكسل فعليًا. وبقياس تدرجات المدخل مباشرةً، بيّن :citet:`Luo.Li.Urtasun.ea.2016` أن مجال الاستقبال *الفعّال* يتضاءل بحدة بعيدًا عن المركز بما يشبه منحنى جرس غاوسي، في الشبكات غير المدرّبة والمدرّبة على السواء. يمكن للوحدات العميقة نظريًا أن ترى بعيدًا عبر الصورة، لكن حساسيتها تتركز بقوة قرب المركز.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
ويفسّر عدّ المسارات شكل الجرس. فالمدخل عند طرف مجال الاستقبال يصل إلى وحدة الخرج عبر سلسلة واحدة من الوصلات، بينما يستطيع المدخل المركزي أن يسلك طرقًا كثيرة عبر الطبقات الوسيطة. ويعدّها :numref:`fig_conv_rf_paths` لبرج :numref:`fig_conv_receptive_field`: للمدخل المركزي سبعة مسارات إلى الوحدة العليا، ولكل مدخل طرفي مسار واحد. وإذا كانت الأوزان كلها متقاربة في الحجم، فإن تأثير المدخل يزداد بعدد مساراته.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![عدد المسارات من كل وحدة إلى الوحدة العليا في :numref:`fig_conv_receptive_field` (ثلاث طبقات أحادية البعد بحجم نواة 3). كل عدد هو مجموع أعداد الوحدات التي تغذيها الخلية. الخلايا الأغمق لها مسارات أكثر من بقية خلايا طبقتها، وتكرر الأعمدة أعداد صف المدخل: 1، 3، 6، 7، 6، 3، 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
تبقى في طبقتنا الأساسية ثغرتان عمليتان: كل التفاف غير محشو يُنقص الأبعاد المكانية ويقلل أخذ العينات من بكسلات الحواف، والنواة الواحدة لا تكشف إلا نمطًا واحدًا. نعالج الحجم المكاني والحدود في :numref:`sec_padding` لاحقًا، وخرائط الميزات متعددة القنوات في :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## الخلاصة

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
تطبّق الطبقة الالتفافية الارتباط المتقاطع ثنائي الأبعاد بنواة متعلَّمة مع انحياز عددي، فتمرر النواة عبر كل نافذة صالحة من المدخل. وتكشف نواة الفرق المحدود البسيطة `[1, -1]` بحجم $1 \times 2$ الحواف العمودية في أي موضع من الصورة، ويسترد النزول التدرجي هذه الأوزان نفسها من بيانات المدخل والخرج وحدها، تمامًا كما تتعلم الشبكات الالتفافية العميقة هرميات كاملة من المرشحات من البداية إلى النهاية. أما الالتفاف الرياضي فيقلب النواة، ولا فرق في ذلك بعد أن تصبح النوى متعلَّمة. ويمكن تقييم الالتفاف بضرب مصفوفات واحد (*im2col*)، ويوسّع تكديس $L$ طبقة التفافية مجال استقبال كل وحدة خطيًا بمقدار $k_i - 1$ لكل طبقة.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="تنفيذ سريع: الالتفاف عبر ضرب المصفوفات"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[تحويل im2col (الصورة إلى أعمدة)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **GEMM كثيف واحد:** سطّح كل نافذة $k_\textrm{h} \times k_\textrm{w}$ إلى صف في `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ واضربها في $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **المقايضة:** أقصى سرعة لنوى Tensor Core في GPU مقابل تكرار الرقع في ذاكرة عمل أكبر بمقدار $k_\textrm{h} k_\textrm{w}\times$.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="رؤية مقابلة: فك النواة إلى مصفوفة شريطية"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[ضرب مصفوفة أوزان توبليتز]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
بدلًا من فرد صورة المدخل إلى رقع مكررة، يمكننا أن نفكّ *النواة* إلى مصفوفة شريطية متناثرة $\mathbf{W}$ تضرب متجه المدخل المسطّح $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **صيغة مصفوفة–متجه:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$، حيث لـ $\mathbf{W}$ عدد من الصفوف قدره $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ وعدد من الأعمدة قدره $n_\textrm{h} n_\textrm{w}$.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **الصلة بـ §{sec_why-conv} وبالانتشار العكسي:** يحمل كل صف من $\mathbf{W}$ عناصر النواة نفسها ($k_\textrm{h} k_\textrm{w}$ عنصرًا) مزاحةً عبر الأعمدة (اللوحة c)، فيصبح التمرير الخلفي بالنسبة إلى $\mathbf{X}$ مجرد ضرب في $\mathbf{W}^\top$ (الالتفاف المنقول).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="خرائط الميزات ومجالات الاستقبال عبر العمق"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[نمو خطي لمجال الاستقبال في الطبقات المكدّسة]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**مجال استقبال** وحدة ما هو منطقة صورة المدخل الأصلية التي يمكن لبكسلاتها أن تؤثر في تنشيط تلك الوحدة أثناء الانتشار الأمامي:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **طبقة واحدة ($L=1$):** ترى نواة $k_1 \times k_1$ رقعة مدخل $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **تكديس $L$ طبقة بخطوة 1:** تصل الطبقة $i$ ذات حجم النواة $k_i$ إلى مسافة $k_i - 1$ بكسلًا أبعد على كل محور من عنصر واحد من الطبقة $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ طبقة $3 \times 3$ مكدّسة:** تضيف كل طبقة $3 - 1 = 2$ بكسل، فيكون طول الضلع $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="لماذا تتفوق مكدّسات النوى 3×3 العميقة على نواة كبيرة واحدة"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[كفاءة المعاملات والتعبيرية اللاخطية]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **مجال استقبال $7 \times 7$ مطابق:** ثلاث طبقات $3 \times 3$ مكدّسة ($r = 1 + 2 + 2 + 2 = 7$) ترى الرقعة $7 \times 7$ نفسها التي تراها طبقة واحدة $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **معاملات أقل بنسبة 45%:** $3 \times 9 c^2 = 27 c^2$ وزنًا بدلًا من $49 c^2$ عبر $c$ قناة.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **تعبيرية أغنى:** تركّب **3 تنشيطات لاخطية** بدلًا من واحد (قاعدة التصميم الأساسية في VGG، انظر §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="مجالات الاستقبال النظرية مقابل الفعّالة"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[تضاؤل غاوسي للحساسية بعيدًا عن المركز]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **مجال الاستقبال النظري ($r = 2L+1$):** المربع المحيط بجميع البكسلات التي لها مسار *واحد على الأقل* إلى وحدة الخرج.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **مبرهنة النهاية المركزية:** تتصل البكسلات المركزية عبر عدد هائل (أسّي) من المسارات، بينما تتصل بكسلات الزوايا عبر **مسار واحد** فقط.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **مجال الاستقبال الفعّال (Luo et al., 2016):** يتضاءل مقدار تدرج المدخل $\partial y / \partial \mathbf{X}$ بمنحنى جرس غاوسي بانحراف معياري $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="اختبر فهمك: الارتباط المتقاطع ومجالات الاستقبال"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[اختبر فهمك]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
اختبر فهمك لأشكال الخرج و`im2col` وحساب مجالات الاستقبال:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **أبعاد مصفوفة im2col:** تطبّق نواة $3 \times 3$ أحادية القناة وغير محشوة على صورة $10 \times 10$ عبر `im2col`. ما شكل الخرج المكاني، وما الأبعاد الدقيقة لمصفوفة `patches` ولمتجه النواة المسطّح؟

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **العمق اللازم لتغطية Fashion-MNIST:** بخطوة 1، كم طبقة $3 \times 3$ مكدّسة يلزم لكي يغطي مجال الاستقبال النظري لوحدة واحدة صورة $28 \times 28$ كاملة؟ وكم طبقة بحجم $5 \times 5$؟

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **إزاحة حدود خسارة الحواف:** في تجربة تعلّم الحواف العمودية $6 \times 8$ لدينا، لماذا بدأت الخسارة عند $\approx 12$، وكم كانت ستكون الخسارة الابتدائية لو امتد الشريط الأسود إلى الحافة اليمنى (`X[:, 2:8] = 0`)؟

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**الإجابات:**
(1) شكل الخرج $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ خرجًا). مصفوفة `patches` شكلها $(64, 9)$ وتُضرب في متجه النواة $(9, 1)$.
(2) لـ $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ طبقة. ولـ $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ طبقات.
(3) كان في كل صف من الصفوف الستة حافتان بقيمة $\pm 1$ ($6 \times 2 \times 1^2 = 12$). وامتداد الشريط الأسود إلى الحد الأيمن يزيل حافة الانتقال من الأسود إلى الأبيض على اليمين، فتبقى حافة واحدة في كل صف وتنخفض الخسارة الابتدائية إلى النصف، أي $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="الملخص والنقاط الرئيسة"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[خلاصة القسم]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **الارتباط المتقاطع ثنائي الأبعاد وتعلّم الحواف:** تنزلق $\mathbf{K}$ عبر النوافذ الصالحة ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$)؛ ويتعلم SGD القيمة $[1, -1]$ في 9 خطوات.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **ضرب مصفوفات im2col:** يقايض تكرار الرقع بمقدار $k_\textrm{h} k_\textrm{w}\times$ بأقصى إنتاجية GEMM على GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **مجالات الاستقبال المكدّسة:** $r = 1 + \sum (k_i - 1) = 2L+1$ لمكدّسات $3 \times 3$، مع حساسية غاوسية مركزية.

=== DECISIONS ===
- "stack (layers)" → تكديس / مكدّس; "tower" → برج; "path" → مسار; "patch matrix" → مصفوفة الرقع; "banded matrix" → مصفوفة شريطية; "sparse" → متناثرة; "flatten" → تسطيح.
- "theoretical / effective receptive field" → مجال الاستقبال النظري / الفعّال; "feature map" → خريطة ميزات (feature map) at first use.
- "dot product" → الجداء النقطي (glossary only has "inner product" → جداء داخلي); "workspace" → ذاكرة عمل مؤقتة (ذاكرة عمل in slides).
- "nonlinearity" → دالة لاخطية; "activation" → تنشيط; "standard deviation" → الانحراف المعياري; "Central Limit Theorem" → مبرهنة النهاية المركزية.
- "transposed convolution" → الالتفاف المنقول; "backward pass" → التمرير الخلفي; "forward propagation" → الانتشار الأمامي.
- Names in citations inside prose stay Latin ("Luo et al., 2016").
- Slide titles use title-style short noun phrases without final punctuation.

=== NOTES ===
- 755ac11a: English says "45% fewer parameters"; 27/49 gives about 45% fewer, kept as is.
- 293436eb (English): "Central Limit Theorem ... exponentially many paths" is loose; translated as written.
