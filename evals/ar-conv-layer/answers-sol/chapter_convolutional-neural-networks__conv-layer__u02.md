<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ar rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## حساب الالتفاف بضرب المصفوفات

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
توضّح الحلقتان المتداخلتان في `corr2d` كيف يجري الحساب، لكن المرور على بكسلات المخرج واحدًا تلو الآخر بلغة Python أبطأ بكثير مما تتطلّبه التطبيقات العملية. ومن الحلول المعروفة، التي ما زالت المكتبات تختار منها، صياغة الارتباط المتقاطع على هيئة عملية واحدة كبيرة من ضرب المصفوفات، وهي العملية التي تنفّذها معالجات GPU بأقصى سرعة. لاحظ أن كل عنصر من المخرج في :numref:`fig_correlation` هو ببساطة الجداء النقطي بين النواة المؤلّفة من $k_\textrm{h} k_\textrm{w}$ عنصرًا ونافذة المدخلة المقابلة لها، بعد تسطيح كلٍّ منهما. نستخرج النوافذ كلها، وعددها $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$، ونرتّبها في صفوف لنحصل على مصفوفة الرقع. وبضربها في النواة المسطّحة نحسب الارتباط المتقاطع بأكمله دفعة واحدة. ويُعرف هذا التحويل المعياري في ترتيب البيانات باسم *im2col* («من الصورة إلى الأعمدة»، أو إلى الصفوف هنا). ويبيّن :numref:`fig_conv_im2col` تطبيقه على مثال :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![حساب الارتباط المتقاطع في :numref:`fig_correlation` بعملية ضرب مصفوفات واحدة. تتحوّل كل واحدة من نوافذ المدخلة الأربع ذات الأبعاد $2 \times 2$ إلى صفّ في مصفوفة الرقع. وبضربها في النواة المسطّحة نحصل على المخرج المسطّح. ويتتبّع التظليل النافذة الأولى إلى الصفّ المقابل لها ثم إلى الالقيمة 19 في المخرج. ويقع البكسل الأوسط، وقيمته 4، في النوافذ الأربع كلها، لذا يظهر في كل صفّ.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
تأتي هذه السرعة بكلفة إضافية في الذاكرة المؤقتة اللازمة للحساب، إذ يُنسخ كل بكسل داخلي في ما يصل إلى $k_\textrm{h} k_\textrm{w}$ رقعة. نتحقّق أدناه، باستخدام مثال :numref:`fig_correlation`، من أن عملية ضرب المصفوفات الواحدة هذه تعطي نتيجة `corr2d` نفسها تمامًا.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
الفرق يساوي صفرًا تمامًا. فقد أعطت عملية ضرب مصفوفات واحدة نتيجة `corr2d` نفسها.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
ويمكننا، على نحو مكافئ، أن نفرد *النواة* بدلًا من المدخلة في مصفوفة شريطية متناثرة $\mathbf{W}$ نضربها في الصورة المسطّحة. وهذا هو النظير ثنائي الأبعاد تمامًا للوحة (c) في :numref:`fig_why_conv_weight_sharing`، حيث تتكرّر المجموعة الصغيرة نفسها من أوزان النواة في كل صف. والنظر إلى الالتفاف بوصفه طبقة كثيفة ذات بنية محدّدة يعيدنا إلى الاشتقاق في :numref:`sec_why-conv` (وفي التمارين نبني $\mathbf{W}$ مباشرةً).

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
لا ترى الطبقة الالتفافية الواحدة سوى رقعة بحجم نواتها، لكن التعرّف على أجسام كاملة يتطلّب جمع معلومات السياق من مناطق أوسع بكثير. تذكّر من :numref:`subsec_why-conv-channels` أن مخرج الطبقة الالتفافية يُسمّى *خريطة الميزات*، مثل خريطة الحواف الرأسية `Y` في مثالنا. ولكل عنصر $x$ في خريطة الميزات *مجال استقبال*، وهو المنطقة من صورة المدخلة (أو من طبقة سابقة) التي يمكن لقيمها أن تؤثّر في $x$ أثناء التمرير الأمامي. وقد استعرنا هذا المصطلح من الفيزيولوجيا العصبية كما في :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### تكديس الطبقات

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
في :numref:`fig_correlation`، للعنصر المظلّل `19` في المخرج مجال استقبال أبعاده $2 \times 2$ يغطي البكسلات الأربعة في الزاوية العلوية اليسرى من $\mathbf{X}$. لنضف الآن طبقة التفافية *ثانية* أبعادها $2 \times 2$ فوق المخرج ذي الأبعاد $2 \times 2$، أي $\mathbf{Y}$، فنحصل على عدد قياسي واحد هو $z$. وتعتمد قيمة $z$ على عناصر $\mathbf{Y}$ الأربعة كلها، وهي تعتمد مجتمعةً على عناصر $\mathbf{X}$ التسعة كلها. وهكذا يتّسع مجال الاستقبال تدريجيًا بزيادة العمق.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
ما سرعة هذا الاتّساع؟ في كل طبقة التفافية أبعادها $k \times k$، يزيد امتداد مجال الاستقبال بمقدار $k - 1$ بكسلًا على امتداده لعنصر واحد من مدخلتها. وعند تكديس $L$ طبقةً أحجام نواها $k_1, \ldots, k_L$ يكون طول ضلع مجال الاستقبال كما يلي.

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
عند تكديس $L$ طبقةً متتالية بأبعاد $3 \times 3$، يزداد طول الضلع بمقدار 2 بكسل مع كل طبقة، فيغطي المجال $(2L + 1) \times (2L + 1)$ عنصرًا من المدخلة. وتصبح أبعاده $5 \times 5$ بعد طبقتين و$7 \times 7$ بعد ثلاث طبقات (:numref:`fig_conv_receptive_field`). وفي هذا الاتّساع الخطي ميزة تغنينا عن اللجوء إلى نوى ضخمة، إذ تعطي ثلاث طبقات متتالية بأبعاد $3 \times 3$ مجال الاستقبال نفسه، بأبعاد $7 \times 7$، الذي تعطيه طبقة واحدة أبعادها $7 \times 7$، ولكن بمعلمات أقلّ بكثير ($3 \times 9 = 27$ وزنًا مقابل 49 لكل زوج من القنوات)، مع إدراج ثلاثة تحويلات لاخطية بدلًا من واحدة.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![مجال استقبال وحدة واحدة بعد ثلاث طبقات أحادية البعد حجم نواتها 3. تمثّل الخلايا المظلّلة الوحدات التي تعتمد عليها الوحدة العلوية، وتصل الخطوط كلًّا منها بمدخلاتها. وتوسّع كل طبقة مجال الاستقبال بمقدار 2، حتى يصل إلى 7 عناصر من المدخلة.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### مجالات الاستقبال الفعلية

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
تذكّر أن $r$ يقيس مجال الاستقبال *النظري*، الذي يضم كل بكسل من المدخلة له *أيّ* مسار إلى وحدة المخرج، من دون أن يبيّن مقدار إسهام كل بكسل فعليًا. وبقياس التدرجات بالنسبة إلى المدخلة مباشرةً، بيّن :citet:`Luo.Li.Urtasun.ea.2016` أن مجال الاستقبال *الفعلي* يتناقص بشدّة كلما ابتعدنا عن المركز، وفق منحنى جرسي غاوسي تقريبًا، في الشبكات المدرَّبة وغير المدرَّبة على السواء. تستطيع الوحدات في الطبقات العميقة نظريًا أن ترى مناطق بعيدة من الصورة، لكن حساسيتها تتركّز بشدّة حول المركز.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
يفسّر عدّ المسارات شكل المنحنى الجرسي. فلعنصر المدخلة الواقع على طرف مجال الاستقبال سلسلة واحدة من الوصلات تقوده إلى وحدة المخرج، بينما يمكن لعنصر المدخلة في المركز أن يسلك مسارات كثيرة عبر الطبقات الوسيطة. ويحصي :numref:`fig_conv_rf_paths` هذه المسارات في الطبقات المكدّسة في :numref:`fig_conv_receptive_field`. فللعنصر الأوسط سبعة مسارات إلى الوحدة العلوية، ولكل عنصر طرفي مسار واحد. وإذا تقاربت مقادير الأوزان جميعها، ازداد تأثير عنصر المدخلة بازدياد عدد مساراته.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![عدد المسارات من كل وحدة إلى الوحدة العلوية في :numref:`fig_conv_receptive_field` (ثلاث طبقات أحادية البعد حجم نواتها 3). يساوي عدد المسارات لكل خلية مجموعَ أعداد المسارات من الوحدات التي تغذّيها. وللخلايا الأغمق مسارات أكثر من سائر الخلايا في طبقتها، وتكرّر الأعمدة البيانية الأعداد في صفّ المدخلة، وهي 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
تبقى في طبقتنا الأساسية مشكلتان عمليّتان. فكل التفاف بلا حشو يقلّص الأبعاد المكانية ويأخذ عينات من بكسلات الحدود بوتيرة أقل، ولا تكشف النواة الواحدة إلا نمطًا واحدًا. نتناول في :numref:`sec_padding` ضبط الأبعاد المكانية والتعامل مع الحدود، ثم نتناول خرائط الميزات متعددة القنوات في :numref:`sec_channels`.

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
تحسب الطبقة الالتفافية الارتباط المتقاطع ثنائي الأبعاد بتمرير نواة متعلَّمة على كل نافذة صالحة في المدخلة، ثم تضيف انحيازًا قياسيًا. وتكشف نواة الفروق المحدودة البسيطة ذات الأبعاد $1 \times 2$، `[1, -1]`، الحواف الرأسية أينما ظهرت في الصورة. ويستعيد النزول التدرجي هذه الأوزان نفسها من بيانات المدخلة والمخرج وحدها، مثلما تتعلّم الشبكات العصبية الالتفافية العميقة تسلسلات هرمية كاملة من المرشحات من طرف إلى طرف. ويقلب الالتفاف الرياضي النواة، وهو فرق لا أثر له متى كانت النوى متعلَّمة. ويمكن حساب الالتفاف بعملية واحدة من ضرب المصفوفات (*im2col*). وعند تكديس $L$ طبقةً التفافية، يتّسع مجال استقبال كل وحدة خطيًا بمقدار $k_i - 1$ لكل طبقة.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="تسريع الالتفاف بضرب المصفوفات"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[تحويل im2col من الصورة إلى الأعمدة]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **عملية GEMM كثيفة واحدة**. نسطّح كل نافذة بأبعاد $k_\textrm{h} \times k_\textrm{w}$ في صفّ من `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$، ثم نضرب المصفوفة في $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **المفاضلة**. بلوغ أقصى سرعة لأنوية Tensor Core في معالج GPU مقابل مضاعفة مساحة العمل للرقع بمعامل $k_\textrm{h} k_\textrm{w}\times$.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="صياغة مكافئة بفرد النواة في مصفوفة شريطية"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[الضرب في مصفوفة أوزان من نوع Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
يمكننا بدلًا من فرد صورة المدخلة في رقع مكرّرة أن نفرد *النواة* في مصفوفة شريطية متناثرة $\mathbf{W}$ نضربها في متجه المدخلة المسطّح $\textrm{vec}(\mathbf{X})$.

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **صيغة ضرب مصفوفة في متجه**. $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$، حيث تحتوي $\mathbf{W}$ على $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ صفًّا و$n_\textrm{h} n_\textrm{w}$ عمودًا.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **الصلة بـ §{sec_why-conv} والانتشار العكسي**. يحوي كل صفّ من $\mathbf{W}$ عناصر النواة نفسها، وعددها $k_\textrm{h} k_\textrm{w}$، مع إزاحتها عبر الأعمدة (اللوحة c). وهكذا يكون التمرير الخلفي بالنسبة إلى $\mathbf{X}$ مجرّد ضرب في $\mathbf{W}^\top$ (التفاف منقول).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="خرائط الميزات ومجالات الاستقبال عبر الطبقات"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[الاتّساع الخطي لمجال الاستقبال مع تكديس الطبقات]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**مجال استقبال** الوحدة هو المنطقة من صورة المدخلة الأصلية التي يمكن لبكسلاتها أن تؤثّر في تفعيل تلك الوحدة أثناء التمرير الأمامي.

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **طبقة واحدة ($L=1$)**. ترى النواة ذات الأبعاد $k_1 \times k_1$ رقعة من المدخلة أبعادها $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **تكديس $L$ طبقةً بخطوة انزلاق 1**. تمتدّ الطبقة $i$ التي حجم نواتها $k_i$ على كل محور مسافةً تزيد بمقدار $k_i - 1$ بكسلًا على امتداد عنصر واحد من الطبقة $i-1$.
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **تكديس $L$ طبقةً بأبعاد $3 \times 3$**. تضيف كل طبقة $3 - 1 = 2$ بكسل، فيصبح طول الضلع $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="لماذا يتفوّق تكديس نوى 3×3 على نواة واحدة كبيرة؟"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[كفاءة استخدام المعلمات والقدرة التعبيرية اللاخطية]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **مجال الاستقبال نفسه بأبعاد $7 \times 7$**. ترى ثلاث طبقات مكدّسة بأبعاد $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) الرقعة نفسها ذات الأبعاد $7 \times 7$ التي تراها طبقة واحدة أبعادها $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **معلمات أقلّ بنسبة 45%**. نحتاج إلى $3 \times 9 c^2 = 27 c^2$ وزنًا بدلًا من $49 c^2$ عند استخدام $c$ قناةً.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **قدرة تعبيرية أكبر**. نجمع **3 عمليات تفعيل لاخطية** متتالية بدلًا من 1 (وهي قاعدة التصميم الأساسية في VGG في §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="مجال الاستقبال النظري ومجال الاستقبال الفعلي"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[تناقص الحساسية وفق منحنى غاوسي بالابتعاد عن المركز]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **مجال الاستقبال النظري ($r = 2L+1$)**. صندوق الإحاطة بجميع البكسلات التي يصلها بوحدة المخرج *مسار واحد على الأقلّ*.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **مبرهنة النهاية المركزية**. يزداد عدد مسارات البكسلات المركزية أُسّيًا، وفي الزوايا يكون عدد المسارات لكل بكسل **1 فقط**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **مجال الاستقبال الفعلي (Luo وآخرون، 2016)**. يتناقص مقدار التدرج بالنسبة إلى المدخلة $\partial y / \partial \mathbf{X}$ وفق منحنى جرسي غاوسي انحرافه المعياري $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="اختبر فهمك للارتباط المتقاطع ومجالات الاستقبال"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[اختبر فهمك]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
اختبر فهمك لأشكال المخرجات، ولتحويل `im2col`، ولحساب أبعاد مجال الاستقبال.

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **أبعاد مصفوفة im2col**. طبّق نواة أحادية القناة بأبعاد $3 \times 3$ بلا حشو على صورة أبعادها $10 \times 10$ باستخدام `im2col`. ما الشكل المكاني للمخرج؟ وما الأبعاد الدقيقة لمصفوفة `patches` ولمتجه النواة المسطّح؟

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **العمق اللازم لتغطية صورة من Fashion-MNIST**. عندما تكون خطوة الانزلاق 1، كم طبقةً مكدّسة بأبعاد $3 \times 3$ نحتاج إليها كي يغطي مجال الاستقبال النظري لوحدة واحدة صورةً كاملةً أبعادها $28 \times 28$؟ وكم نحتاج إذا كانت أبعاد الطبقات $5 \times 5$؟

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **أثر تحريك الحدّ في خسارة كشف الحواف**. في تجربة تعلّم كشف الحواف الرأسية في صورة أبعادها $6 \times 8$، لماذا بدأت الخسارة عند $\approx 12$؟ وكم كانت ستبلغ الخسارة الابتدائية لو امتدّ الشريط الأسود إلى الحافة اليمنى للصورة (`X[:, 2:8] = 0`)؟

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**الإجابات**
(1) شكل المخرج $(10-3+1) \times (10-3+1) = 8 \times 8$ (فيه $64$ عنصرًا). ومصفوفة `patches` شكلها $(64, 9)$، ونضربها في متجه النواة ذي الشكل $(9, 1)$.
(2) لأبعاد $3 \times 3$، نحتاج إلى $2L + 1 \ge 28 \implies L = 14$ طبقةً. ولأبعاد $5 \times 5$، نحتاج إلى $4L + 1 \ge 28 \implies L = 7$ طبقات.
(3) في كلٍّ من الصفوف الـ6 حافتان بقيمتي $\pm 1$ ($6 \times 2 \times 1^2 = 12$). وتمديد الشريط الأسود إلى الحدّ الأيمن يزيل الحافة اليمنى التي ينتقل عندها اللون من الأسود إلى الأبيض. فتبقى حافة واحدة في كل صفّ، وتنخفض الخسارة الابتدائية إلى النصف، أي إلى $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="خلاصة القسم وأبرز نتائجه"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[أهمّ أفكار القسم]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **الارتباط المتقاطع ثنائي الأبعاد وتعلّم كشف الحواف**. نُزلق $\mathbf{K}$ على النوافذ الصالحة ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). ويتعلّم النزول التدرجي العشوائي (SGD) الأوزان $[1, -1]$ في 9 خطوات.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **ضرب المصفوفات بتحويل im2col**. يكرّر بيانات الرقع بمعامل $k_\textrm{h} k_\textrm{w}\times$ لبلوغ أقصى إنتاجية لعمليات GEMM على معالجات GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **مجال الاستقبال عند تكديس الطبقات**. $r = 1 + \sum (k_i - 1) = 2L+1$ لطبقات أبعادها $3 \times 3$، مع حساسية تتركّز حول المركز وفق منحنى غاوسي.

=== DECISIONS ===
- patch matrix → «مصفوفة الرقع»; flatten → «سطّح»; unfold/unroll into rows or bands → «فرد»; sparse banded matrix → «مصفوفة شريطية متناثرة»; stacked layers → «طبقات مكدّسة»; effective receptive field → «مجال الاستقبال الفعلي»; Gaussian bell curve → «منحنى جرسي غاوسي».
- Keep im2col, GEMM, GPU, Tensor Core and Toeplitz in Latin script, as in the source. In Arabic prose, describe im2col as a transformation, and keep the distinction between matrix multiplication and the output of a multiplication clear.

=== NOTES ===
- 00011fb7: The slide uses SGD, as in 456670c5. The actual loop takes the full gradient of the loss for the single input–output pair.
- 293436eb, f47d181e: The slide compresses the qualifications in the prose. The Gaussian shape is approximate. The path-count argument in the exercises gives the square-root spread for the profile of the expected squared gradient.
