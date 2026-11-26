<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=tr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Matris Çarpımı Olarak Evrişim

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
`corr2d` içindeki iç içe döngüler hesabı anlaşılır kılar, ancak Python’da çıktı piksellerini tek tek dolaşmak gerçek iş yükleri için çok yavaştır. Kütüphanelerin hâlâ başvurduğu klasik çözümlerden biri, çapraz korelasyonu tek bir büyük matris çarpımı olarak yazmaktır. GPU’lar tam da bu işlemi en yüksek hızda yapar. :numref:`fig_correlation` üzerindeki her çıktı elemanı, düzleştirilmiş $k_\textrm{h} k_\textrm{w}$ elemanlı çekirdekle karşılık gelen düzleştirilmiş girdi penceresinin nokta çarpımıdır. Tüm $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ pencereyi çıkarıp satır olarak dizince bir parça matrisi elde ederiz. Bu matrisi düzleştirilmiş çekirdekle çarparak çapraz korelasyonun tamamını tek seferde hesaplarız. Bu standart düzen dönüşümüne *im2col* (“görüntüden sütunlara”, burada ise satırlara) denir. :numref:`fig_conv_im2col` bu dönüşümü :numref:`fig_correlation` örneği üzerinde gösterir.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` üzerindeki çapraz korelasyonun tek bir matris çarpımıyla hesabı. Girdideki dört $2 \times 2$ pencerenin her biri parça matrisinin bir satırı olur. Bu matrisi düzleştirilmiş çekirdekle çarpmak düzleştirilmiş çıktıyı verir. Gölgeli hücreler ilk pencerenin karşılık geldiği satırı ve 19 değerli çıktıyı gösterir. Ortadaki 4 değerli piksel dört pencerenin hepsinde yer aldığından her satırda görünür.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Bunun bedeli, işlem sırasında daha fazla bellek kullanılmasıdır. Her iç piksel en çok $k_\textrm{h} k_\textrm{w}$ parçaya kopyalanır. Aşağıda :numref:`fig_correlation` örneğinde bu tek matris çarpımının `corr2d` ile birebir aynı sonucu verdiğini doğruluyoruz.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Fark tam olarak sıfırdır. Tek bir matris çarpımı `corr2d` ile aynı sonucu verir.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Aynı işlemi, girdi yerine *çekirdeği* açarak da yapabiliriz. Böylece düzleştirilmiş görüntüyle çarpılan seyrek, bantlı bir $\mathbf{W}$ matrisi elde ederiz. Bu, :numref:`fig_why_conv_weight_sharing` (c) panelinin iki boyutlu karşılığıdır ve her satırda aynı birkaç çekirdek ağırlığı tekrarlanır. Evrişimi belirli bir yapıya sahip yoğun bir katman olarak düşünmek bizi yeniden :numref:`sec_why-conv` bölümündeki türetime götürür (alıştırmalarda $\mathbf{W}$ matrisini açıkça oluşturacağız).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Öznitelik Haritaları ve Alıcı Alanlar

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Tek bir evrişim katmanı yalnızca çekirdeği boyutunda bir parça görür. Oysa bir nesneyi bütünüyle tanımak için çok daha geniş bölgelerden gelen bağlamı bir araya getirmek gerekir. :numref:`subsec_why-conv-channels` bölümünden hatırlayacağımız gibi, bir evrişim katmanının çıktısına *öznitelik haritası* denir (dikey kenar haritamız `Y` gibi). Öznitelik haritasındaki herhangi bir $x$ elemanının *alıcı alanı*, girdi görüntüsünde (veya önceki bir katmanda) değerleri ileri geçiş sırasında $x$ elemanını etkileyebilen bölgedir. Bu terimi :numref:`field_visual` bölümündeki nörofizyoloji kullanımından alıyoruz.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Üst Üste Katmanlar

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` üzerindeki gölgeli `19` çıktısının $2 \times 2$ alıcı alanı, $\mathbf{X}$ matrisinin sol üstteki dört pikselini kapsar. Şimdi *ikinci* bir $2 \times 2$ evrişim katmanını $2 \times 2$ boyutundaki çıktı $\mathbf{Y}$ üzerine koyup tek bir skaler $z$ elde edelim. $z$, $\mathbf{Y}$ matrisinin dört elemanına da bağlıdır. Bu dört eleman birlikte $\mathbf{X}$ matrisinin dokuz elemanına da bağlıdır. Katmanları üst üste koyduğumuzda derinlik arttıkça alıcı alan düzenli olarak genişler.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Peki ne kadar hızlı genişler? Her $k \times k$ evrişim katmanı, girdisindeki tek bir elemandan $k - 1$ piksel daha uzağa erişir. Dolayısıyla $L$ katmanı, çekirdek boyutları $k_1, \ldots, k_L$ olacak şekilde üst üste koyduğumuzda alıcı alanın kenar uzunluğunu aşağıdaki gibi buluruz.

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
Ardışık $L$ adet $3 \times 3$ katmanda her katman kenar uzunluğuna 2 piksel ekler. Böylece girdinin $(2L + 1) \times (2L + 1)$ boyutunda bir bölgesi kapsanır. Bu bölge iki katmandan sonra $5 \times 5$, üç katmandan sonra $7 \times 7$ boyutundadır (:numref:`fig_conv_receptive_field`). Bu doğrusal büyüme, büyük çekirdeklere başvurmayı gerektirmek yerine bir avantaj sağlar. Üst üste konan üç $3 \times 3$ katman, aynı $7 \times 7$ alıcı alana tek bir $7 \times 7$ katmandan çok daha az parametreyle ulaşır (kanal çifti başına 49 yerine $3 \times 9 = 27$ ağırlık) ve araya bir yerine üç doğrusal olmayan dönüşüm katar.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Çekirdek boyutu 3 olan üç tek boyutlu katmandan sonra bir birimin alıcı alanı. Gölgeli hücreler en üstteki birimin bağlı olduğu birimlerdir. Çizgiler her birini kendi girdilerine bağlar. Her katman alıcı alanı 2 girdi konumu kadar genişletir ve alan sonunda 7 girdiyi kapsar.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Etkin Alıcı Alanlar

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
$r$ değerinin *kuramsal* alıcı alanı (bir çıktı birimine ulaşan *en az bir* yolu olan tüm girdi piksellerini) ölçtüğünü unutmayalım. Bu ölçü her pikselin gerçekte ne kadar katkıda bulunduğunu söylemez. Girdi gradyanlarını doğrudan ölçen :citet:`Luo.Li.Urtasun.ea.2016`, *etkin* alıcı alanın hem eğitilmemiş hem de eğitilmiş ağlarda merkezden uzaklaştıkça yaklaşık Gauss biçiminde bir çan eğrisi izleyerek hızla azaldığını gösterdi. Derin katmanlardaki birimler kuramsal olarak görüntünün uzak noktalarını görebilir, ancak duyarlılıkları büyük ölçüde merkez çevresinde yoğunlaşır.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Yolları sayarak çan biçimini açıklayabiliriz. Alıcı alanın kenarındaki bir girdi, çıktı birimine tek bir bağlantı zinciri üzerinden ulaşır. Merkezdeki girdi ise aradaki katmanlardan geçen birçok yol izleyebilir. :numref:`fig_conv_rf_paths`, :numref:`fig_conv_receptive_field` üzerindeki katmanlar için bu yolları sayar. Merkezdeki girdiden en üstteki birime yedi yol, kenardaki her girdiden ise bir yol vardır. Tüm ağırlıkların büyüklüğü birbirine yakınsa girdinin etkisi, yol sayısı arttıkça büyür.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![:numref:`fig_conv_receptive_field` üzerindeki her birimden en üstteki birime giden yol sayısı (çekirdek boyutu 3 olan üç tek boyutlu katman). Her hücredeki sayı, o hücrenin girdi sağladığı birimlerdeki sayıların toplamıdır. Daha koyu hücrelerin kendi katmanlarındaki diğer hücrelerden daha çok yolu vardır. Çubuklar girdi satırındaki 1, 3, 6, 7, 6, 3, 1 değerlerini tekrar gösterir.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Temel katmanımızın uygulamada iki eksiği var. Dolgusuz her evrişim hem uzamsal boyutları küçültür hem de sınır piksellerini daha az örnekler. Tek bir çekirdek ise yalnızca bir örüntüyü algılar. Uzamsal boyutları ve sınırları sıradaki :numref:`sec_padding` bölümünde, çok kanallı öznitelik haritalarını ise :numref:`sec_channels` bölümünde ele alacağız.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Özet

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
Evrişim katmanı, öğrenilmiş çekirdeği girdinin her geçerli penceresi üzerinde kaydırarak iki boyutlu çapraz korelasyon uygular ve sonuca skaler bir ek girdi ekler. Basit bir $1 \times 2$ sonlu fark çekirdeği `[1, -1]`, görüntünün her yerindeki dikey kenarları algılar. Gradyan inişi tam olarak bu ağırlıkları yalnızca girdi-çıktı verisinden bulur, tıpkı derin evrişimli sinir ağlarının bütün filtre hiyerarşilerini uçtan uca öğrenmesi gibi. Matematiksel evrişim çekirdeği ters çevirir. Bu, çekirdekler öğrenildiğinde sonucu değiştirmez. Evrişim tek bir matris çarpımıyla (*im2col*) hesaplanabilir. $L$ evrişim katmanı üst üste konduğunda her birimin alıcı alanı, katman başına $k_i - 1$ kadar doğrusal olarak genişler.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Matris Çarpımıyla Hızlı Evrişim"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col (Görüntüden Sütunlara) Dönüşümü]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Tek bir yoğun GEMM.** Her $k_\textrm{h} \times k_\textrm{w}$ pencereyi `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ matrisinin bir satırı olacak şekilde düzleştirip matrisi $\textrm{vec}(\mathbf{K})$ ile çarpın.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Bellek ve hız.** GPU’nun Tensor Core birimleri, parçaları işlem belleğinde $k_\textrm{h} k_\textrm{w}\times$ çoğaltma maliyetiyle en yüksek hızda çalışır.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="İkili Bakış: Çekirdeğin Bantlı Matris Olarak Açılması"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Toeplitz Ağırlık Matrisiyle Çarpım]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Girdi görüntüsünü tekrarlanan parçalara ayırmak yerine *çekirdeği* açarak, düzleştirilmiş girdi vektörü $\textrm{vec}(\mathbf{X})$ ile çarpılan seyrek, bantlı bir $\mathbf{W}$ matrisi elde edebiliriz.

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Matris-vektör biçimi.** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$. $\mathbf{W}$ matrisinin $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ satırı ve $n_\textrm{h} n_\textrm{w}$ sütunu vardır.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} ve geri yayımla bağlantı.** $\mathbf{W}$ matrisinin her satırında aynı $k_\textrm{h} k_\textrm{w}$ çekirdek elemanı farklı sütunlara kaydırılır (c paneli). Böylece $\mathbf{X}$ için geri geçiş, yalnızca $\mathbf{W}^\top$ ile çarpımdır (devrik evrişim).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Derinliğe Göre Öznitelik Haritaları ve Alıcı Alanlar"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Üst Üste Katmanlarda Alıcı Alanın Doğrusal Büyümesi]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Bir birimin **alıcı alanı**, asıl girdi görüntüsünde, pikselleri ileri geçişte o birimin aktivasyonunu etkileyebilen bölgedir.

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Tek katman ($L=1$).** $k_1 \times k_1$ çekirdek, $k_1 \times k_1$ boyutunda bir girdi parçasını görür.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Adım aralığı 1 olan $L$ katman üst üste.** $i$ numaralı katmanın çekirdek boyutu $k_i$ ise, alıcı alanı her eksende $k_i - 1$ piksel daha uzağa erişir. Karşılaştırma, $i-1$ numaralı katmandaki tek bir elemanın alıcı alanına göredir.
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **Üst üste $L$ adet $3 \times 3$ katman.** Her katman kenara $3 - 1 = 2$ piksel ekler ve kenar uzunluğu $r = 2L + 1$ olur ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Üst Üste 3×3 Çekirdeklerin Tek Büyük Çekirdeğe Üstünlüğü"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Parametre Verimliliği ve Doğrusal Olmayan İfade Gücü]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Aynı $7 \times 7$ alıcı alan.** Üst üste üç $3 \times 3$ katman ($r = 1 + 2 + 2 + 2 = 7$), tam olarak tek bir $7 \times 7$ katmanın gördüğü $7 \times 7$ parçayı görür.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **%45 daha az parametre.** $3 \times 9 c^2 = 27 c^2$ ağırlık kullanılır. Tek katmanda ise $49 c^2$ ağırlık vardır ($c$ kanal).

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Daha yüksek ifade gücü.** Bir aktivasyon yerine **3 doğrusal olmayan aktivasyonu** bileştirir (§{sec_blocks} bölümündeki temel VGG tasarım ilkesi).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Kuramsal ve Etkin Alıcı Alanlar"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Duyarlılığın Merkezden Uzaklaştıkça Gauss Eğrisine Göre Azalması]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Kuramsal alıcı alan ($r = 2L+1$).** Çıktı birimine ulaşan *en az bir* yolu olan tüm piksellerin sınırlayıcı kutusu.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Merkezi limit teoremi.** Merkez pikselleri üstel sayıda yolla bağlanır. Köşe pikselleri ise yalnızca **1 yolla** bağlanır.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Etkin alıcı alan (Luo vd., 2016).** Girdi gradyanının büyüklüğü $\partial y / \partial \mathbf{X}$, standart sapması $\sigma \propto \sqrt{L}$ olan bir Gauss çan eğrisine göre azalır.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Kavrama Soruları: Çapraz Korelasyon ve Alıcı Alanlar"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Kavrama Soruları]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Çıktı şekilleri, `im2col` ve alıcı alan hesaplarıyla ilgili bilginizi sınayın.

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col matris boyutları.** Dolgusuz, tek kanallı $3 \times 3$ çekirdeği $10 \times 10$ boyutundaki görüntüye `im2col` ile uyguladığınızı düşünün. Çıktının uzamsal şekli nedir? `patches` matrisinin ve düzleştirilmiş çekirdek vektörünün boyutları tam olarak nedir?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST görüntüsünü kapsama derinliği.** Adım aralığı 1 iken, üst üste kaç $3 \times 3$ katman koyarsak tek bir birimin kuramsal alıcı alanı $28 \times 28$ boyutundaki görüntünün tamamını kapsar? $5 \times 5$ katmanlarla kaç katman gerekir?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Şerit sınırının kayba etkisi.** $6 \times 8$ boyutundaki görüntüde dikey kenarları öğrenme deneyinde kayıp neden $\approx 12$ değerinden başladı? Siyah şerit sağ kenara kadar uzansaydı (`X[:, 2:8] = 0`), başlangıç kaybı ne olurdu?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Yanıtlar**
(1) Çıktının şekli $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ çıktı). `patches` matrisinin şekli $(64, 9)$ olup bu matris $(9, 1)$ şeklindeki çekirdek vektörüyle çarpılır.
(2) $3 \times 3$ için $2L + 1 \ge 28 \implies L = 14$ katman, $5 \times 5$ için $4L + 1 \ge 28 \implies L = 7$ katman.
(3) 6 satırın her birinde $\pm 1$ değerli iki kenar vardı ($6 \times 2 \times 1^2 = 12$). Siyah şerit sağ sınıra kadar uzatılınca sağdaki siyahtan beyaza geçiş ortadan kalkar. Satır başına bir kenar kalır ve başlangıç kaybı yarıya, $\approx 6$ değerine iner.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Özet ve Temel Sonuçlar"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Bölüme Genel Bakış]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **İki boyutlu çapraz korelasyon ve kenar öğrenme.** $\mathbf{K}$ çekirdeği geçerli pencereler üzerinde kayar ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). Gradyan inişi $[1, -1]$ ağırlıklarını 9 adımda yaklaşık olarak öğrenir.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col ile matris çarpımı.** Parçaları $k_\textrm{h} k_\textrm{w}\times$ çoğaltma maliyetiyle GPU’da GEMM iş hacmi en yüksek düzeye çıkar.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Üst üste katmanlarda alıcı alan.** $r = 1 + \sum (k_i - 1) = 2L+1$ ifadesi $3 \times 3$ katmanlar için geçerlidir. Duyarlılık merkez çevresinde Gauss eğrisi biçiminde dağılır.

=== DECISIONS ===
- “flatten” → “düzleştirmek”; “unroll” (çekirdeği matris olarak yazmak) → “açmak”; “patch matrix” → “parça matrisi”; “banded matrix” → “bantlı matris”.
- “theoretical receptive field” → “kuramsal alıcı alan”; “effective receptive field” → “etkin alıcı alan”; “sensitivity” → “duyarlılık”.
- “nonlinearity” işlem anlamında → “doğrusal olmayan dönüşüm” (doğrusal olmama burada bir özelliği değil katmanlar arasına konan işlemleri anlatıyor).
=== NOTES ===
- b302e3ef: Türkçede karşılaştırmanın doğal sırası için 49, ağırlık sayısını veren formülden önce yazıldı.
- bd1d7cc0: Tek katmanın boyutu Türkçede nitelediği ifadeden önce geldiği için son iki formül yer değiştirdi.
- 5898ba57: Düzleştirilmiş vektörü niteleyen yan cümle Türkçede matristen önce geldiği için iki formülün sırası değişti.
- 00011fb7: Örnek stokastik güncelleme kullanmıyor ve sonlu sayıda adımda hedef ağırlıklara yaklaşık olarak ulaşıyor. Slayt buna göre çevrildi.
- eed4e2a0: “shape” uyarısı yanlış eşleşme. Burada tensör şekli değil, eğrinin çan biçimi anlatılıyor.
- e3aa1086: “1” uyarısı yanlış eşleşme. Sayı Türkçede “Bir aktivasyon” ifadesiyle korunuyor.
