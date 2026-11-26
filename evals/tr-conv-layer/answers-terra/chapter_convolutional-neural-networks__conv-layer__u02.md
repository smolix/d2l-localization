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

`corr2d` içindeki iç içe döngüler matematiği açıkça gösterse de Python ile her seferinde bir çıktı pikselini ziyaret etmek gerçek iş yükleri için çok yavaştır. Klasik bir çözüm, bugün de kütüphanelerin seçebildiği algoritmalardan biri olan çapraz korelasyonu tek bir büyük matris çarpımı olarak yeniden ifade etmektir; bu, GPU’ların en yüksek hızda gerçekleştirdiği işlemin ta kendisidir. :numref:`fig_correlation` içindeki her çıktı elemanının, düzleştirilmiş $k_\textrm{h} k_\textrm{w}$ elemanlı çekirdek ile karşılık gelen düzleştirilmiş girdi penceresinin noktasal çarpımından ibaret olduğuna dikkat edin. Tüm $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ pencereyi çıkarıp satırlar olarak üst üste koymak bir parça matrisi verir; bunu düzleştirilmiş çekirdekle çarpmak tüm çapraz korelasyonu tek seferde hesaplar. Bu standart yerleşim dönüşümüne *im2col* ("görüntüden sütunlara", burada ise satırlara) denir. :numref:`fig_conv_im2col` bunu :numref:`fig_correlation` örneği üzerinde gerçekleştirir.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->

![:numref:`fig_correlation` içindeki çapraz korelasyonun tek bir matris çarpımı olarak gösterimi. Girdinin dört $2 \times 2$ penceresinin her biri parça matrisinin bir satırına dönüşür ve düzleştirilmiş çekirdekle çarpım düzleştirilmiş çıktıyı verir. Gölgeli hücreler ilk pencereyi satırına ve 19 çıktısına kadar izler. Merkezdeki 4 pikseli dört pencerenin de içinde yer aldığından her satırda görünür.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->

Bunun karşılığında ek çalışma alanı belleği gerekir; çünkü her iç piksel en fazla $k_\textrm{h} k_\textrm{w}$ parçada çoğaltılır. Aşağıda :numref:`fig_correlation` üzerinde bu tek matris çarpımının `corr2d` ile tam olarak eşleştiğini doğruluyoruz:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->

Fark tam olarak sıfırdır: tek bir matris çarpımı `corr2d` sonucunu eksiksiz yeniden üretir.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->

Eşdeğer olarak, girdi yerine *çekirdeği* düzleştirilmiş görüntüyle çarpılan seyrek bantlı bir $\mathbf{W}$ matrisine açabiliriz. Bu, her satırın aynı birkaç çekirdek ağırlığını tekrarladığı :numref:`fig_why_conv_weight_sharing` içindeki (c) panelinin tam 2D karşılığıdır. Evrişimi yapılandırılmış bir yoğun katman olarak görmek bizi :numref:`sec_why-conv` bölümündeki türetime geri götürür (alıştırmalarda da $\mathbf{W}$ açıkça oluşturulur).

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

Tek bir evrişim katmanı yalnızca çekirdeği boyutunda bir parça görür; oysa nesneleri bütün olarak tanımak çok daha geniş alanlardaki bağlamı birleştirmeyi gerektirir. :numref:`subsec_why-conv-channels` bölümünden hatırlayacağınız gibi, bir evrişim katmanının çıktısına *öznitelik haritası* denir (dikey kenar haritamız `Y` gibi). Bir öznitelik haritasındaki herhangi bir $x$ elemanının *alıcı alanı*, girdi görüntüsünün (veya önceki bir katmanın) değerleri ileri geçiş sırasında $x$ değerini etkileyebilen bölgesidir; bu terim :numref:`field_visual` bölümündeki nörofizyolojiden alınmıştır.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->

### Üst Üste Konmuş Katmanlar

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->

:numref:`fig_correlation` içindeki gölgeli `19` çıktısının, $\mathbf{X}$ matrisinin sol üstteki dört pikselini kapsayan $2 \times 2$ bir alıcı alanı vardır. Şimdi $2 \times 2$ çıktı $\mathbf{Y}$ üzerine *ikinci* bir $2 \times 2$ evrişim katmanı koyarak tek bir skaler $z$ üretelim: $z$, $\mathbf{Y}$ matrisinin dört elemanının tümüne bağlıdır ve bunlar birlikte $\mathbf{X}$ matrisinin dokuz elemanının tümüne bağlıdır. Katmanları üst üste koymak alıcı alanı derinlikle birlikte düzenli biçimde genişletir.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->

Peki ne kadar hızlı büyür? Her $k \times k$ evrişim katmanı, girdisinin tek bir elemanından $k - 1$ piksel daha uzağa ulaşır; dolayısıyla çekirdek boyutları $k_1, \ldots, k_L$ olan $L$ katmanı üst üste koymak şu kenar uzunluğuna sahip bir alıcı alan verir:

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

Art arda $L$ adet $3 \times 3$ katmandan oluşan bir yığında her katman 2 piksel ekler ve $(2L + 1) \times (2L + 1)$ girdiyi kapsar: iki katmandan sonra $5 \times 5$, üç katmandan sonra $7 \times 7$ (:numref:`fig_conv_receptive_field`). Bu doğrusal büyüme, çok büyük çekirdeklere başvurmak için bir neden değil, aslında bir avantajdır: üst üste konmuş üç $3 \times 3$ katman, tek bir $7 \times 7$ katmanla aynı $7 \times 7$ alıcı alanı çok daha az parametre kullanarak (kanal çifti başına 49 yerine $3 \times 9 = 27$ ağırlık) elde ederken aralara bir yerine üç doğrusal olmayan işlem yerleştirir.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->

![Çekirdek boyutu 3 olan üç tek boyutlu katmandan sonra bir birimin alıcı alanı. Gölgeli hücreler en üstteki birimin bağlı olduğu birimlerdir ve çizgiler her birini girdilerine bağlar. Her katman alıcı alanı 2 genişleterek 7 girdiye çıkarır.](../img/conv-receptive-field.svg)

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

$r$ değerinin, her pikselin gerçekte ne kadar güçlü katkıda bulunduğunu söylemeden *teorik* alıcı alanı (bir çıktı birimine *herhangi bir* yolu olan her girdi pikselini) ölçtüğünü unutmayın. :citet:`Luo.Li.Urtasun.ea.2016`, girdi gradyanlarını doğrudan ölçerek hem eğitilmemiş hem de eğitilmiş ağlarda *etkin* alıcı alanın merkezden uzaklaştıkça yaklaşık Gauss biçimli bir çan eğrisiyle hızla azaldığını gösterdi. Derindeki birimler teorik olarak görüntüde çok uzağı görebilir, ancak duyarlılıkları büyük ölçüde merkez yakınında yoğunlaşır.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->

Yolları saymak çan şeklini açıklar. Alıcı alanın kenarındaki bir girdi, çıktı birimine tek bir bağlantı zinciri boyunca ulaşırken merkezdeki girdi aradaki katmanlardan birçok yoldan geçebilir. :numref:`fig_conv_rf_paths`, :numref:`fig_conv_receptive_field` içindeki kule için bu yolları sayar: merkez girdinin en üst birime yedi yolu, her kenar girdisinin ise bir yolu vardır. Tüm ağırlıkların büyüklüğü benzerse bir girdinin etkisi yol sayısıyla birlikte artar.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->

![:numref:`fig_conv_receptive_field` içindeki her birimden en üst birime giden yolların sayısı (çekirdek boyutu 3 olan üç tek boyutlu katman). Her sayı, hücrenin beslediği birimlerin sayılarının toplamıdır. Daha koyu hücrelerin kendi katmanlarındaki diğer hücrelerden daha fazla yolu vardır ve çubuklar girdi satırının sayılarını tekrarlar: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->

Temel katmanımızda iki pratik eksik daha var: dolgusu olmayan her evrişim, sınır piksellerini daha az örneklerken uzamsal boyutları küçültür ve tek bir çekirdek yalnızca tek bir desen algılar. Sırada uzamsal boyutlandırma ve sınırları :numref:`sec_padding` bölümünde, çok kanallı öznitelik haritalarını ise :numref:`sec_channels` bölümünde ele alacağız.

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

Bir evrişim katmanı, çekirdeği girdinin her geçerli penceresi üzerinde kaydırarak öğrenilmiş bir çekirdek ve skaler ek girdiyle iki boyutlu çapraz korelasyon uygular. Basit bir $1 \times 2$ sonlu fark çekirdeği `[1, -1]`, bir görüntünün herhangi bir yerindeki dikey kenarları algılar ve derin evrişimli sinir ağlarının tüm filtre hiyerarşilerini uçtan uca öğrenmesi gibi, gradyan inişi de yalnızca girdi–çıktı verilerinden bu ağırlıkların aynısını geri kazanır. Matematiksel evrişim çekirdeği ters çevirir; çekirdekler öğrenildiğinde bu bir fark yaratmaz. Bir evrişim tek bir matris çarpımı olarak hesaplanabilir (*im2col*) ve $L$ evrişim katmanını üst üste koymak her birimin alıcı alanını katman başına $k_i - 1$ kadar doğrusal biçimde genişletir.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->

::: {.slide title="Hızlı Hesaplama: Matris Çarpımıyla Evrişim"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->

[im2col (Görüntüden Sütunlara) Dönüşümü]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->

- **Tek yoğun GEMM:** Her $k_\textrm{h} \times k_\textrm{w}$ pencereyi `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ içinde bir satıra düzleştirin ve $\textrm{vec}(\mathbf{K})$ ile çarpın.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->

- **Ödünleşim:** Parça çalışma alanının $k_\textrm{h} k_\textrm{w}\times$ çoğaltılması karşılığında en yüksek GPU Tensor Core hızı.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->

::: {.slide title="İkili Bakış: Çekirdeği Bantlı Matriste Açma"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->

[Toeplitz Ağırlık Matrisi Çarpımı]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->

Girdi görüntüsünü yinelenen parçalara açmak yerine *çekirdeği*, düzleştirilmiş girdi vektörü $\textrm{vec}(\mathbf{X})$ ile çarpılan seyrek bantlı bir $\mathbf{W}$ matrisine açabiliriz:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->

- **Matris–vektör biçimi:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$; burada $\mathbf{W}$ matrisinin $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ satırı ve $n_\textrm{h} n_\textrm{w}$ sütunu vardır.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->

- **§{sec_why-conv} ve geri yayılımla bağlantı:** $\mathbf{W}$ matrisinin her satırı, sütunlar boyunca kaydırılmış aynı $k_\textrm{h} k_\textrm{w}$ çekirdek elemanını içerir (panel c); böylece $\mathbf{X}$ için geri geçiş yalnızca $\mathbf{W}^\top$ ile çarpımdır (devrik evrişim).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->

::: {.slide title="Derinlik Boyunca Öznitelik Haritaları ve Alıcı Alanlar"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->

[Üst Üste Katmanlarda Alıcı Alanın Doğrusal Büyümesi]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->

Bir birimin **alıcı alanı**, özgün girdi görüntüsünün pikselleri ileri geçiş sırasında o birimin aktivasyonunu etkileyebilen bölgesidir:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->

- **Tek katman ($L=1$):** $k_1 \times k_1$ çekirdek $k_1 \times k_1$ bir girdi parçası görür.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->

- **Adım aralığı 1 olan $L$ katmanı üst üste koyma:** Çekirdek boyutu $k_i$ olan $i$ katmanı, $i-1$ katmanının tek bir elemanından her eksende $k_i - 1$ piksel daha uzağa ulaşır:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->

- **Üst üste $L$ adet $3 \times 3$ katman:** Her katman $3 - 1 = 2$ piksel ekler ve $r = 2L + 1$ kenar uzunluğunu verir ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->

::: {.slide title="Derin 3×3 Çekirdek Yığınları Neden Tek Bir Büyük Çekirdekten İyidir?"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->

[Parametre Verimliliği ve Doğrusal Olmayan İfade Gücü]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->

- **Eşleşen $7 \times 7$ alıcı alan:** Üst üste üç $3 \times 3$ katman ($r = 1 + 2 + 2 + 2 = 7$), tek bir $7 \times 7$ katmanla aynı $7 \times 7$ parçayı görür.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->

- **45% daha az parametre:** $c$ kanal boyunca $49 c^2$ yerine $3 \times 9 c^2 = 27 c^2$ ağırlık.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->

- **Daha zengin ifade gücü:** 1 yerine **3 doğrusal olmayan aktivasyonu** birleştirir (§{sec_blocks} içindeki temel VGG tasarım kuralı).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->

::: {.slide title="Teorik ve Etkin Alıcı Alanlar"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->

[Duyarlılığın Merkezden Uzaklaştıkça Gauss Biçiminde Azalması]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->

- **Teorik alıcı alan ($r = 2L+1$):** Çıktı birimine *en az bir* yolu olan tüm piksellerin sınırlayıcı kutusu.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->

- **Merkezi Limit Teoremi:** Merkez pikseller üstel sayıda yolla bağlanırken köşe pikselleri yalnızca **1 yolla** bağlanır.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->

- **Etkin alıcı alan (Luo ve diğerleri, 2016):** Girdi gradyanının büyüklüğü $\partial y / \partial \mathbf{X}$, standart sapması $\sigma \propto \sqrt{L}$ olan Gauss biçimli bir çan eğrisiyle azalır.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->

::: {.slide title="Kavram Kontrolü: Çapraz Korelasyon ve Alıcı Alanlar"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->

[Kavram Kontrolü]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->

Çıktı şekilleri, `im2col` ve alıcı alan hesapları hakkındaki bilginizi sınayın:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->

1. **im2col matris boyutları:** `im2col` ile $10 \times 10$ bir görüntüye dolgusuz, tek kanallı $3 \times 3$ bir çekirdek uyguluyorsunuz. Çıktının uzamsal şekli nedir; `patches` matrisinin ve düzleştirilmiş çekirdek vektörünün tam boyutları nelerdir?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->

2. **Fashion-MNIST görüntüsünü kapsayacak derinlik:** Adım aralığı 1 iken tek bir birimin teorik alıcı alanının $28 \times 28$ görüntünün tamamını kapsaması için üst üste kaç $3 \times 3$ katman gerekir? $5 \times 5$ katmanlarla kaç tane gerekir?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->

3. **Kenar kaybında sınır kayması:** $6 \times 8$ dikey kenar öğrenme deneyimizde kayıp neden $\approx 12$ ile başladı ve siyah şerit sağ sınıra kadar uzansaydı (`X[:, 2:8] = 0`) başlangıç kaybı ne olurdu?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->

**Yanıtlar:**
(1) Çıktı şekli $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ çıktı). `patches` matrisi $(64, 9)$ şeklindedir ve $(9, 1)$ çekirdek vektörüyle çarpılır.
(2) $3 \times 3$ için: $2L + 1 \ge 28 \implies L = 14$ katman. $5 \times 5$ için: $4L + 1 \ge 28 \implies L = 7$ katman.
(3) 6 satırın her birinde iki $\pm 1$ kenar vardı ($6 \times 2 \times 1^2 = 12$). Siyah şeridi sağ sınıra uzatmak sağdaki siyahtan beyaza kenarı ortadan kaldırır; satır başına bir kenar kalır ve başlangıç kaybı yarıya, $\approx 6$ değerine iner.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->

::: {.slide title="Özet ve Temel Çıkarımlar"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->

[Bölümün Sentezi]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->

- **2D Çapraz Korelasyon ve Kenar Öğrenme:** $\mathbf{K}$ geçerli pencerelerde kaydırılır ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD $[1, -1]$ değerlerini 9 adımda öğrenir.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->

- **im2col Matris Çarpımı:** En yüksek GPU GEMM işlem hızı için parçalar $k_\textrm{h} k_\textrm{w}\times$ çoğaltılır.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->

- **Üst Üste Alıcı Alanlar:** $3 \times 3$ yığınları için $r = 1 + \sum (k_i - 1) = 2L+1$; merkez duyarlılığı Gauss biçimindedir.

=== NOTES ===
- 24a25ec4, f95c9394, 5898ba57, 01c175bf, 755ac11a, 281c7c78, 16976d18, 3af8acd0: Matematik öğelerinin sırası Türkçe söz dizimine göre değiştirildi.
