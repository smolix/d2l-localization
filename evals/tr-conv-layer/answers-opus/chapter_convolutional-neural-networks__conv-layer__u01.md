<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=tr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Görüntüler için Evrişimler

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` bölümünde evrişim katmanını iki geometrik ilkeden,
ötelemeye eşdeğişim ve yerellikten, kâğıt üzerinde türettik. Şimdi bu formülün
bir görüntü üzerinde gerçekte neyi hesapladığını ve gradyan inişinin yararlı
çekirdekleri verilerden kendiliğinden bulup bulamayacağını göreceğiz.

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
Parlak bir bölgenin karanlık bir bölgeyle buluştuğu dikey kenarları bulmak
istediğimizi düşünelim. Her pikselden sağındaki komşusunu çıkarmak
($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) tekdüze bölgelerin içinde sıfır,
beyazın siyaha geçtiği yerde $+1$, siyahın yeniden beyaza döndüğü yerde ise
$-1$ verir. İki ağırlıklı $(1, -1)$ dedektörünü görüntüdeki her geçerli konum
üzerinde gezdirmek bir *kenar haritası* üretir. Bu bölümde iki boyutlu çapraz
korelasyonu gerçekleştireceğiz, elle tasarladığımız kenar dedektörünü
sınayacağız, gradyan inişinin $(1, -1)$ çekirdeğini doğrudan verilerden
*öğrenmesini* sağlayacağız, bir evrişimin nasıl tek bir matris çarpımına
dönüştüğünü (*im2col*) göstereceğiz ve katmanları üst üste yığmanın bir birimin
alıcı alanını nasıl genişlettiğini türeteceğiz.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Çapraz Korelasyon

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### İşlem

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Kanalları şimdilik bir kenara bırakarak girdi $\mathbf{X}$ bir
$n_\textrm{h} \times n_\textrm{w}$ matris, çekirdek $\mathbf{K}$ ise bir
$k_\textrm{h} \times k_\textrm{w}$ matris olsun. Çekirdek,
$k_\textrm{h} \times k_\textrm{w}$ şeklinde kayan bir *pencere* tanımlar.
Pencerenin sol üst köşesini $(i, j)$ konumuna yerleştirip kapladığı girdi
parçasını eleman eleman $\mathbf{K}$ ile çarpar ve çarpımları toplarız; böylece
$(i, j)$ konumundaki çıktı elemanını elde ederiz:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Bu, $\mathbf{X}$ ile $\mathbf{K}$ arasındaki *iki boyutlu çapraz
korelasyondur*; kodla uyumlu sıfır tabanlı indislerle
$\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ biçiminde yazılır.
:numref:`sec_why-conv` bölümünde kaymaları $-\Delta$ ile $\Delta$ arasında
simetrik olarak merkezlemiştik; sol üst köşeye sabitlemek bundan yalnızca
$\Delta$ kadarlık sabit bir kaymayla ayrılır, üstelik merkez pikseli olmayan
çift boyutlu çekirdekleri (örneğin $2 \times 2$ bir çekirdeği ya da
$1 \times 2$ kenar dedektörümüzü) doğal biçimde ele alır.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![İki boyutlu çapraz korelasyon. Gölgeli kısımlar ilk çıktı elemanı ile onu hesaplamak için kullanılan girdi ve çekirdek elemanlarıdır: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` içinde $2 \times 2$ bir çekirdeği $3 \times 3$ bir
girdi üzerinde soldan sağa ve yukarıdan aşağıya, her seferinde bir piksel
kaydırırız. Pencere dört konuma tam olarak sığar ve $2 \times 2$ çıktıyı verir:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Çekirdeğin tamamen girdinin içinde kalması gerektiğinden çıktının her eksen
boyunca biraz küçüldüğüne dikkat edin. Genişliği $k$ olan bir pencereyi $n$
uzunluğuna yerleştirmek $n - k + 1$ geçerli konum bırakır; dolayısıyla çıktının
şekli şöyledir:

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Böylece $2 \times 2$ bir çekirdekle $3 \times 3$ bir girdi $2 \times 2$,
$1 \times 2$ bir çekirdekle $6 \times 8$ bir görüntü $6 \times 7$, $5 \times 5$
bir çekirdekle de $28 \times 28$ boyutlu bir Fashion-MNIST görüntüsü
$24 \times 24$ verir. Uzamsal boyutları korumak istediğimizde girdiyi sıfırlardan
oluşan bir kenarlıkla (*dolgu*) çevreleriz; dolguyu adım aralığı ve genişletmeyle
birlikte :numref:`sec_padding` bölümünde inceleyeceğiz.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Gerçekleştirim

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` denklemini, her çıktı koordinatı üzerinde dolaşan iki döngüyle
doğrudan `corr2d` içinde gerçekleştiriyoruz.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` fonksiyonunu :numref:`fig_correlation` içindeki elle yaptığımız
hesapla doğrulamak için $3 \times 3$ girdi tensörü `X` ile $2 \times 2$ çekirdek
`K` değerlerini oluşturup `corr2d(X, K)` ifadesini hesaplıyoruz.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Çıktı, :numref:`fig_correlation` ile eleman eleman aynıdır (ve
:numref:`sec_why-conv` bölümünde gösterildiği gibi, ters çevrilmiş bir
$\mathbf{K}'$ çekirdeğiyle yapılan katı matematiksel evrişim tam olarak aynı
katman ailesini öğrenir; bu yüzden standart derin öğrenme terminolojisini izleyip
çapraz korelasyona "evrişim" diyoruz).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Evrişim Katmanları

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Katman

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
İki boyutlu bir evrişim katmanı, girdisinin öğrenilmiş bir çekirdekle çapraz
korelasyonunu alır ve buna skaler bir ek girdi ekler. :numref:`chap_perceptrons`
içindeki çok katmanlı algılayıcı ağırlıklarımızda olduğu gibi, çekirdeği
$\mathcal{N}(0, 0.01^2)$ dağılımından, ek girdiyi ise sıfır olarak ilklendiririz.
Aşağıda `Conv2D` her iki parametreyi kaydeder ve ileri geçişinde `corr2d`
fonksiyonunu çağırır; yüksekliği $h$, genişliği $w$ olan bir çekirdeğe sahip
katmana $h \times w$ evrişim katmanı denir.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Yoğun bir katmanın aksine, evrişim katmanı sabit bir girdi çözünürlüğüne bağlı
değildir, çünkü parametreleri yalnızca çekirdek boyutuna bağlıdır. Tamamen aynı
$2 \times 2$ katman (toplam 5 parametre) hem $3 \times 3$ tensörümüz `X` için
hem de $6 \times 8$ bir görüntü için çalışır ve sırasıyla $2 \times 2$ ve
$5 \times 7$ şekilli çıktılar döndürür:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Kenar Algılama

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Şimdi girişteki dikey kenar dedektörünü, ortadaki dört sütunu siyah (`0`), dış
sütunları beyaz (`1`) olan $6 \times 8$ boyutlu yapay bir görüntü üzerinde
sınıyoruz.

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ çekirdeği şöyle tanımlıyoruz:

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Bu çekirdek her $(i, j)$ konumunda yatay *sonlu farkı*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ hesaplar. Bu fark, parlaklığın yatay
türevinin (işaret farkıyla) ayrık bir yaklaşımıdır: düz bölgelerde sıfır olur,
yoğunluğun sıçradığı her yerde ise keskin biçimde yükselir.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ boyutlu şeritli görüntü `X` ile $1 \times 2$ boyutlu yatay fark
çekirdeği `K` arasında `corr2d(X, K)` ile çapraz korelasyon almak $6 \times 7$
boyutlu kenar haritası `Y` sonucunu üretir:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Beklendiği gibi `Y`, beyazdan siyaha geçiş sınırı boyunca `+1`, siyahtan beyaza
geçiş sınırı boyunca `-1`, diğer her yerde ise `0` verir. Tamamen aynı çekirdeği
devrik görüntüye uygulamak (dikey şeridi yatay bir şeride çevirerek) yalnızca
sıfırlar verir:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Açıkça görülüyor ki `[1, -1]` yalnızca dikey kenarlara tepki verir. Yatay
kenarları, köşegenleri, köşeleri ya da dokuları algılamak bambaşka çekirdekler
gerektirir.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Bir Çekirdeği Öğrenmek

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
İstediğimiz özniteliği tam olarak bildiğimizde `[1, -1]` çekirdeğini elle
tasarlamak kolaydır; ancak hiçbir mühendis derin bir ağdaki binlerce ara filtreyi
elle belirleyemez. Gradyan inişi `[1, -1]` çekirdeğini yalnızca `(X, Y)`
girdi–çıktı çiftinden geri elde edebilir mi? Bunu her çerçevenin yerleşik 2B
evrişim katmanıyla (ek girdi olmadan ve ağırlıklar $\mathcal{N}(0, 0.01^2)$
dağılımından ilklendirilerek) sınıyoruz: 42 çıktı elemanının tümü üzerinden
toplanan karesel hatayı $0.03$ öğrenme oranıyla 10 gradyan adımı boyunca en aza
indiriyoruz. Yerleşik katmanlar yığın ve kanal eksenlerini de içeren 4B
tensörler beklediğinden (:numref:`sec_channels`), `X` ve `Y` tensörlerini buna
göre yeniden şekillendiriyoruz.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Başlangıçta kayıp 12 civarındadır (sıfıra yakın rastgele çekirdek, 12 adet
$\pm 1$ kenar pikselinin her birini yaklaşık 1 farkla ıskalar) ve yalnızca dokuz
adımda 0.01'in altına düşer. Öğrenilen çekirdeği incelemek neyi bulduğunu
doğrular:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Her iki eleman da `[1, -1]` değerlerinin yaklaşık 0.02 yakınına düşer. Bu döngüdeki
hiçbir şey kenarlar hakkında bir şey bilmiyordu: bir kayıp sinyali verildiğinde
gradyan inişi filtreyi kendi başına keşfetti. Burada tek katmanı doğrudan `Y`
ile denetledik; derin bir evrişimli sinir ağında ise geri yayılım her katmanın
çekirdeklerini yalnızca nihai görev kaybından uçtan uca öğrenir.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Görüntüler için Evrişimler<br>**2B çapraz korelasyon · kenar algılama · öğrenilen çekirdekler · im2col · alıcı alanlar**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Bir Görüntüde Yerel Kenarları Algılamak"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivasyon ve Temel Zihinsel Model]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Yapay $6 \times 8$ test görüntüsü:** Beyaz dış sütunlar (`1`), 4 sütunluk siyah dikey bir bandı (`0`) çevreler.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Yatay sonlu fark:** $\mathbf{K} = [1, -1]$ gezdirilince $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ hesaplanır; beyazdan siyaha kenarlarda `+1`, siyahtan beyaza kenarlarda `-1` çıkar.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Öğrenmek ve elle tasarlamak:** Derin evrişimli ağlar bu tür binlerce çekirdeği gradyan inişiyle kendiliğinden keşfeder.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2B Çapraz Korelasyon İşlemi"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Kayan Pencere İç Çarpımları]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ boyutlu bir girdi matrisi $\mathbf{X}$ ve $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir çekirdek $\mathbf{K}$ verildiğinde, 2B çapraz korelasyon $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ pencerenin sol üst köşesini $(i, j)$ konumuna yerleştirir ve eleman eleman çarpımları toplar:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Sol üst köşe sabitlemesi:** Sıfır tabanlı $a, b \ge 0$ indisleri koddaki dizi indislemesiyle örtüşür ve çift boyutlu çekirdekleri ($2 \times 2$ ya da $1 \times 2$ gibi) sorunsuz ele alır.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Gölgeli sol üst çıktı $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Geçerli Pencere Konumları ve Çıktının Uzamsal Şekli"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Dolgusuz Evrişimler Izgarayı Neden Küçültür]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Geçerli pencere yerleşimleri:** $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir pencere, $n_\textrm{h} \times n_\textrm{w}$ boyutlu bir ızgaraya her eksende $n - k + 1$ konumda sığar.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Şekil örnekleri:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="2B Çapraz Korelasyonu Kodla Gerçekleştirmek"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[İç İçe Döngülerle Doğrudan Gerçekleştirim]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
`corr2d` fonksiyonu $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ boyutlu çıktı tensörünü ayırır ve her elemanını `X[i:i+h, j:j+w] * K` ifadesinin toplamı olarak hesaplar:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Her yineleme `X` içinden $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir alt pencere dilimler, onu eleman eleman `K` ile çarpar ve `sum` ile indirger.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d Doğrulaması ve Ters Çevrilmiş Çekirdek Denkliği"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Çapraz Korelasyon ve Matematiksel Evrişim]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° ters çevrilmiş çekirdek $\mathbf{K}'$:** $\mathbf{K}$ ile çapraz korelasyon, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ ile evrişime eşittir.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Aynı öğrenilen sınıf:** Çerçeveler çapraz korelasyon hesaplar ve buna evrişim der.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2B Evrişim Katmanı Modülü Oluşturmak"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Öğrenilebilir Çekirdek Ağırlıkları ve Skaler Ek Girdi]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Bir evrişim katmanı `corr2d` fonksiyonunu öğrenilebilir $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir ağırlık parametresi ($\mathcal{N}(0, 0.01^2)$ dağılımından ilklendirilir) ve skaler bir ek girdiyle ($0$ olarak ilklendirilir) sarmalar:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Evrişim Katmanlarının Çözünürlükten Bağımsızlığı"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Her Girdi Boyutunda Aynı Parametreler]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ boyutundan bağımsız:** Tamamen aynı 5 parametre ($2 \times 2$ çekirdek + 1 ek girdi), ağırlıklar yeniden şekillendirilmeden $3 \times 3$, $6 \times 8$ ya da megapiksellik girdileri işler.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Yapay Bir Dikey Kenar Görüntüsü Oluşturmak"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 İkili Test Deseni]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Sol ve sağ kenarlarında birler, ortadaki dört sütununda sıfırlar (`X[:, 2:6] = 0`) bulunan $6 \times 8$ boyutlu test görüntüsü `X` tensörünü oluşturuyoruz:

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Parçalı sabit sütunlar:** Yoğunluk $0\text{--}1$, $2\text{--}5$ ve $6\text{--}7$ sütunlarında sabittir (`1` ya da `0`); yalnızca $1 \to 2$ ve $5 \to 6$ sütunları arasında sıçrar.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Yatay Sonlu Fark Çekirdeğini Uygulamak"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] ile Dikey Kenarları Algılamak]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` ile $1 \times 2$ boyutlu $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ çekirdeği arasında çapraz korelasyon almak, her yatay çiftte $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ değerini hesaplar:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **İşaretli kenar haritası:** 1. sütun `+1` (beyazdan siyaha geçiş), 5. sütun `-1` (siyahtan beyaza geçiş) verir; sabit iç sütunların tümü `0` çıkar.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Kenar Çekirdeklerinin Yön Seçiciliği"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Tek Bir Çekirdek Neden Her Yönü Algılayamaz]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Yatay kenarlara kör:** `X` devrildiğinde (`X.T`) yatay komşular özdeş olur, bu yüzden $[1, -1]$ yalnızca sıfır üretir.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Filtre bankaları (§{sec_channels}):** Görü katmanları yatay, dikey ve köşegen yönler için paralel kanallar öğrenir.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Gradyan İnişiyle Bir Kenar Dedektörü Öğrenmek"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Yerleşik Bir Evrişim Katmanını (X, Y) ile Denetlemek]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Çerçevelerin 4B tensörleri:** `X` ve `Y` 4B tensörlere (bir örnek, bir kanal) yeniden şekillendirilir ve $\eta = 0.03$ ile $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ en aza indirilir.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Öğrenilen Çekirdek Ağırlıklarını İncelemek"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Başlangıç Kaybı Neden 12 ve Neden [1, -1] Değerine Yakınsar]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Kayıp neden $12.0$ ile başlar:** Sıfıra yakın ağırlıklar $\hat{Y} \approx 0$ tahmin eder ve `Y` içindeki on iki $\pm 1$ kenar pikselini ıskalar ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 adımda yakınsama:** Stokastik gradyan inişi $[0.99, -0.99]$ değerlerini verilerden kendiliğinden geri elde eder.

=== DECISIONS ===
- Register: first-person plural "biz" for the authors and reader working together ("türettik", "sınıyoruz"); direct instructions to the reader in the polite plural ("dikkat edin").
- Headings in title case (Turkish capitalization: İ/ı respected), noun-phrase or -mAk infinitive form for slide/section titles ("Bir Çekirdeği Öğrenmek").
- Decimal numbers keep the English point as written (`0.03`, `0.01`, `0.02`); Turkish suffixes attach with an apostrophe ("0.01'in").
- References are not suffixed directly; they take a noun: ":numref:`sec_x` bölümünde", ":numref:`fig_x` içinde", ":eqref:`eq_x` denklemini". Inline code and math take a following noun or suffix-free construction where possible ("`corr2d` fonksiyonu", "`X` tensörü").
- English terms are not added in parentheses (glossary terms are standard in Turkish ML texts); `im2col` stays as written.
- "2D/4D" → "2B/4B" (iki/dört boyutlu).
- "edge map" → "kenar haritası"; "edge detection" → "kenar algılama"; "edge detector" → "kenar dedektörü".
- "window" (sliding window) → "pencere"; "sliding window" → "kayan pencere".
- "entry" and "element" (of a matrix/tensor/output) → "eleman" ("girdi" is reserved for input; bias is "ek girdi").
- "implementation" (heading) → "Gerçekleştirim"; "implement" → "gerçekleştirmek".
- "initialize" → "ilklendirmek".
- "framework" (deep learning framework) → "çerçeve".
- "dense layer" → "yoğun katman"; "fully connected" → "tam bağlı".
- "transpose" → "devrik" (adj.), "devirmek" (verb).
- "spatial" → "uzamsal"; "resolution" → "çözünürlük".
- "flipped kernel" → "ters çevrilmiş çekirdek".
- "SGD" in slides → "stokastik gradyan inişi" (glossary term, written out).
- "unit" (a network's neuron) → "birim".

=== NOTES ===
- 12d81c14, 0a4b456b, a569780c: "tokens reordered" warnings come from Turkish word order (kernel before input in 12d81c14; the code `X[:, 2:6] = 0` precedes `X` in 0a4b456b; $\eta = 0.03$ precedes the loss in a569780c); no token was changed.
- 456670c5 vs 67d25b0b: the slide says the kernel converges to $[0.99, -0.99]$, the prose says "within roughly 0.02 of `[1, -1]`"; both translated as written.
