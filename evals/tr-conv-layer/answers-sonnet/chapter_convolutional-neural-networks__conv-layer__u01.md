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
:numref:`sec_why-conv` bölümünde evrişim katmanını, ötelemeye eşdeğişim (translation equivariance) ve yerellik olmak üzere iki geometrik ilkeden yola çıkıp kâğıt üzerinde türetmiştik. Şimdi bu formülün bir görüntü üzerinde aslında neyi hesapladığını ve gradyan inişinin yararlı çekirdekleri verilerden kendiliğinden bulup bulamayacağını görüyoruz.

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
Parlak bir bölgenin koyu bir bölgeyle buluştuğu yerdeki dikey kenarları bulmak isteyelim. Her pikselden sağındaki komşusunu çıkarmak ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), tekdüze bölgelerin içinde sıfır, beyazdan siyaha geçilen yerde $+1$, siyahtan beyaza dönülen yerde ise $-1$ verir. İki ağırlıklı $(1, -1)$ dedektörünü geçerli her görüntü konumunda gezdirmek bir *kenar haritası* üretir. Bu bölümde iki boyutlu çapraz korelasyonu gerçekleştiriyor, elle tasarladığımız kenar dedektörünü deniyor, gradyan inişinin $(1, -1)$ çekirdeğini doğrudan verilerden *öğrenmesini* sağlıyor, bir evrişimin tek bir matris çarpımına nasıl dönüştüğünü (*im2col*) gösteriyor ve katmanları üst üste koymanın bir birimin alıcı alanını nasıl genişlettiğini türetiyoruz.

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
Kanalları şimdilik bir kenara bırakalım; girdi $\mathbf{X}$ bir $n_\textrm{h} \times n_\textrm{w}$ matrisi, çekirdek $\mathbf{K}$ de bir $k_\textrm{h} \times k_\textrm{w}$ matrisi olsun. Çekirdek, $k_\textrm{h} \times k_\textrm{w}$ şeklinde kayan bir *pencere* tanımlar. Pencerenin sol üst köşesini $(i, j)$ konumuna yerleştirip kapsadığı girdi yamasını $\mathbf{K}$ ile eleman eleman çarpar ve çarpımları toplayarak $(i, j)$ konumundaki çıktı girdisini elde ederiz:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Bu, $\mathbf{X}$ ile $\mathbf{K}$'nin *iki boyutlu çapraz korelasyonudur* ve kodla uyumlu olması için sıfır tabanlı indislerle $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ biçiminde yazılır. :numref:`sec_why-conv` bölümünde kaydırmaları $-\Delta$'dan $\Delta$'ya kadar simetrik olarak ortalamıştık; sol üst köşeye demirlemek ise bundan yalnızca $\Delta$'lık sabit bir kaymayla ayrılır, üstelik merkez pikseli olmayan çift boyutlu çekirdekleri (örneğin $2 \times 2$ çekirdek ya da $1 \times 2$ kenar dedektörümüz) doğal biçimde ele alır.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![İki boyutlu çapraz korelasyon. Gölgeli bölgeler ilk çıktı elemanını ve onu hesaplamak için kullanılan girdi ile çekirdek elemanlarını gösterir: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` içinde $2 \times 2$ boyutlu bir çekirdeği $3 \times 3$ boyutlu bir girdi üzerinde soldan sağa ve yukarıdan aşağıya doğru her seferinde bir piksel kaydırıyoruz. Pencere dört konuma tam olarak sığar ve $2 \times 2$ boyutlu çıktıyı verir:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Çekirdeğin tamamen girdinin içinde kalması gerektiğinden çıktının her eksen boyunca biraz küçüldüğüne dikkat edin. Genişliği $k$ olan bir pencereyi uzunluğu $n$ olan bir doğruya yerleştirmenin $n - k + 1$ geçerli konumu vardır; dolayısıyla çıktının şekli şöyledir:

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Böylece $3 \times 3$ boyutlu bir girdi ile $2 \times 2$ boyutlu bir çekirdek $2 \times 2$ boyutlu bir çıktı, $6 \times 8$ boyutlu bir görüntü ile $1 \times 2$ boyutlu bir çekirdek $6 \times 7$ boyutlu bir çıktı, $28 \times 28$ boyutlu bir Fashion-MNIST görüntüsü ile $5 \times 5$ boyutlu bir çekirdek ise $24 \times 24$ boyutlu bir çıktı verir. Uzamsal boyutları korumak istediğimizde girdiyi sıfırlardan oluşan bir kenarlıkla çevreleriz (*dolgu*); bunu adım aralığı ve genişletme ile birlikte :numref:`sec_padding` bölümünde inceleyeceğiz.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Gerçekleştirim

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` denklemini, her çıktı koordinatı üzerinde bir çift döngü kullanarak doğrudan `corr2d` içinde gerçekleştiriyoruz.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` işlevini :numref:`fig_correlation` içindeki elle hesabımızla karşılaştırarak doğrulamak için $3 \times 3$ boyutlu girdi tensörü `X` ile $2 \times 2$ boyutlu çekirdek `K` oluşturuyor ve `corr2d(X, K)` değerini hesaplıyoruz.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Çıktı, :numref:`fig_correlation` ile eleman eleman örtüşür (ve :numref:`sec_why-conv` bölümünde gösterildiği gibi, çekirdeği çevirerek yapılan katı matematiksel evrişim $\mathbf{K}'$ ile aynı katman ailesini öğrenir; bu yüzden standart derin öğrenme terminolojisini izleyip çapraz korelasyona "evrişim" diyoruz).

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
İki boyutlu bir evrişim katmanı, girdisini öğrenilen bir çekirdekle çapraz korelasyona sokar ve skaler bir ek girdi ekler. :numref:`chap_perceptrons` bölümündeki MLP ağırlıklarımızda olduğu gibi, çekirdeği $\mathcal{N}(0, 0.01^2)$ dağılımından, ek girdiyi ise sıfırdan başlatırız. Aşağıda `Conv2D` her iki parametreyi de kaydeder ve ileri geçişinde `corr2d` işlevini çağırır; yüksekliği $h$, genişliği $w$ olan bir çekirdeğe sahip katmana $h \times w$ evrişim katmanı denir.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Tam bağlı bir katmanın aksine, bir evrişim katmanı sabit bir girdi çözünürlüğüne bağlı değildir, çünkü parametreleri yalnızca çekirdek boyutuna bağlıdır. Aynı $2 \times 2$ katman (toplam 5 parametre) hem $3 \times 3$ boyutlu `X` tensörümüzü hem de $6 \times 8$ boyutlu bir görüntüyü işleyerek sırasıyla $2 \times 2$ ve $5 \times 7$ şekillerinde çıktılar döndürür:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Kenar Algılama

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Şimdi girişte sözünü ettiğimiz dikey kenar dedektörünü, ortadaki dört sütunu siyah (`0`), dıştaki sütunları beyaz (`1`) olan $6 \times 8$ boyutlu yapay bir görüntü üzerinde deniyoruz.

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ boyutlu çekirdeği tanımlıyoruz:

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Bu çekirdek her $(i, j)$ konumunda, yatay *sonlu farkı* (finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ hesaplar. Bu fark, parlaklığın yatay türevinin (işaret dışında) ayrık bir yaklaşımıdır; düz bölgelerde sıfır olur, yoğunluğun sıçradığı her yerde ise ani bir değer alır.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ boyutlu şerit görüntüsü `X` ile $1 \times 2$ boyutlu yatay fark çekirdeği `K` arasında `corr2d(X, K)` ile çapraz korelasyon almak, $6 \times 7$ boyutlu `Y` kenar haritasını üretir:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Gerçekten de `Y`, beyazdan siyaha geçiş sınırında `+1`, siyahtan beyaza geçiş sınırında `-1`, başka her yerde ise `0` değerini alır. Aynı çekirdeği, dikey şeridi yatay şeride çeviren devrik görüntüye uyguladığımızda hep sıfır elde ederiz:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Açıkça görülüyor ki `[1, -1]` yalnızca dikey kenarlara tepki verir. Yatay kenarları, çaprazları, köşeleri ya da dokuları algılamak için tamamen farklı çekirdekler gerekir.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Bir Çekirdeği Öğrenme

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
`[1, -1]` çekirdeğini elle tasarlamak, istediğimiz özniteliği tam olarak bildiğimizde kolaydır; ancak hiçbir mühendis derin bir ağın içindeki binlerce ara filtreyi elle belirleyemez. Gradyan inişi, `[1, -1]` çekirdeğini yalnızca `(X, Y)` girdi-çıktı çiftinden geri kazanabilir mi? Bunu, her çerçevenin yerleşik 2B evrişim katmanıyla (ek girdi olmadan, ağırlıklar $\mathcal{N}(0, 0.01^2)$ dağılımından başlatılarak) sınıyoruz: 42 çıktı girdisinin tümü üzerinden toplanan karesel hatayı, $0.03$ öğrenme oranıyla 10 gradyan adımı boyunca en küçültüyoruz. Yerleşik katmanlar yığın ve kanal eksenlerini de içeren 4B tensörler beklediğinden (:numref:`sec_channels`), `X` ve `Y` tensörlerinin şeklini buna göre değiştiriyoruz.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Başlangıçta kayıp 12 civarındadır (sıfıra yakın rastgele çekirdek, $\pm 1$ değerli 12 kenar pikselinin hepsini kabaca 1 farkla ıskalar) ve yalnızca dokuz adım içinde 0.01'in altına düşer. Öğrenilen çekirdeği incelemek, bulduğu şeyi doğrular:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
İki girdi de `[1, -1]` değerine yaklaşık 0.02 içinde ulaşır. Bu döngüdeki hiçbir şey kenarlar hakkında bir şey bilmiyordu: bir kayıp sinyali verildiğinde gradyan inişi filtreyi kendi başına keşfetti. Burada tek katmanı doğrudan `Y` ile denetledik; derin bir CNN'de ise geri yayılım, her katmanın çekirdeklerini yalnızca nihai görev kaybından uçtan uca öğrenir.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Görüntüler için Evrişimler<br>**2B çapraz korelasyon · kenar algılama · öğrenilen çekirdekler · im2col · alıcı alanlar**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Bir Görüntü Boyunca Yerel Kenarları Algılama"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivasyon ve Temel Zihinsel Model]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Yapay $6 \times 8$ test görüntüsü:** 4 sütunluk siyah (`0`) dikey bir bandı çevreleyen beyaz (`1`) dış sütunlar.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Yatay sonlu fark:** $\mathbf{K} = [1, -1]$ çekirdeğini gezdirmek $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ değerini hesaplar; beyazdan siyaha geçişte `+1`, siyahtan beyaza geçişte `-1` verir.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Öğrenme ve elle tasarlama:** Derin CNN'ler bu tür binlerce çekirdeği gradyan inişiyle kendiliğinden keşfeder.

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
- **Sol üst çıpa:** $a, b \ge 0$ sıfır tabanlı indisleri koddaki dizi indislemesiyle örtüşür ve çift boyutlu çekirdekleri ($2 \times 2$ veya $1 \times 2$ gibi) sorunsuz ele alır.

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
::: {.slide title="2B Çapraz Korelasyonu Kodla Gerçekleştirme"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Doğrudan İç İçe Döngülerle Gerçekleştirim]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
`corr2d` işlevi $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ boyutlu çıktı tensörünü ayırır ve her girdiyi `X[i:i+h, j:j+w] * K` toplamı olarak hesaplar:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Her yineleme `X` tensörünün $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir alt penceresini dilimler, `K` ile eleman eleman çarpar ve `sum` ile indirger.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d Doğrulaması ve Çevrilmiş Çekirdek Denkliği"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Çapraz Korelasyon ve Matematiksel Evrişim]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° çevrilmiş çekirdek $\mathbf{K}'$:** $\mathbf{K}$ ile çapraz korelasyon almak, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ ile evrişim almaya eşittir.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Aynı öğrenilen sınıf:** Çerçeveler çapraz korelasyonu hesaplar ve ona evrişim der.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2B Evrişim Katmanı Modülü Kurmak"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Öğrenilebilir Çekirdek Ağırlıkları ve Skaler Ek Girdi]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Bir evrişim katmanı `corr2d` işlevini öğrenilebilir $k_\textrm{h} \times k_\textrm{w}$ boyutlu bir ağırlık parametresiyle ($\mathcal{N}(0, 0.01^2)$ dağılımından başlatılan) ve skaler bir ek girdiyle ($0$ olarak başlatılan) sarmalar:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Evrişim Katmanlarının Çözünürlükten Bağımsızlığı"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Rastgele Girdi Boyutlarında Sabit Parametreler]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ değerinden bağımsız:** Aynı 5 parametre ($2 \times 2$ çekirdek + 1 ek girdi), ağırlıkların şeklini değiştirmeden $3 \times 3$, $6 \times 8$ ya da megapiksel boyutlu girdileri işler.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Yapay Bir Dikey Kenar Görüntüsü Oluşturma"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 İkili Test Deseni]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
`X` test görüntüsünü, sol ve sağ kenarlarında birler, ortadaki dört sütunda sıfırlar (`X[:, 2:6] = 0`) olacak biçimde $6 \times 8$ boyutlu olarak oluşturuyoruz:

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Parçalı sabit sütunlar:** Yoğunluk, $0\text{--}1$, $2\text{--}5$ ve $6\text{--}7$ sütunlarının her birinde sabittir (`1` ya da `0`) ve yalnızca $1 \to 2$ ile $5 \to 6$ sütunları arasında sıçrar.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Yatay Sonlu Fark Çekirdeğini Uygulama"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] ile Dikey Kenarları Algılama]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` ile $1 \times 2$ boyutlu $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ çekirdeği arasında çapraz korelasyon almak, her yatay çift için $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ değerini hesaplar:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **İşaretli kenar haritası:** 1. sütun `+1` (beyazdan siyaha geçiş), 5. sütun `-1` (siyahtan beyaza geçiş) verir; sabit olan tüm iç sütunlar `0` olur.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Kenar Çekirdeklerinin Yönelim Seçiciliği"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Tek Bir Çekirdek Neden Her Yönelimi Algılayamaz]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Yatay kenarlara kör:** `X` tensörünü devirmek (`X.T`) yatay komşuları aynı yapar; bu yüzden $[1, -1]$ hep sıfır çıktısı verir.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Filtre bankaları (§{sec_channels}):** Görü katmanları yatay, dikey ve çapraz yönelimler için paralel kanallar öğrenir.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Gradyan İnişiyle Kenar Dedektörü Öğrenme"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Yerleşik Bir Evrişim Katmanını (X, Y) Üzerinde Denetleme]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4B çerçeve tensörleri:** `X` ve `Y` tensörlerini 4B tensörlere (bir örnek, bir kanal) dönüştürün ve $\eta = 0.03$ ile $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ değerini en küçültün.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Öğrenilen Çekirdek Ağırlıklarını İnceleme"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Başlangıç Kaybı Neden 12 ve [1, -1] Değerine Yakınsıyor]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Kayıp neden $12.0$ ile başlıyor:** Sıfıra yakın ağırlıklar $\hat{Y} \approx 0$ tahmin eder ve `Y` içindeki $\pm 1$ değerli on iki kenar pikselini ıskalar ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 adımda yakınsama:** SGD, $[0.99, -0.99]$ çekirdeğini verilerden kendiliğinden geri kazanır.

=== DECISIONS ===
- Register: first person plural ("-iz" forms), formal-neutral technical prose; imperative in slides ("dönüştürün") only where English is imperative.
- First use of a key term in a unit: Turkish term with the English in parentheses only where the English text itself gives it (e.g. *sonlu fark* (finite difference)); otherwise Turkish only.
- Turkish suffixes after math/code use an apostrophe ($\mathbf{K}$'nin); never attach suffixes directly to protected tokens otherwise.
- "unit" (neuron/activation unit) → "birim"; "window" → "pencere"; "patch" → "yama"; "edge" (image) → "kenar"; "edge map" → "kenar haritası"; "stripe" → "şerit".
- "stack (layers)" → "üst üste koymak"; "yığın" is reserved for batch.
- "dense layer" → "tam bağlı katman"; "scalar bias" → "skaler ek girdi"; "nonlinearity" → "doğrusal olmama"; "implementation" → "gerçekleştirim"; "framework" → "çerçeve"; "activation" → "aktivasyon"; "sparse" → "seyrek"; "banded matrix" → "bant matris".
- "Convolutions for Images" → "Görüntüler için Evrişimler"; "Concept Check" → "Kavram Sınaması"; "Summary" → "Özet"; "Exercises" → "Alıştırmalar".
- Numbers in prose keep a decimal point as written in English (0.01, 0.03), since they match code output.

=== NOTES ===
- 05fceed0, a569780c: Turkish word order (verb last, objects before verbs) moves the code or math tokens relative to the English; all tokens are present and unchanged.
- ae7e906d, 4741e012: false match on "shape": the translation uses "şekli"/"şeklinde", the inflected form of şekil, which drops the vowel and so is not recognised as the glossary term.
