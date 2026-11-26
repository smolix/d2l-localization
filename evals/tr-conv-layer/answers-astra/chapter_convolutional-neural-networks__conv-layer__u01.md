<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=tr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Görüntülerde evrişim

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` bölümünde evrişim katmanını iki geometrik ilkeden, ötelemeye eşdeğişim ve yerellikten yola çıkarak kâğıt üzerinde türettik. Şimdi bu formülün bir görüntü üzerinde ne hesapladığına ve gradyan inişinin veriden yararlı çekirdekleri kendiliğinden bulup bulamadığına bakalım.

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
Aydınlık bir bölgeyle karanlık bir bölgenin birleştiği dikey kenarları bulmak istediğimizi düşünelim. Her pikselin değerinden sağındaki komşusunun değerini çıkardığımızda ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), tekdüze bölgelerin içinde sıfır, beyazdan siyaha geçişte $+1$, siyahtan beyaza geçişte ise $-1$ elde ederiz. İki ağırlıklı $(1, -1)$ algılayıcısını görüntüdeki tüm geçerli konumlarda gezdirdiğimizde bir *kenar haritası* çıkar. Bu bölümde iki boyutlu çapraz korelasyonu kodlayacak, elle tasarladığımız kenar algılayıcısını sınayacak ve gradyan inişinin $(1, -1)$ ağırlıklarını doğrudan veriden *öğrenmesini* sağlayacağız. Ardından evrişimin tek bir matris çarpımına nasıl dönüştüğünü (*im2col*) gösterecek ve katmanları üst üste eklemenin bir birimin alıcı alanını ne kadar genişlettiğini türeteceğiz.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Çapraz korelasyon

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### İşlemin tanımı

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Kanalları şimdilik bir yana bırakalım. Girdi $\mathbf{X}$, $n_\textrm{h} \times n_\textrm{w}$ boyutunda bir matris, çekirdek $\mathbf{K}$ ise $k_\textrm{h} \times k_\textrm{w}$ boyutunda bir matris olsun. Çekirdek, şekli $k_\textrm{h} \times k_\textrm{w}$ olan kayan bir *pencere* tanımlar. Pencerenin sol üst köşesini $(i, j)$ konumuna yerleştirir, pencerenin kapsadığı girdi yamasını $\mathbf{K}$ ile eleman eleman çarparız. Çarpımları toplayarak $(i, j)$ konumundaki çıktı elemanını elde ederiz:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Bu işlem, $\mathbf{X}$ ile $\mathbf{K}$ arasındaki *iki boyutlu çapraz korelasyondur*. Bunu $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ biçiminde yazar, koddaki gibi sıfırdan başlayan indisler kullanırız. :numref:`sec_why-conv` bölümünde göreli konumlar $-\Delta$ ile $\Delta$ arasında merkeze göre simetrikti. Sol üst köşeyi başlangıç almak yalnızca $\Delta$ kadar sabit bir kaymaya yol açar. Böylece merkez pikseli olmayan çift boyutlu çekirdekleri de (örneğin $2 \times 2$ boyutunda bir çekirdeği ya da $1 \times 2$ boyutundaki kenar algılayıcımızı) aynı biçimde kullanabiliriz.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![İki boyutlu çapraz korelasyon. Gölgeli hücreler ilk çıktı elemanını ve onu hesaplamak için kullanılan girdi ve çekirdek elemanlarını gösterir. Hesap şu şekildedir: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` örneğinde $2 \times 2$ boyutundaki çekirdeği $3 \times 3$ boyutundaki girdi üzerinde soldan sağa ve yukarıdan aşağıya birer piksel kaydırırız. Pencere dört konumda girdinin içine bütünüyle sığar ve $2 \times 2$ boyutunda şu çıktıyı verir:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Çekirdeğin tamamı girdinin içinde kalmak zorunda olduğundan çıktı her eksende biraz küçülür. Genişliği $k$ olan bir pencere, uzunluğu $n$ olan bir eksende $n - k + 1$ geçerli konuma sığar. Dolayısıyla çıktının şekli şöyle olur:

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Böylece $3 \times 3$ boyutunda bir girdiye $2 \times 2$ boyutunda bir çekirdek uyguladığımızda $2 \times 2$, $6 \times 8$ boyutunda bir görüntüye $1 \times 2$ boyutunda bir çekirdek uyguladığımızda $6 \times 7$, $28 \times 28$ boyutunda bir Fashion-MNIST görüntüsüne $5 \times 5$ boyutunda bir çekirdek uyguladığımızda ise $24 \times 24$ boyutunda bir çıktı elde ederiz. Uzamsal boyutları korumak istediğimizde girdinin çevresini sıfırlarla doldururuz (*dolgu*). Dolguyu, adım aralığı ve genişletmeyle birlikte :numref:`sec_padding` bölümünde inceleyeceğiz.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### İşlemin kodlanması

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` denklemini doğrudan `corr2d` işlevinde kodlayalım. Bunun için tüm çıktı koordinatlarını iki iç içe döngüyle dolaşırız.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` işlevinin sonucunu :numref:`fig_correlation` örneğindeki el hesabımızla karşılaştırmak için $3 \times 3$ boyutundaki girdi tensörü `X` ile $2 \times 2$ boyutundaki `K` çekirdeğini oluşturup `corr2d(X, K)` çağrısının sonucunu hesaplayalım.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Çıktının her elemanı :numref:`fig_correlation` örneğindekiyle aynıdır. (:numref:`sec_why-conv` bölümünde gördüğümüz gibi, tam matematiksel tanımıyla evrişim, ters çevrilmiş $\mathbf{K}'$ çekirdeğiyle aynı katman ailesini öğrenir. Bu nedenle derin öğrenmede yerleşmiş terimi kullanıp çapraz korelasyona “evrişim” diyoruz.)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Evrişim katmanları

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Katmanın yapısı

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
İki boyutlu bir evrişim katmanı, girdisiyle öğrenilmiş bir çekirdek arasındaki çapraz korelasyonu hesaplar ve sonuca skaler bir ek girdi ekler. :numref:`chap_perceptrons` bölümündeki çok katmanlı algılayıcının ağırlıklarını başlatırken yaptığımız gibi çekirdeğin başlangıç değerlerini $\mathcal{N}(0, 0.01^2)$ dağılımından seçeriz. Ek girdiyi sıfırla başlatırız. Aşağıda `Conv2D` her iki parametreyi de kaydeder ve ileri geçişte `corr2d` işlevini çağırır. Çekirdeğinin yüksekliği $h$, genişliği $w$ olan bir katmana $h \times w$ evrişim katmanı denir.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Tam bağlı katmandan farklı olarak evrişim katmanı belirli bir girdi çözünürlüğüne bağlı değildir, çünkü parametreleri yalnızca çekirdek boyutuna bağlıdır. Aynı $2 \times 2$ evrişim katmanı (toplam 5 parametre) hem $3 \times 3$ boyutundaki `X` tensörümüzü hem de $6 \times 8$ boyutunda bir görüntüyü işler. Çıktıların şekilleri sırasıyla $2 \times 2$ ve $5 \times 7$ olur:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Kenar algılama

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Şimdi girişteki dikey kenar algılayıcısını $6 \times 8$ boyutunda yapay bir görüntü üzerinde sınayalım. Ortadaki dört sütun siyah (`0`), dıştaki sütunlar ise beyazdır (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ boyutundaki çekirdeği şöyle tanımlayalım:

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Her $(i, j)$ konumunda bu çekirdek yatay *sonlu farkı*, yani $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ değerini hesaplar. Bu fark, işaret dışında, parlaklığın yatay yöndeki türevinin ayrık bir yaklaştırımıdır. Parlaklığın sabit olduğu bölgelerde sıfırdır, birden değiştiği yerlerde ise büyüklüğü sıçrama gösterir.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ boyutundaki şerit görüntüsü `X` ile $1 \times 2$ boyutundaki yatay fark çekirdeği `K` arasındaki çapraz korelasyonu `corr2d(X, K)` çağrısıyla hesapladığımızda $6 \times 7$ boyutundaki kenar haritası `Y` elde edilir:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Beklediğimiz gibi `Y`, beyazdan siyaha geçilen sınır boyunca `+1`, siyahtan beyaza geçilen sınır boyunca `-1`, diğer her yerde ise `0` değerini alır. Görüntünün devriğini alarak dikey şeridi yatay hâle getirip aynı çekirdeği uyguladığımızda çıktının tamamı sıfır olur:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Görüldüğü gibi `[1, -1]` yalnızca dikey kenarlara tepki verir. Yatay veya çapraz kenarları, köşeleri ya da dokuları algılamak için bambaşka çekirdekler gerekir.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Çekirdeğin öğrenilmesi

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
Aradığımız özniteliği tam olarak biliyorsak `[1, -1]` çekirdeğini elle tasarlamak kolaydır. Ancak hiçbir mühendis derin bir ağın ara katmanlarındaki binlerce filtreyi tek tek belirleyemez. Gradyan inişi `[1, -1]` çekirdeğini yalnızca `(X, Y)` girdi-çıktı çiftinden bulabilir mi? Bunu her yazılım çatısının sunduğu iki boyutlu evrişim katmanıyla sınayalım. Ek girdiyi kullanmadan ağırlıkları $\mathcal{N}(0, 0.01^2)$ dağılımından başlatalım. 42 çıktı elemanının tamamı üzerinden topladığımız karesel hatayı en aza indirmek için öğrenme oranı $0.03$ olacak şekilde 10 gradyan adımı atalım. Hazır katmanlar yığın ve kanal eksenlerini de içeren dört boyutlu tensörler beklediğinden (:numref:`sec_channels`), `X` ve `Y` tensörlerini buna göre yeniden şekillendiririz.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Başlangıçta kayıp 12 dolayındadır, çünkü ağırlıkları sıfıra yakın rastgele çekirdeğin tahmini, değeri $\pm 1$ olan 12 kenar pikselinin her birinde yaklaşık 1 kadar sapar. Yalnızca dokuz adımda kayıp 0.01'in altına iner. Öğrenilen çekirdeğe bakarak gradyan inişinin ne bulduğunu doğrulayabiliriz:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
İki elemanın her biri `[1, -1]` çiftindeki karşılığından en fazla yaklaşık 0.02 kadar sapar. Döngüde kenarlara ilişkin hiçbir önbilgi yoktu. Gradyan inişi, kayıp sinyalinden yola çıkarak filtreyi kendi başına buldu. Burada tek katmanı doğrudan `Y` ile denetimli olarak eğittik. Derin bir evrişimli sinir ağında ise tüm katmanların çekirdekleri, yalnızca nihai görevin kaybından hareketle geri yayılım yoluyla uçtan uca öğrenilir.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Görüntülerde evrişim<br>**İki boyutlu çapraz korelasyon · kenar algılama · öğrenilen çekirdekler · im2col · alıcı alanlar**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Görüntü boyunca yerel kenarların algılanması"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Çıkış noktası ve temel fikir]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ boyutunda yapay test görüntüsü:** Beyaz dış sütunlar (`1`), dört sütun genişliğinde siyah bir dikey şeridi (`0`) çevreler.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Yatay sonlu fark:** $\mathbf{K} = [1, -1]$ çekirdeği görüntüde gezdirildiğinde $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ farkı hesaplanır. Sonuç beyazdan siyaha geçişte `+1`, siyahtan beyaza geçişte `-1` olur.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Öğrenme ve elle tasarım:** Derin evrişimli sinir ağları bu türden binlerce çekirdeği gradyan inişiyle kendiliğinden bulur.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="İki boyutlu çapraz korelasyon işlemi"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Kayan pencerede iç çarpımlar]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ boyutunda bir $\mathbf{X}$ girdi matrisi ve $k_\textrm{h} \times k_\textrm{w}$ boyutunda bir $\mathbf{K}$ çekirdeği için iki boyutlu çapraz korelasyonu ($\mathbf{Y} = \mathbf{X} \star \mathbf{K}$) hesaplarken pencerenin sol üst köşesini $(i, j)$ konumuna yerleştirir ve karşılıklı elemanların çarpımlarını toplarız:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Başlangıç noktası sol üst köşe:** Sıfırdan başlayan $a, b \ge 0$ indisleri koddaki dizi indisleriyle örtüşür ve çift boyutlu çekirdeklerle de ($2 \times 2$ veya $1 \times 2$ gibi) doğrudan kullanılabilir.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Sol üstteki gölgeli çıktı $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Geçerli pencere konumları ve çıktının uzamsal şekli"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Dolgusuz evrişim ızgarayı neden küçültür?]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Geçerli pencere konumları:** $k_\textrm{h} \times k_\textrm{w}$ boyutunda bir pencere, $n_\textrm{h} \times n_\textrm{w}$ boyutunda bir ızgaranın içine her eksende $n - k + 1$ konumda sığar.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Şekil örnekleri:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) ve $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="İki boyutlu çapraz korelasyonun kodlanması"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[İç içe döngülerle doğrudan hesaplama]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
`corr2d` işlevi $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ boyutundaki çıktı tensörü için yer ayırır ve her elemanı `X[i:i+h, j:j+w] * K` işleminin sonucundaki elemanları toplayarak hesaplar:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Her yinelemede $k_\textrm{h} \times k_\textrm{w}$ boyutunda bir alt pencere `X` tensöründen dilim olarak alınır, `K` ile eleman eleman çarpılır ve sonuç `sum` ile toplanır.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d sonucunun ve ters çevrilmiş çekirdekle eşdeğerliğin doğrulanması"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Çapraz korelasyon ve matematiksel evrişim]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° döndürülmüş çekirdek $\mathbf{K}'$:** $\mathbf{K}$ ile çapraz korelasyon, $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ ile evrişime eşdeğerdir.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Öğrenilen katman ailesi aynıdır:** Yazılım çatıları çapraz korelasyonu hesaplar ve bu işleme evrişim adını verir.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="İki boyutlu evrişim katmanı modülünün kurulması"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Öğrenilebilir çekirdek ağırlıkları ve skaler ek girdi]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Evrişim katmanı, `corr2d` işlevini öğrenilebilir bir $k_\textrm{h} \times k_\textrm{w}$ ağırlık parametresiyle (başlangıç değerleri $\mathcal{N}(0, 0.01^2)$ dağılımından seçilir) ve skaler bir ek girdiyle (başlangıç değeri $0$) bir araya getirir:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Evrişim katmanlarının çözünürlükten bağımsızlığı"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Her girdi boyutunda aynı parametreler]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ boyutundan bağımsızlık:** Aynı 5 parametre ($2 \times 2$ çekirdek + 1 ek girdi) ile ağırlıkları yeniden şekillendirmeden $3 \times 3$, $6 \times 8$ veya megapiksel boyutunda girdiler işlenebilir.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Dikey kenarlar içeren yapay bir görüntünün oluşturulması"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 boyutunda ikili test deseni]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ boyutundaki test görüntüsü `X` için sol ve sağ kenarlardaki elemanları bir, ortadaki dört sütunu sıfır yaparız (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Parçalı sabit sütunlar:** Parlaklık (`1` veya `0`), $0\text{--}1$, $2\text{--}5$ ve $6\text{--}7$ sütun gruplarının her birinde sabittir. Yalnızca $1 \to 2$ ve $5 \to 6$ geçişlerinde değişir.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Yatay sonlu fark çekirdeğinin uygulanması"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] ile dikey kenarların algılanması]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` ile $1 \times 2$ boyutundaki $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ çekirdeği arasındaki çapraz korelasyon, yatayda komşu her piksel çifti için $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ farkını hesaplar:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **İşaretli kenar haritası:** 1. sütunda `+1` (beyazdan siyaha geçiş), 5. sütunda `-1` (siyahtan beyaza geçiş), parlaklığın sabit kaldığı tüm iç sütunlarda ise `0` elde edilir.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Kenar çekirdeklerinde yön seçiciliği"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Tek bir çekirdek neden her yönü algılayamaz?]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Yatay kenarlara duyarsızlık:** `X` tensörünün devriği alındığında (`X.T`) yatay komşular eşit olur. Bu yüzden $[1, -1]$ çekirdeğinin çıktısı her yerde sıfırdır.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Filtre bankaları (§{sec_channels}):** Görüntü işleyen katmanlar yatay, dikey ve çapraz yönler için paralel kanallar öğrenir.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Gradyan inişiyle kenar algılayıcısının öğrenilmesi"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Hazır bir evrişim katmanının (X, Y) çiftiyle denetimli eğitimi]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Yazılım çatılarında dört boyutlu tensörler:** `X` ve `Y` tensörlerini dört boyutlu olacak şekilde yeniden şekillendirin (bir örnek, bir kanal) ve $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ kaybını $\eta = 0.03$ öğrenme oranıyla en aza indirin.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Öğrenilen çekirdek ağırlıklarının incelenmesi"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Kayıp neden 12'den başlar ve ağırlıklar neden [1, -1] değerlerine yakınsar?]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Kayıp neden $12.0$ değerinden başlar?** Sıfıra yakın ağırlıklar $\hat{Y} \approx 0$ tahminini verir. Dolayısıyla değeri $\pm 1$ olan on iki kenar pikselinde tahmin, `Y` içindeki hedeften yaklaşık bir birim sapar ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 adımda yakınsama:** Stokastik gradyan inişi (SGD), $[0.99, -0.99]$ ağırlıklarını veriden kendiliğinden bulur.

=== DECISIONS ===
- Anlatımda birinci çoğul kişi, alıştırma yönergelerinde çoğul emir kipi kullanılır. Başlıklar cümle düzeninde yazılır. Terimlere düzenli olarak İngilizce karşılık eklenmez. Koddaki ve sayısal örneklerdeki ondalık nokta korunur.
- “edge detection / edge detector / edge map” → “kenar algılama / kenar algılayıcısı / kenar haritası”; “locality” → “yerellik”; “patch” → “yama”; “dense layer” → “tam bağlı katman”; “framework” → “yazılım çatısı”; “filter bank” → “filtre bankası”.

=== NOTES ===
- 7e4ef44b: Kaynakta [1, -1] değerlerine yakınsayan nicelik kayıp gibi görünüyor. Deneyde bu değerlere yakınsayanlar çekirdek ağırlıklarıdır.
- 05fceed0: Denetleyici 4 rakamını arıyor. Çeviride aynı sayı “dört sütun” ifadesinde yazıyla verilmiştir.
