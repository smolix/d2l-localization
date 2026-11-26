<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=tr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Alıştırmalar

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **Elle tasarlanan çekirdekler.** Birkaç çekirdeği elle tasarlayın.
    1. Verilen bir $\mathbf{v} = (v_1, v_2)$ yön vektörü için $\mathbf{v}$ vektörüne dik kenarları, yani $(v_2, -v_1)$ yönündeki kenarları algılayan bir çekirdek türetin.
    1. İkinci türev için bir sonlu fark işleci türetin. Görüntülerde hangi yapılar bu işlece en güçlü tepkiyi verir?
    1. $d$ mertebesinden bir türev elde etmek için gereken en küçük çekirdek boyutu nedir?
    1. Bir bulanıklaştırma çekirdeğini nasıl tasarlardınız? Böyle bir çekirdeği neden kullanmak isteyebilirsiniz?
    1. Bir fotoğraf yükleyin, tek kanala dönüştürün ve alt örneklemeyle yaklaşık $100 \times 100$ piksele küçültün (`corr2d` işlevindeki döngüler Python'da çalışır). Kenar algılayıcısını, bulanıklaştırma çekirdeğini ve “birim çekirdeğin iki katı eksi bulanıklaştırma çekirdeği” biçiminde tanımlanan keskinleştirme çekirdeğini uygulayın. Her sonucu özgün fotoğrafın yanında gösterin. Fotoğrafın kenar haritası, siyah şeridin kenar haritasında göremediğiniz neleri gösteriyor?

    *Stanford CS231n dersinin [2. ödevindeki](https://cs231n.github.io/assignments2024/assignment2/) ConvolutionalNetworks not defterinden uyarlanmıştır.*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **Bant matrisi olarak evrişim.** :numref:`subsec_conv_matmul` bölümündeki yama matrisini oluştururken girdiyi açmıştık. Bunun tersine, çekirdeği de açabiliriz. $\mathbf{X}$, $4 \times 4$ boyutunda bir girdi, $\mathbf{K}$ ise $3 \times 3$ boyutunda bir çekirdek olsun. $\textrm{vec}(\cdot)$ işlemi satırları sırayla alıp tek boyuta açmayı göstersin.
    1. $4 \times 16$ boyutundaki $\mathbf{W}$ matrisini $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ eşitliğini sağlayacak biçimde oluşturun. Burada $\mathbf{Y}$, $\mathbf{X}$ ile $\mathbf{K}$ arasındaki çapraz korelasyondur. Her satırda sıfırdan farklı kaç eleman vardır? $\mathbf{W}$ matrisinde kaç farklı değer bulunur?
    1. $\mathbf{W}$ matrisini kodda `K` çekirdeğinden oluşturun ve matris çarpımının `corr2d(X, K)` ile aynı sonucu verdiğini doğrulayın.
    1. $n \times n$ boyutunda bir girdi ve $k \times k$ boyutunda bir çekirdek için, $\mathbf{W}$ matrisini yoğun bir dizi olarak sakladığınızda dizide kaç eleman olduğunu, $\mathbf{W}$ matrisinde kaç tane sıfırdan farklı eleman bulunduğunu ve yama matrisinde kaç eleman olduğunu hesaplayın. Bu sayıların her biri $n$ ile nasıl büyür?
    1. $\mathbf{W}$ matrisinin seyreklik örüntüsünü koruyun, ancak her satıra kendine özgü $k^2$ değer verin. Böylece yerel bağlı bir katman elde edersiniz. Dördüncü mertebeden $\mathsf{V}$ tensörünü :numref:`sec_why-conv` bölümündeki $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ ifadesinde düşünün. Ötelemeye eşdeğişim ilkesi gereği ağırlıkları birbirine eşitlemeden önce tensörü $k \times k$ boyutunda bir pencere içindeki $(a, b)$ göreli konumlarıyla sınırladığınızda bu katmanı elde edersiniz. $n \times n$ boyutunda bir girdi için bu katmanın, aynı girdi ve çıktı şekillerine sahip tam bağlı bir katmanın ve evrişim katmanının parametre sayılarını bulun.

    *CMU 11-785 dersinin [2. ödevinin 1. kısmından](https://deeplearning.cs.cmu.edu/S21/index.html) uyarlanmıştır.*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **Etkin alıcı alan.** ● Bu bölümdeki kuramsal alıcı alan $r$, hangi girdilerin bir çıktıyı *etkileyebileceğini* belirler. Etkin alıcı alan ise bu girdilerin gerçekte ne kadar etkili olduğunu ölçer :cite:`Luo.Li.Urtasun.ea.2016`. Üst üste $L$ tane tek kanallı, $3 \times 3$ boyutunda evrişim katmanı kurun. Dolgu, ek girdi ve doğrusal olmayan dönüşüm kullanmayın. $32 \times 32$ boyutunda bir girdiyle çalışın ve katmanları oluşturmak için :numref:`subsec_learning_kernel` bölümündeki hazır katmanı kullanın. Her çekirdek elemanını standart Gauss dağılımından bağımsız olarak çekin.
    1. $L = 3$ için çekirdekleri bir kez örnekleyin. Son katmanın merkezine yakın bir çıktı biriminden girdiye birim gradyanı geriye yayın ve girdi gradyanının büyüklüğünü çizin. Gradyanın sıfırdan farklı olduğu bölgeyi $7 \times 7$ boyutundaki kuramsal alıcı alanla karşılaştırın ve büyüklüğün nerede en yüksek olduğunu belirtin.
    1. Çekirdekleri 500 kez örnekleyerek girdi gradyanının karesinin ortalamasını hem $L = 3$ hem de $L = 10$ için hesaplayın. Her ortalama haritada satırları toplayın ve ortaya çıkan sütun profilini, elemanlarının toplamı bir olacak şekilde normalleştirin. Profilin standart sapmasının kuramsal kenar uzunluğu $2L + 1$ değerine oranı derinlikle nasıl değişir?
    1. :numref:`fig_conv_rf_paths` tek boyutta $L = 3$ için yolların sayısını gösterir. Herhangi bir $L$ için girdi gradyanının karesinin beklenen değerini satırlar üzerinden topladığınızda sonucun $L$ adet $(1, 1, 1)$ vektörünün evrişimiyle orantılı olduğunu gösterin. Normalleştirilmiş profilin standart sapmasının $L$ ile nasıl büyüdüğünü türetin.

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **Evrişimde geri geçiş.** ● $\mathbf{Y}$, bir $\mathbf{X}$ girdisiyle bir $\mathbf{K}$ çekirdeği arasındaki tek kanallı çapraz korelasyonun `corr2d` ile hesaplanan sonucu olsun. $\ell$ ise $\mathbf{X}$ ve $\mathbf{K}$ değerlerine yalnızca $\mathbf{Y}$ üzerinden bağlı olan skaler bir kayıp olsun.
    1. $\partial \ell/\partial \mathbf{K}$ ve $\partial \ell/\partial \mathbf{X}$ türevlerini $\partial \ell/\partial \mathbf{Y}$ cinsinden, :numref:`sec_backprop` bölümündeki geri yayılım denklemlerine benzer biçimde türetin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi otomatik türev alma kullanmadan $\ell = \sum_{i,j} G_{ij} Y_{ij}$ kaybı için kodlayın. Burada $\mathbf{G}$, $\mathbf{Y}$ ile aynı şekle sahip, rastgele seçilmiş sabit bir matristir. Sonucunuzu adım büyüklüğü $10^{-2}$ olan merkezî sonlu farkla hesaplanan gradyanla karşılaştırın. Karşılaştırmayı küçük, rastgele seçilmiş `X` ve `K` matrisleri üzerinde yapın ve en büyük bağıl hatayı bildirin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi, otomatik türev alma yoluyla :numref:`subsec_learning_kernel` bölümündeki deneyde hesaplanan çekirdek gradyanıyla karşılaştırarak doğrulayın.

    *Michael Nielsen'ın [Sinir Ağları ve Derin Öğrenme](http://neuralnetworksanddeeplearning.com/chap6.html) kitabının 6. bölümünden ve Stanford CS231n dersinin [2. ödevindeki](https://cs231n.github.io/assignments2024/assignment2/) ConvolutionalNetworks not defterinden uyarlanmıştır.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Tartışmalar](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Tartışmalar](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Tartışmalar](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Tartışmalar](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- “blur kernel / sharpening kernel” → “bulanıklaştırma çekirdeği / keskinleştirme çekirdeği”; “downsample” → “alt örnekleme yapmak”; “locally connected layer” → “yerel bağlı katman”.
- Katman bağlamında “nonlinearity”, doğrusal olmama özelliğini taşıyan işlemin kendisini anlattığında “doğrusal olmayan dönüşüm” olarak çevrilir.

=== NOTES ===
- 6ab5b67c: Kitap adındaki “Neural Networks”, sözlükteki “sinir ağı” teriminin çoğulu olan “Sinir Ağları” ile karşılanmıştır.
