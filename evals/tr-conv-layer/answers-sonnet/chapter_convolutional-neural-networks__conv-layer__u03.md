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
1. [code] **Elle tasarlanmış çekirdekler.** Bazı çekirdekleri elle tasarlayın.
    1. $\mathbf{v} = (v_1, v_2)$ yön vektörü verildiğinde, $\mathbf{v}$'ye dik kenarları, yani $(v_2, -v_1)$ yönündeki kenarları algılayan bir kenar algılama çekirdeği türetin.
    1. İkinci türev için bir sonlu fark işleci türetin. Görüntülerdeki hangi yapılar ona en güçlü tepkiyi verir?
    1. $d$ mertebeden bir türev elde etmek için çekirdeğin en küçük boyutu nedir?
    1. Bir bulanıklaştırma çekirdeğini nasıl tasarlardınız? Böyle bir çekirdeği neden kullanmak isteyebilirsiniz?
    1. Bir fotoğraf yükleyin, tek kanala çevirin ve yaklaşık $100 \times 100$ piksele küçültün (Python'daki `corr2d` döngüleri). Kenar dedektörünü, bulanıklaştırma çekirdeğini ve "kimliğin iki katından bulanıklaştırmanın çıkarılması" biçimindeki keskinleştirme çekirdeğini uygulayın ve her sonucu özgün görüntünün yanında gösterin. Bir fotoğrafın kenar haritası, siyah bandın kenar haritasının gösteremediği neyi gösterir?

    *Stanford CS231n, [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks defterinden uyarlanmıştır.*

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
2. [code] **Bant matris olarak evrişim.** :numref:`subsec_conv_matmul` içindeki yama matrisi girdiyi açar. Ters yapı ise çekirdeği açar. $\mathbf{X}$ bir $4 \times 4$ girdi, $\mathbf{K}$ bir $3 \times 3$ çekirdek ve $\textrm{vec}(\cdot)$ satır satır düzleştirme olsun.
    1. $\mathbf{Y}$, $\mathbf{X}$ ile $\mathbf{K}$'nin çapraz korelasyonu olmak üzere, $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ koşulunu sağlayan $4 \times 16$ boyutlu $\mathbf{W}$ matrisini kurun. Her satırda kaç sıfırdan farklı girdi vardır ve $\mathbf{W}$ içinde kaç farklı değer görünür?
    1. $\mathbf{W}$'yi kodda `K` çekirdeğinden kurun ve çarpımın `corr2d(X, K)` sonucunu yeniden ürettiğini doğrulayın.
    1. $n \times n$ boyutlu bir girdi ve $k \times k$ boyutlu bir çekirdek için, $\mathbf{W}$'nin yoğun bir dizi olarak saklanan girdilerini, $\mathbf{W}$'nin sıfırdan farklı girdilerini ve yama matrisinin girdilerini sayın. Her sayı $n$ ile nasıl büyür?
    1. $\mathbf{W}$'nin seyreklik örüntüsünü koruyup her satıra kendi $k^2$ değerini verin. Ortaya yerel bağlantılı bir katman çıkar: ağırlıkları ötelemeye eşdeğişim bağlamadan önce $(a, b)$ kaydırmalarının $k \times k$ boyutlu bir penceresiyle sınırlanmış olan $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ ifadesindeki, :numref:`sec_why-conv` bölümünün dördüncü mertebeden $\mathsf{V}$ tensörü. $n \times n$ boyutlu bir girdi için bu katmanın, aynı girdi ve çıktı şekillerine sahip tam bağlı bir katmanın ve evrişimin parametrelerini sayın.

    *CMU 11-785, [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html) ödevinden uyarlanmıştır.*

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
3. [code] **Etkin alıcı alan.** ● Bu bölümdeki kuramsal alıcı alan $r$, hangi girdilerin bir çıktıyı etkileyebileceğini sınırlar; etkin alıcı alan ise girdilerin onu gerçekte ne kadar etkilediğini ölçer :cite:`Luo.Li.Urtasun.ea.2016`. $32 \times 32$ boyutlu bir girdi üzerinde; dolgusuz, ek girdisiz ve doğrusal olmama içermeyen $L$ tane tek kanallı $3 \times 3$ evrişim katmanından oluşan bir yığın kurun; bunun için :numref:`subsec_learning_kernel` içindeki yerleşik katmanı kullanın. Her çekirdek girdisini standart normal dağılımdan bağımsız olarak çekin.
    1. $L = 3$ ve çekirdeklerin tek bir çekilişi için, son katmanın merkezine yakın bir çıktı biriminden girdiye doğru birim gradyanı geri yayın ve girdi gradyanının büyüklüğünü çizdirin. Desteğini $7 \times 7$ boyutlu kuramsal alıcı alanla karşılaştırın ve büyüklüğün en fazla olduğu yeri söyleyin.
    1. $L = 3$ ve $L = 10$ için, karesi alınmış girdi gradyanının ortalamasını çekirdeklerin 500 çekilişi üzerinden alın. Ortalaması alınmış her haritayı satırları boyunca toplayın ve ortaya çıkan sütun profilini toplamı bire eşit olacak biçimde normalize edin. Profilin standart sapmasının kuramsal kenar uzunluğu $2L + 1$'e oranı derinlikle nasıl değişir?
    1. :numref:`fig_conv_rf_paths` yolları $L = 3$ için tek boyutta sayar. Genel $L$ için, satırlar üzerinden toplanmış beklenen karesel girdi gradyanının $(1, 1, 1)$ vektörünün $L$ kopyasının evrişimiyle orantılı olduğunu gösterin. Normalize edilmiş profilin standart sapmasının $L$ ile nasıl büyüdüğünü türetin.

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
4. [code] **Bir evrişimin geri geçişi.** ● $\mathbf{Y}$, `corr2d` ile hesaplandığı gibi bir $\mathbf{X}$ girdisinin bir $\mathbf{K}$ çekirdeğiyle tek kanallı çapraz korelasyonu olsun ve $\ell$, $\mathbf{X}$ ile $\mathbf{K}$'ye yalnızca $\mathbf{Y}$ üzerinden bağlı skaler bir kayıp olsun.
    1. :numref:`sec_backprop` içindeki geri yayılım denklemlerinin tarzında, $\partial \ell/\partial \mathbf{K}$ ve $\partial \ell/\partial \mathbf{X}$ ifadelerini $\partial \ell/\partial \mathbf{Y}$ cinsinden türetin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi, $\mathbf{G}$ şekli $\mathbf{Y}$ ile aynı olan sabit bir rastgele matris olmak üzere $\ell = \sum_{i,j} G_{ij} Y_{ij}$ kaybı için otomatik türev alma kullanmadan gerçekleştirin. Bunu, küçük rastgele `X` ve `K` üzerinde adım büyüklüğü $10^{-2}$ olan merkezî sonlu fark gradyanıyla karşılaştırın ve en büyük göreli hatayı bildirin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi, :numref:`subsec_learning_kernel` içindeki deneyde otomatik türev almanın hesapladığı çekirdek gradyanıyla karşılaştırarak doğrulayın.

    *Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html), 6. Bölüm ile Stanford CS231n, [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks defterinden uyarlanmıştır.*

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
- "Discussions" (forum links) → "Tartışmalar".
- "Exercises" → "Alıştırmalar"; exercise imperative uses polite plural ("türetin", "hesaplayın"); "Adapted from X" → "X ... uyarlanmıştır."
- "locally connected layer" → "yerel bağlantılı katman"; "centered finite difference" → "merkezî sonlu fark"; "relative error" → "göreli hata"; "normalize" → "normalize etmek".
- "draw (random numbers)" → "çekmek" ("çekiliş" for a draw); "unit gradient" → "birim gradyan".

=== NOTES ===
- ec8621f8, 54bcee26, 6ab5b67c: math tokens reordered for Turkish word order (e.g. the displayed expression moves after its explanation in exercise 2.4); all tokens are present and unchanged.
- 6ab5b67c: false match on "Neural Networks" (the title of Nielsen's book, kept in English inside a link); "shape" is rendered "şekli", which drops the vowel of şekil.
