

#### A/B Testing Project (Bağımsız İki Örneklem T Testi) ####

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# İki grup ortalaması arasında karşılaştırma yapılmak istendiği zaman kullanılır.


# Yapılan çalışmada Facebook 'un teklif verme türleri olan average bidding ve max bidding karşılaştırılmak
# istenmektedir.
# Bu çalışmada kontrol(maximum bidding) ve test(average bidding) grubu olmak üzere iki grubumuz vardır.
# Hedefimiz Facebook'un reklam verme türü olan biddinglerin yapılan değişiklik öncesi ve sonrası getirdiği
# dönüşümler incelenip istatistiksel olarak yapılan değişikliğin faydalı olup olmadığı hakkında bir karara varabilmektir.


#### Değişkenler ####

# Impression: Reklam görüntüleme sayısı
# Click: Tıklanma (Görüntülenen reklama tıklanma sayısını belirtir)
# Purchase: Satın alım (Tıklanan reklamlar sonrası satın alınan ürün sayısı)
# Earning: Kazanç (Satın alınan ürünler sonrası elde edilen kazanç)



#### Veri Setini Hazırlama ####

df_control = pd.read_excel('datasets/ab_testing.xlsx', sheet_name='Control Group')
df_testing = pd.read_excel('datasets/ab_testing.xlsx', sheet_name='Test Group')
df_control.head()
df_testing.head()


#### Görev 1 ####

# A/B Testinin hipotezini tanımlayınız.

# H0: Kontrol grubu ve test grubu arasında dönüşümleri açısından istatistiksel olarak anlamlı bir farklılık yoktur.
# H1: Kontrol grubu ve test grubu arasında dönüşümleri açısından istatistiksel olarak anlamlı bir farklılık vardır.

df_control['Purchase'].mean()  # 550.8941
df_testing['Purchase'].mean()  # 582.1061

# Test sonuçlarına bakıldığında anlamlı bir fark ''varmış'' gibi gözükse de bu farkın şans eseri olup olmadığını
# birkaç istatistiksel yöntem ile test etmek gereklidir.



#### Görev 2 ####

# Normallik Varsayımı:

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

test_stat, pvalue = shapiro(df_control['Purchase'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p value: 0.5891

test_stat, pvalue = shapiro(df_testing['Purchase'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p value: 0.1541

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Burada kontrol grubu için hesaplanan p value değeri 0.05'den küçük olmadığı için normallik varsıyımı sağlanmaktadır.
# Normallik varsayımının sağlanması parametrik test yapılabileceği anlamına gelmektedir. Şimdi diğer varsayımı görmek
# için varyans homojenliğini test edeceğiz.


# Varyans Homojenliği Varsayımı:

# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.

test_stat, pvalue = levene(df_control['Purchase'].dropna(),
                           df_testing['Purchase'].dropna(), equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)

# p value: o.3493

# p- value > 0.05 olduğu için H0 Reddedilemez. Varyans homojenliği varsayımı sağlanmaktadır.


# İki varsayım da sağlandığına göre, A/B Testini (bağımsız iki örneklem t testi) uygulayabiliriz. Bu durumda iki grubun ortalamaları arasında
# anlamlı bir fark vardır varsayımı kabul edilir.


#### Görev 3 ####

# Bu çalışmada A/B Testi yapmayı hedefledik. Genelde birçok sektörde yapılan çalışmalar öncesi ve sonrası olarak iki grup arasında test edilmek istenir.
# Bu test karar verebilmek için neyi bilmek istediğimize bağlı olarak bazen oranlar bazen ortalamalar için yapılır. Biz de bu çalışmada
# Facebook'un yapılan ufak değişiklikler sonrasında reklam verme türünü araştırıyoruz ve satın alma ortalamaları üzerinden karar vermeye çalışıyoruz.
# Ortalamalar kıyaslandığı zaman istaitsiksel olarak anlamlı bir fark olduğu görülebiliyor fakat bunu şansa yer vermeden istatistiksel olarak test etmek ve A/B testi
# yapabilmek için belirli varsayımları test ediyoruz.

# A/B Testinin iki tane varsayımı vardır.
# - Normallik varsayımı
# - Varyans homojenliği varsayımı

# Normallik varsayımını 'Shapiro Wilks' ile Varyans homojenliğini de 'Levene' testi ile sınıyoruz.
# Bu iki varsayım sağlandığı müddetçe parametrik testler uygulanabilir. Eğer varsayımlardan biri bile sağlanmazsa özellikle normallik varsayımı
# o zaman nonparametrik testler kullanılmalıdır.


#### Görev 4 ####

# Reklam verme türünde yapılan değişikliğin faydasını tam olarak görmek için conversation rate ler hesaplanabilir.
# Diğer faktörler de test edilebilir ve böylece öncesi ve sonrası arasındaki fark daha anlaşılır şekilde gözlemlenmiş olur.
