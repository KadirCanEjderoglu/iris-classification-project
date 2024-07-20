import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Iris veri setini okuyun
data = pd.read_csv('iris.data', delimiter=',', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Sütun adlarını kontrol edin
print(data.columns)

# Sadece sayısal sütunları seçin
numeric_data = data.select_dtypes(include=[float, int])

# Korelasyon matrisini hesaplayın
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Korelasyon matrisini görselleştirin
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Pair Plot: Tüm özellikler arasındaki ilişkileri incelemek için pair plot kullanın.
sns.pairplot(data, hue='class')
plt.show()

# Violin Plot: Kategori bazında dağılımları göstermek için violin plot kullanın.
sns.violinplot(x='class', y='sepal_length', data=data)
plt.title('Violin Plot of Sepal Length by Class')
plt.show()
# Kutu grafiği (box plot)
sns.boxplot(x=data['sepal_length'])
plt.title('Sepal Length Box Plot')
plt.show()

# Histogram
sns.histplot(data['petal_width'], kde=True)
plt.title('Petal Width Distribution')
plt.show()

# Dağılım grafiği (scatter plot)
sns.scatterplot(x='sepal_length', y='sepal_width', hue='class', data=data)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Kategori bazında ortalama değerleri karşılaştırma (bar plot)
sns.barplot(x='class', y='sepal_length', data=data)
plt.title('Average Sepal Length per Class')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Özellikler ve hedef değişkeni ayırma
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['class']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
