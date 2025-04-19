import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# veriyi yükleme
data = pd.read_csv("titanic.csv")

# seçilen özellikler
df = data[['Pclass','Sex','Age','Survived']].copy()

# cinsiyeti sayısala çevirme 
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# eksik yaş verisi var mı kontrol (gerekirse doldurabilirdik)
# print("Eksik yaş sayısı:", df['Age'].isnull().sum())

print(df.head())

# giriş ve çıkış ayrımı
X = df.drop('Survived', axis=1)
y = df['Survived']

# eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lojistik regresyon
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# değerlendirme
y_pred = model.predict(X_test_scaled)
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Karışıklık matrisi:\n", confusion_matrix(y_test, y_pred))

# yorumlama 
# %79.3 doğruluk basit bir model için iyi
# confusion matrix : 96 TN, 15 FN, 25 FP, 40 TP
# veride pclass,sex and age değişkenleri öenmliydi
# daha iyi sonuç almak için daha fazla özellik ekleyebiliriz yada daha karmaşık modeller deneyebiliriz(randomforest yada SVM gibi)