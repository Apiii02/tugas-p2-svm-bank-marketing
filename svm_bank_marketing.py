import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Lokasi file: argumen terminal jika ada, kalau tidak -> bank.csv di folder file ini
if len(sys.argv) > 1:
    DATA_PATH = Path(sys.argv[1])
else:
    DATA_PATH = Path(__file__).with_name("bank.csv")
OUTPUT_PATH = DATA_PATH.with_name("bank_hasil_preprocessing.csv")

TARGET = "deposit"
LEAKAGE_COL = "duration"


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ======================================================================
# 1. INFO & TIPE KOLOM
# ======================================================================
df = pd.read_csv(DATA_PATH)

judul("1. INFO & TIPE KOLOM")
print(f"Ukuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
print("\nTipe kolom awal:")
print(df.dtypes.to_string())
print("\nStatistik deskriptif (numerik):")
print(df.describe().T.round(2))

# Koreksi tipe:
#  'month' berupa singkatan teks (jan, feb, ...) -> diubah ke angka 1-12,
#  lalu diperlakukan sebagai kategori (bulan bersifat siklik, bukan besaran).
#  'day' (tanggal 1-31) tetap numerik.
BULAN = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
         "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
df["month"] = df["month"].str.strip().str.lower().map(BULAN)
print("\nKoreksi tipe: 'month' -> angka 1-12 (diperlakukan sebagai kategori).")

# ======================================================================
# 2. MISSING VALUES
# ======================================================================
judul("2. MISSING VALUES")
print("Jumlah NaN per kolom:")
print(df.isna().sum().to_string())
print(f"\nTotal NaN: {int(df.isna().sum().sum())}")

kolom_teks = df.select_dtypes(exclude=np.number).columns.tolist()
print("\nMissing tersembunyi (nilai 'unknown'):")
for c in kolom_teks:
    n = int((df[c] == "unknown").sum())
    if n:
        print(f"  {c:10s}: {n:5d} ({n / len(df) * 100:.1f}%)")

# Penanganan 'unknown':
#  - job & education : porsi kecil -> dianggap missing (NaN), diisi modus
#                      oleh imputer di dalam Pipeline (dipelajari dari data latih).
#  - contact & poutcome : porsi besar & bermakna (belum pernah dihubungi /
#                      tidak diketahui) -> dipertahankan sebagai kategori sendiri.
for c in ["job", "education"]:
    df[c] = df[c].replace("unknown", np.nan)
print("\n'unknown' pada job & education -> NaN (diisi modus di Pipeline).")
print("'unknown' pada contact & poutcome -> dipertahankan sebagai kategori.")

# pdays = -1 berarti belum pernah dihubungi di kampanye sebelumnya (bukan angka sebenarnya)
df["pernah_dihubungi"] = (df["pdays"] != -1).astype(int)
df["pdays"] = df["pdays"].replace(-1, 0)
print("pdays = -1 -> 0 dan dibuat fitur biner 'pernah_dihubungi'.")

# ======================================================================
# 3. DUPLIKASI
# ======================================================================
judul("3. DUPLIKASI")
n_dup = int(df.duplicated().sum())
print(f"Jumlah baris duplikat: {n_dup}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Ukuran data setelah pengecekan duplikat: {df.shape}")

# ======================================================================
# 4. KATEGORI (EJAAN / KAPITALISASI)
# ======================================================================
judul("4. CEK KATEGORI (EJAAN / KAPITALISASI)")
for c in df.select_dtypes(exclude=np.number).columns:
    df[c] = df[c].str.strip().str.lower()   # seragamkan huruf kecil & buang spasi
    print(f"\n{c}: {df[c].nunique()} kategori -> {sorted(df[c].dropna().unique())}")
print("\nKesimpulan: ejaan & kapitalisasi sudah konsisten (distandarkan ke huruf kecil).")

# ======================================================================
# 5. OPSIONAL
# ======================================================================
judul("5a. OUTLIER (IQR)")
for kol in ["balance", "campaign", "previous"]:
    q1, q3 = df[kol].quantile([0.25, 0.75])
    iqr = q3 - q1
    bb, ba = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[kol] < bb) | (df[kol] > ba)).sum())
    print(f"{kol:9s}: Q1={q1:8.1f} Q3={q3:8.1f} IQR={iqr:8.1f} "
          f"batas=[{bb:9.1f}, {ba:9.1f}]  outlier={n_out} ({n_out / len(df) * 100:.1f}%)")
print("\nPenanganan: capping (winsorizing) pada 'balance', 'campaign', 'previous',")
print("batas IQR dihitung dari data latih di dalam Pipeline (data tidak dibuang).")

judul("5b. KETIDAKSEIMBANGAN KELAS")
dist = df[TARGET].value_counts()
print(dist.to_string())
print((df[TARGET].value_counts(normalize=True) * 100).round(2).to_string())
rasio = dist.max() / dist.min()
print(f"\nRasio mayoritas:minoritas = {rasio:.2f}:1 ->",
      "relatif seimbang, tidak perlu resampling." if rasio < 1.5
      else "tidak seimbang, pertimbangkan resampling / class_weight.")

judul("5c. INDIKASI KEBOCORAN DATA (LEAKAGE)")
y_num = (df[TARGET] == "yes").astype(int)
korelasi = df.select_dtypes(include=np.number).corrwith(y_num).sort_values(ascending=False)
print("Korelasi fitur numerik dengan target:")
print(korelasi.round(3).to_string())
print(f"\n'{LEAKAGE_COL}' (lama telepon) baru diketahui SETELAH panggilan selesai,")
print("sehingga tidak tersedia saat prediksi nyata -> indikasi leakage -> di-drop.")
print("Selain itu scaler/imputer/encoder di-fit hanya pada data latih (split dulu).")


# ======================================================================
# 6. PIPELINE PRA-PEMROSESAN
# ======================================================================
class IQRCapper(BaseEstimator, TransformerMixin):
    """Capping outlier dengan batas IQR (Q1-1.5*IQR, Q3+1.5*IQR) dari data latih."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.bawah_, self.atas_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.bawah_, self.atas_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


judul("6. NORMALISASI, STANDARISASI & ENCODING")

X = df.drop(columns=[TARGET, LEAKAGE_COL]).copy()
X["month"] = X["month"].astype(str)          # month diperlakukan sebagai kategori
y = (df[TARGET] == "yes").astype(int)

fitur_outlier = ["balance", "campaign", "previous"]            # di-cap IQR lalu distandarkan
fitur_numerik = ["age", "day", "pdays"]                        # langsung distandarkan
fitur_biner = ["default", "housing", "loan"]                   # yes/no -> 0/1
fitur_kategori = ["job", "marital", "education", "contact", "poutcome", "month"]
fitur_lain = ["pernah_dihubungi"]                              # sudah 0/1

for c in fitur_biner:
    X[c] = X[c].map({"yes": 1, "no": 0})

# Split SEBELUM preprocessing agar tidak ada kebocoran dari scaler/imputer
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Data latih: {X_train.shape}  Data uji: {X_test.shape}")

# Demonstrasi: sebelum vs sesudah normalisasi & standarisasi (data latih)
kolom_demo = fitur_outlier + fitur_numerik
ringkas = ["mean", "std", "min", "max"]
print("\nSebelum scaling:")
print(X_train[kolom_demo].describe().loc[ringkas].round(2))
print("\nSetelah Normalisasi (Min-Max, rentang 0-1):")
print(pd.DataFrame(MinMaxScaler().fit_transform(X_train[kolom_demo]),
                   columns=kolom_demo).describe().loc[ringkas].round(2))
print("\nSetelah Standarisasi (Z-score, mean~0, std~1):")
print(pd.DataFrame(StandardScaler().fit_transform(X_train[kolom_demo]),
                   columns=kolom_demo).describe().loc[ringkas].round(2))
print("\nCatatan: SVM (RBF) berbasis jarak -> fitur wajib satu skala. "
      "Model memakai Standarisasi.")

pipe_outlier = Pipeline([("cap", IQRCapper()), ("scaler", StandardScaler())])
pipe_numerik = Pipeline([("scaler", StandardScaler())])
pipe_biner = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
pipe_kategori = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),   # isi 'unknown' job/education
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("outlier", pipe_outlier, fitur_outlier),
    ("num", pipe_numerik, fitur_numerik),
    ("bin", pipe_biner, fitur_biner),
    ("cat", pipe_kategori, fitur_kategori),
    ("lain", "passthrough", fitur_lain),
])

# ======================================================================
# 7. KLASIFIKASI SVM
# ======================================================================
judul("7. KLASIFIKASI SVM (kernel RBF)")
model = Pipeline([
    ("preprocessing", preprocessor),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc:.4f} ({acc * 100:.2f}%)")
print("\nConfusion matrix (baris=aktual, kolom=prediksi; urutan no, yes):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["no", "yes"], zero_division=0))

# ======================================================================
# 8. 5 TERATAS & 5 TERBAWAH HASIL PEMROSESAN
# ======================================================================
prep = model.named_steps["preprocessing"]
hasil = prep.transform(X)
if hasattr(hasil, "toarray"):
    hasil = hasil.toarray()
nama_kolom = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
hasil_df = pd.DataFrame(hasil, columns=nama_kolom, index=df.index)
hasil_df[TARGET] = df[TARGET].values
hasil_df.to_csv(OUTPUT_PATH, index=False)

judul("8a. 5 DATA TERATAS HASIL PEMROSESAN")
print(hasil_df.head(5).round(3).to_string())
judul("8b. 5 DATA TERBAWAH HASIL PEMROSESAN")
print(hasil_df.tail(5).round(3).to_string())
print(f"\nUkuran data hasil pemrosesan: {hasil_df.shape}")
print(f"Disimpan ke: {OUTPUT_PATH}")