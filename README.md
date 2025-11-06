# 🎾 Klasyfikacja uderzeń w tenisie

## 📌 Opis projektu

Rozróżniamy 5 klas uderzeń:
- **forhend** (ang. forehand groundstroke, czyli uderzenie po odbiciu piłki od kortu)
- **bekhend** (ang. backhand groundstroke, czyli uderzenie po odbiciu od kortu)
- **forhend wolej** (ang. forehand volley)
- **bekhend wolej** (ang. backhand volley)
- **serwis/smecz** (ang. serve/smatch) (uderzenie od góry)

Tenisiści obecni na filmach, to:
- **Novak Djokovic**
- **Carlos Alcaraz**
- **Pablo Carreño-Busta**
- **Taylor Fritz**
- **Jack Sock**
- **Nieznany** sparingpartner Djokovica z drugiego filmu (z białą koszulką)

Pierwsze człony anotacji nazywają się odpowiednio:
1. **FH**
2. **BH**
3. **FHV**
4. **BHV**
5. **S**

Drugie człony anotacji mogą się nazywać odpowiednio:
1. **ND**
2. **CA**
3. **PBC**
4. **TF**
5. **JS**
6. **U** [od słowa unknown]

Przykład anotacji: Jeśli **Novak Djokovic** wykonuje uderzenie typu **forhend volley**, to anotacja będzie się nazywać **FHV ND**

### Główne elementy:
- Ekstrakcja współrzędnych punktów kluczowych ciała z użyciem **MediaPipe Pose**. Korzystamy z węzłów o następujących numerach: 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28

- **Klasyfikacja:** Zastosowanie i porównanie czterech różnych modeli klasyfikacji:
  1. **DTW + k-NN**
  2. **LDMLT**
  3. **TS2Vec + SVM**
  4. **GRU**
  5. **BiLSTM**

### Rodzaje normalizacji danych:
1. Uniezależnienie od położenia (punkt środkowy zawsze w punkcie zerowym)
2. Uniezależnienie od położenia (punkt w punkcie zerowym tylko w pierwszej klatce, a potem szkielet się porusza tak jak w oryginalnej akcji)
3. Uniezależnienie od położenia + rozmiaru (punkt środkowy zawsze w punkcie zerowym)
4. Uniezależnienie od położenia + rozmiaru (punkt w punkcie zerowym tylko w pierwszej klatce, a potem szkielet się porusza tak jak w oryginalnej akcji)

## 📁 Struktura katalogów

Poniżej znajduje się opis głównych folderów i plików projektu:

- `tennis_stroke_recognition/` - główny katalog projektu
  - `.venv/` - środowisko wirtualne dla Pythona 3.12
  - `data/` - dane wejściowe i wyjściowe
    - `annotations_elan/` - adnotacje w formacie ELAN (.eaf)
    - `raw_videos/` - oryginalne pliki wideo (.mp4)
    - `processed_videos_30fps/` - wideo przekonwertowane do 30 klatek na sekundę
    - `annotations_csv/` - adnotacje wyeksportowane do formatu CSV
    - `processed_features/` - pliki `.pkl` z cechami szkieletu (surowe + znormalizowane)
    - `training_results/` - zapisane wyniki (raport i macierz pomyłek) z powytrenowaniu modelu
    - `annotations_elan.rar` - archiwum zawierające wszystkie pliki adnotacji (.eaf) dla analizowanych filmów, wykonane w programie ELAN
    - `matlab_data_for_LDMLT.mat` - plik w formacie .mat zawierający przetworzone sekwencje cech (punkty szkieletu) oraz odpowiadające im pełne etykiety ('uderzenie gracz'). Jest to plik wejściowy dla skryptów w środowisku MATLAB
  - `src/`
    - `00_extract_features.py` - ekstrakcja punktów szkieletu z wideo
    - `01_normalize_pos.py` - normalizacja względem położenia (punkt środkowy zawsze w punkcie zerowym)
    - `10_train_dtw_knn.py` - skrypt treningowy dla DTW + k-NN
  - `tools/` - skrypty pomocnicze
    - `fetch_30fps_video.py` - konwersja wideo do 30 FPS
    - `convert_pkl_to_mat.py` - skrypt pozwalający przekonwertować dane po normalizacji zapisane w formacie .pkl na format dla Matlaba
    - `matlab_scripts/` - skrypty do eksportu adnotacji z programu ELAN
      - `ELAN m-funkcje/` - folder z funkcjami pomocniczymi dla Matlaba
      - `LDMLT_TS/` - folder zawierający pliki źródłowe biblioteki LDMLT dla środowiska MATLAB
      - `train_ldmlt_knn_hiperparameters.m` - skrypt służący do tuningu hiperparametrów (k, tripletsfactor, cycle i alphafactor) klasyfikatora LDMLT
      - `train_ldmlt_knn.m` - główny skrypt treningowy dla modelu LDMLT w MATLAB
      - `extract_elan_annotations_to_csv.m` - eksport anotacji do CSV
  - `README.md` - dokumentacja projektu
  - `requirements.txt` - plik z listą zależności dla Pythona 3.12

## 🧪 Instrukcja uruchomienia projektu

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/Keriw01/pro_tennis_stroke_recognition.git
cd tennis_stroke_recognition
```

```bash
# Wymaga zainstalowanego Pythona 3.12
py -3.12 -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process # Windows
.\.venv\Scripts\activate          
# source .venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

## 🔄 Przepływ przetwarzania danych

### Krok 0: Instalacja w systemie FFmpeg
Pobierz wersję z: 

https://www.gyan.dev/ffmpeg/builds/

Dodaj ścieżkę `C:\ffmpeg\bin` do zmiennych środowiskowych systemu Windows dla `zmiennych systemowych` w `PATH`. 

### Krok 1: Pobranie i konwersja filmów do 30 FPS
Uruchom skrypt pobierający oryginalne wideo oraz konwertujący wszystkie filmy do 30 klatek na sekundę:

```bash
# Wykonać na środowisku venv z Python 3.12
python tools/fetch_30fps_video.py
```

### Krok 2: Wypakuj wymagany plik i uruchom skrypt w Matlab aby uzyskać adnotacje w formacie .csv
Wypakuj plik `annotations_elan.rar` (ścieżka ma być **data/annotations_elan/**... pliki anotacji). Następnie uruchom skrypt `tools/matlab_scripts/extract_elan_annotations_to_csv.m` w Matlab, aby uzyskać plik .csv potrzebny do uruchomienia skryptu `00_extract_features.py`

### Krok 3: Ekstrakcja cech
Po zakończeniu konwersji uruchom ekstrakcję cech:

```bash
# Wykonać na środowisku venv z Python 3.12
python src/00_extract_features.py
```
Skrypt:
* Wczytuje filmy oraz adnotacje
* Dla każdej adnotacji wyznacza sekwencję punktów szkieletu
* Zapisuje dane do pliku 00_extracted_features.pkl


### Krok 4: Normalizacja cech
```bash
# Wykonać na środowisku venv z Python 3.12
python src/01_normalize_pos.py
```
Skrypt wczytuje surowe dane i normalizuje je poprzez:
* Uniezależnienie od położenia (przesunięcie środka ciężkości do zera)
* Zapisuje dane do pliku 01_normalized_sequences_pos.pkl

### Krok 5.1: Trening i ocena modelu DTW + kNN
```bash
# Wykonać na środowisku venv z Python 3.12
python src/10_train_dtw_knn.py
```
Skrypt:
* Wczytuje znormalizowane dane
* Dzieli je na zbiór treningowy i testowy
* Trenuje klasyfikator DTW + k-NN
* Zapisuje i wyświetla:
    * Dokładność
    * Raport klasyfikacji
    * Macierz pomyłek

Wyniki zostaną zapisane do nowego folderu w:
```bash
data/training_results/10_dtw_knn_results/<timestamp>/
```

### Krok 5.2: Trening i ocena modelu LDMLT
```bash
# Wykonać na środowisku venv z Python 3.12
python tools/convert_pkl_to_mat.py
```
Skrypt:
* Wczytuje znormalizowane dane i zapisuje w formacie który pozwoli na otworzenie tych danych w Matlab

Plik zostanie zapisany w:
```bash
data/matlab_data_for_LDMLT.mat
```

```bash
# Wykonać plik w MATLAB dodając wcześniej folder roboczy wraz biblioteką LDMLT_TS
tools/matlab_scripts/train_ldmlt_knn.m
```

Wyniki zostaną zapisane do nowego folderu w:
```bash
data/training_results/ldmlt_knn_results/<timestamp>/
```

## 🛠️ Wersje oprogramowania
* ELAN: 6.1
* Środowisko .venv: Python 3.12, pip: 25.0.1
* MATLAB R2024a
* Visual Studio Code: September 2025 (version 1.105)