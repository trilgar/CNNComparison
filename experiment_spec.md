# Експериментальне порівняння архітектур для класифікації HAM10000

## Контекст

Це завдання є частиною літературного огляду "Класифікація ракових захворювань шкіри з використанням згорткових мереж". Мета — провести власні експерименти з порівняння архітектур, описаних у рефераті, і отримати таблиці метрик для нового розділу.

---

## 1. Датасет

**HAM10000** — 10 015 дерматоскопічних зображень, 7 класів.

Завантаження:
```bash
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```

Альтернативно: `torchvision` не містить HAM10000, тому можна використати [medmnist](https://medmnist.com/) (`DermaMNIST`), але краще оригінальний HAM10000 з Kaggle.

### Класи HAM10000

| Клас | Скорочення | Кількість | Частка |
|---|---|---|---|
| Melanocytic nevus | nv | 6705 | 66.95% |
| Melanoma | mel | 1113 | 11.11% |
| Benign keratosis | bkl | 1099 | 10.97% |
| Basal cell carcinoma | bcc | 514 | 5.13% |
| Actinic keratosis | akiec | 327 | 3.27% |
| Vascular lesion | vasc | 142 | 1.42% |
| Dermatofibroma | df | 115 | 1.15% |

### Критично важливо: split strategy

**НЕ** використовувати випадковий train/test split. HAM10000 містить кілька зображень одного ураження (різні збільшення, кути). Якщо одне ураження потрапить і в train, і в test — буде data leakage.

Правильний підхід:
1. У файлі `HAM10000_metadata.csv` є колонка `lesion_id`.
2. Split робити на рівні `lesion_id`, а не `image_id`.
3. Усі зображення одного `lesion_id` мають бути або в train, або в test.

Рекомендований split: **80% train / 20% test** на рівні `lesion_id`, стратифіковано за класом.

---

## 2. Архітектури для порівняння

### 2.1 ResNet-50 (CNN, residual connections)
- Джерело: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- PyTorch: `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- Замінити останній `fc` шар на `nn.Linear(2048, 7)`

### 2.2 DenseNet-121 (CNN, dense connections)
- Джерело: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
- PyTorch: `torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)`
- Замінити `classifier` на `nn.Linear(1024, 7)`

### 2.3 EfficientNet-B0 (CNN, compound scaling)
- Джерело: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
- PyTorch: `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`
- Замінити `classifier[1]` на `nn.Linear(1280, 7)`

### 2.4 ViT-B/16 (Vision Transformer)
- Джерело: Dosovitskiy et al., "An Image is Worth 16×16 Words", ICLR 2021
- PyTorch: `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`
- Замінити `heads.head` на `nn.Linear(768, 7)`
- **Вхідний розмір: 224×224** (ViT потребує фіксований розмір)

### 2.5 Hybrid CNN-Transformer
- Ідея: Nie et al., "A deep CNN transformer hybrid model for skin lesion classification", Diagnostics 2023
- Реалізація: ResNet-50 як feature extractor (видалити останній fc + avgpool), вихід — feature map 7×7×2048. Розділити на патчі (кожен патч = один 2048-вимірний вектор), додати positional encoding, подати у кілька шарів `nn.TransformerEncoderLayer`, забрати [CLS] token або global average, подати у `nn.Linear(dim, 7)`.

Конкретна архітектура гібриду:

```python
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=7, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        # CNN backbone: ResNet-50 без останніх шарів
        resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # output: B×2048×7×7
        
        # Projection: 2048 → d_model
        self.proj = nn.Linear(2048, d_model)
        
        # Positional encoding (learnable, 49 patches + 1 CLS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # CNN features: B×2048×7×7
        features = self.backbone(x)
        B, C, H, W = features.shape
        
        # Reshape to patches: B×49×2048 → project → B×49×d_model
        patches = features.flatten(2).transpose(1, 2)  # B×49×2048
        patches = self.proj(patches)  # B×49×d_model
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # B×50×d_model
        tokens = tokens + self.pos_embed
        
        # Transformer
        out = self.transformer(tokens)
        
        # Classify from CLS token
        return self.head(out[:, 0])
```

---

## 3. Тренування

### 3.1 Загальні гіперпараметри (однакові для всіх моделей)

| Параметр | Значення |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 (backbone), 1e-3 (head) — differential LR |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR, T_max = num_epochs |
| Batch size | 32 |
| Epochs | 30 |
| Input size | 224 × 224 |
| Нормалізація | ImageNet mean/std |

### 3.2 Аугментація (однакова для всіх)

```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 3.3 Боротьба з дисбалансом

Використовувати **weighted cross-entropy**, де ваги обернено пропорційні частотам класів:

```python
class_counts = [6705, 1113, 1099, 514, 327, 142, 115]
total = sum(class_counts)
weights = [total / (len(class_counts) * c) for c in class_counts]
weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

### 3.4 Transfer learning стратегія

Для всіх моделей:
1. Заморозити backbone на перші 5 епох (тренувати тільки classification head).
2. Розморозити все і тренувати ще 25 епох з differential learning rate.

Для гібридної моделі: заморозити ResNet backbone на перші 5 епох, тренувати Transformer + head. Потім розморозити все.

---

## 4. Метрики

Обчислити на **тестовій вибірці** після завершення тренування кожної моделі:

### 4.1 Загальні метрики (одне число на модель)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

### 4.2 Per-class метрики (для аналізу мінорних класів)

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
```

### 4.3 Confusion matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_true, y_pred)
# Зберегти як зображення для кожної моделі
```

---

## 5. Очікувані результати (формат таблиць)

### Таблиця 1. Загальне порівняння архітектур на HAM10000 (test set)

| Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Параметри (M) | Час тренування |
|---|---|---|---|---|---|---|---|
| ResNet-50 | — | — | — | — | — | 23.5 | — |
| DenseNet-121 | — | — | — | — | — | 7.0 | — |
| EfficientNet-B0 | — | — | — | — | — | 4.0 | — |
| ViT-B/16 | — | — | — | — | — | 86.6 | — |
| Hybrid CNN-Transformer | — | — | — | — | — | ~30 | — |

### Таблиця 2. Per-class F1-score

| Модель | nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|---|
| ResNet-50 | — | — | — | — | — | — | — |
| DenseNet-121 | — | — | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — | — | — |
| ViT-B/16 | — | — | — | — | — | — | — |
| Hybrid CNN-Transformer | — | — | — | — | — | — | — |

---

## 6. Що зберегти

Після завершення всіх експериментів зберегти у папку `results/`:

1. **`results/comparison_table.csv`** — Таблиця 1 у CSV форматі
2. **`results/per_class_f1.csv`** — Таблиця 2 у CSV форматі  
3. **`results/confusion_matrices/`** — PNG confusion matrix для кожної моделі
4. **`results/training_logs/`** — CSV з train/val loss і accuracy по епохах для кожної моделі
5. **`results/training_curves.png`** — графік loss і accuracy по епохах для всіх моделей на одному рисунку
6. **`results/classification_reports/`** — текстовий `classification_report` для кожної моделі
7. **`models/`** — збережені ваги найкращої епохи (best val loss) для кожної моделі

---

## 7. Структура проєкту

```
skin-cancer-classification/
├── data/                    # HAM10000 дані (завантажуються скриптом)
├── models/                  # збережені ваги
├── results/
│   ├── comparison_table.csv
│   ├── per_class_f1.csv
│   ├── confusion_matrices/
│   ├── training_logs/
│   ├── training_curves.png
│   └── classification_reports/
├── src/
│   ├── dataset.py           # HAM10000Dataset, split logic
│   ├── models.py            # всі 5 архітектур
│   ├── train.py             # тренувальний цикл
│   ├── evaluate.py          # обчислення метрик, confusion matrices
│   └── config.py            # гіперпараметри
├── download_data.py         # скрипт завантаження HAM10000
├── run_all.py               # запуск усіх експериментів послідовно
├── requirements.txt
└── README.md
```

---

## 8. requirements.txt

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
Pillow>=9.0
```

---

## 9. Важливі нюанси для Claude Code

1. **Відтворюваність**: встановити `torch.manual_seed(42)`, `numpy.random.seed(42)`, `torch.backends.cudnn.deterministic = True` на початку.

2. **Валідація**: виділити 10% від train як validation set (також на рівні `lesion_id`). Зберігати модель з найкращим val loss.

3. **Mixed precision**: використовувати `torch.cuda.amp` для прискорення тренування на GPU.

4. **Логування**: кожну епоху виводити train loss, val loss, val accuracy. Зберігати у CSV.

5. **Час тренування**: вимірювати wall-clock час тренування кожної моделі і включити у фінальну таблицю.

6. **Кількість параметрів**: порахувати `sum(p.numel() for p in model.parameters())` і включити у таблицю.

7. **Early stopping**: якщо val loss не покращується 10 епох поспіль — зупинити тренування.

8. **Гібридна модель**: якщо виникнуть проблеми з реалізацією — допустимо спростити (наприклад, менше Transformer layers, менший d_model). Головне — зберегти ідею CNN backbone + Transformer head.

9. **Завантаження даних**: HAM10000 можна отримати:
   - Kaggle API: `kaggle datasets download -d kmader/skin-cancer-mnist-ham10000`
   - Або через прямий URL (якщо Kaggle API недоступне): `https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000` з curl
   - Файли: `HAM10000_images_part_1/`, `HAM10000_images_part_2/`, `HAM10000_metadata.csv`

---

## 10. Prompt для Claude Code

Скопіюй цей промпт і дай його Claude Code як завдання:

```
Прочитай файл experiment_spec.md у кореневій папці проєкту. Це специфікація експериментального порівняння 5 архітектур нейронних мереж (ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B/16, Hybrid CNN-Transformer) на датасеті HAM10000 (7-класова класифікація раку шкіри).

Твоя задача:
1. Створити структуру проєкту згідно зі специфікацією.
2. Реалізувати весь код: завантаження даних, dataset, моделі, тренування, оцінка.
3. Послідовно натренувати всі 5 моделей.
4. Зібрати результати у таблиці (CSV) і візуалізації (PNG).
5. Зберегти все у папку results/.

ВАЖЛИВО:
- Split даних робити на рівні lesion_id, НЕ image_id.
- Використовувати weighted cross-entropy для боротьби з дисбалансом.
- Transfer learning: заморозити backbone на 5 епох, потім розморозити.
- Зберегти модель з найкращим val loss.
- Всі деталі (гіперпараметри, аугментація, метрики) описані у spec файлі.

Специфікації комп'ютера:
- 32 гб оперативної памʼяті
 - RTX 5070TI (16 gb)
 - Ryzen 5800X
 
Шлях до файлів:
"F:/datasets/SkinCancer" - коренева папка.
GroundTruth.csv - це файл із мітками
в папці images лежать самі картинки

Почни з завантаження даних і перевірки що все працює на одній моделі, потім запусти решту.
```
