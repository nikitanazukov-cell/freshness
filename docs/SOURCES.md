# Официальные источники и происхождение компонентов

Проверено при подготовке проекта: 13 ноября 2026 года. Используются не обязательно самые новые версии, а явно согласованная пара PyTorch/torchvision и описанные в requirements диапазоны остальных пакетов.

1. PyTorch: совместимые версии 2.10.0 / torchvision 0.25.0 и CPU-установка:
   https://pytorch.org/get-started/previous-versions/
2. Torchvision: MobileNetV3-Small, ImageNet-веса, нормализация, число параметров и размер исходного файла:
   https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
3. Исходная реализация MobileNetV3 в torchvision:
   https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
4. Albumentations 2.0.8, описание релиза:
   https://pypi.org/project/albumentations/2.0.8/
5. Albumentations: воспроизводимость и собственный seed Compose:
   https://albumentations.ai/docs/faq/
6. CPython 3.13: установка Windows, pip, launcher, Tcl/Tk:
   https://docs.python.org/3.13/using/windows.html
7. Tkinter:
   https://docs.python.org/3/library/tkinter.html
8. Карточка набора из задания:
   https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
9. Запись автора набора в Zenodo:
   https://zenodo.org/records/4788775

Фото из внешнего набора и веса ImageNet в архиве проекта не распространяются. Пользователь самостоятельно проверяет условия использования источников и соблюдает лицензии установленных библиотек. Внешние материалы не следует автоматически считать свободными от ограничений только потому, что они доступны для скачивания.
