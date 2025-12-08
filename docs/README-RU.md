[English](../README.md) | [Русский](README-RU.md)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)

![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-FF6F00?style=for-the-badge&logo=google&logoColor=white)

![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)

![NumPy](https://img.shields.io/badge/NumPy-<2.0-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Active-FF6B6B?style=for-the-badge&logo=opencv&logoColor=white)

</div>

# VisionPlay Board

> Интерактивное приложение для игр на электронной доске с использованием компьютерного зрения, которое позволяет создавать игры, управляемые жестами, используя OpenCV и детекцию позы MediaPipe.

## Описание

VisionPlay Board - это приложение, которое использует компьютерное зрение для создания интерактивных игр. Приложение работает в полноэкранном режиме OpenCV и позволяет активировать игры как кликом мыши, так и детекцией человека через MediaPipe.

## Возможности

- **Полноэкранный режим** с OpenCV
- **Упрощенная архитектура** с 2 слоями: фон и скелет
- **Многопоточная обработка** детекции позы
- **Детекция человека** через MediaPipe с постоянной отрисовкой скелета
- **Зеркальное отображение** камеры для естественного взаимодействия
- **Интерактивные плитки игр** на главном экране
- **Автоматический запуск** случайной игры при детекции человека (3 секунды)
- **Игра "Skeleton Viewer"** - просмотр скелета до 3 человек
- **Мониторинг производительности** с FPS и статистикой

## Установка

1. Убедитесь, что у вас установлен Python 3.11.11 (через pyenv):
   ```bash
   pyenv install 3.11.11
   pyenv local 3.11.11
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск

### Через pyenv (рекомендуется):
```bash
# Убедитесь, что pyenv настроен
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

# Установите Python 3.11.11 если не установлен
pyenv install 3.11.11
pyenv shell 3.11.11

# Установите зависимости
pip install -r requirements.txt

# Запустите приложение
python main.py
```

### Через скрипты запуска:
```bash
# Простой запуск (рекомендуется)
./scripts/start.sh

# Или полный скрипт с установкой
./scripts/run.sh
```

## Управление

- **ESC** - выход из приложения или возврат в главное меню
- **Q** - выход из приложения
- **S** - показать статистику производительности
- **Клик мыши** - активация игры на плитке
- **Детекция человека** - автоматический запуск случайной игры через 3 секунды

## Конфигурация

Настройки приложения находятся в файле `.env`:

### Основные настройки:
- `CAMERA_INDEX` - индекс камеры (по умолчанию 0)
- `CAMERA_WIDTH/HEIGHT` - разрешение камеры
- `FULLSCREEN_MODE` - полноэкранный режим (true/false)
- `HUMAN_DETECTION_TIMEOUT` - время детекции человека для автозапуска

### Настройки детекции позы:
- `SHOW_BODY_LANDMARKS` - показывать точки тела
- `SHOW_FACE_LANDMARKS` - показывать точки лица
- `SHOW_HAND_LANDMARKS` - показывать точки рук
- `SHOW_POSE_CONNECTIONS` - показывать соединения тела
- `SHOW_FACE_CONNECTIONS` - показывать соединения лица
- `SHOW_HAND_CONNECTIONS` - показывать соединения рук

### Настройки производительности:
- `SHOW_FPS` - показывать FPS на экране
- `SHOW_STATISTICS_ON_S_KEY` - показывать статистику по клавише S
- `ENABLE_POSE_DETECTION` - включить/выключить детекцию позы (false для лучшей производительности)

### Настройки игр:
- `MAX_PEOPLE_IN_FRAME` - максимальное количество людей в кадре (до 3)

## Структура проекта

```
VisionPlay-Board/
├── main.py                 # Точка входа
├── requirements.txt        # Зависимости
├── README.md              # Документация
├── .env                   # Конфигурация
├── .python-version        # Версия Python
├── scripts/               # Скрипты запуска
│   ├── start.sh          # Быстрый запуск
│   └── run.sh            # Полный запуск с установкой
├── tests/                 # Тестовые файлы
│   ├── test.py           # Тест производительности камеры
│   ├── minimal_test.py   # Минимальный тест камеры
│   ├── simple_test.py    # Простой тест камеры
│   └── camera_check.py   # Проверка камеры
├── docs/                  # Документация
│   └── camera_report.md  # Отчет о камере
├── models/                # Модели машинного обучения
│   └── yolov8n.pt        # YOLO модель детекции людей
└── src/                   # Исходный код приложения
    ├── app.py             # Основное приложение
    ├── utils/
    │   ├── config.py      # Управление конфигурацией
    │   ├── pose_detector.py # Детекция позы MediaPipe
    │   ├── layers.py      # Система слоев для отрисовки
    │   ├── thread_manager.py # Многопоточная обработка
    │   ├── scaling.py     # Адаптивное масштабирование
    │   ├── yolo_person_detector.py # YOLO детектор людей
    │   └── yolo_holistic_detector.py # YOLO + MediaPipe детектор
    └── games/
        ├── base_game.py   # Базовый класс игры
        ├── skeleton_viewer_game.py # Игра "Skeleton Viewer"
        └── hide_and_seek_game.py # Игра "Hide and Seek"
```

## Архитектура приложения

### Упрощенная система отрисовки:
1. **Background Layer** - камера + UI элементы
2. **Skeleton Layer** - отрисовка детекции позы и скелета

### Многопоточная обработка:
- **Main Thread** - захват кадров с камеры и отрисовка
- **Pose Detection Thread** - обработка MediaPipe в отдельном потоке

### Преимущества упрощенной архитектуры:
- **Улучшенная производительность** - меньше слоев, быстрее отрисовка
- **Простота** - легко понимать и модифицировать
- **Мониторинг** - отслеживание FPS и статистики
- **Адаптивность** - автоматическая настройка производительности

## Игра "Skeleton Viewer"

### Возможности:
- **Просмотр скелета** до 3 человек одновременно
- **Детекция тела, лица и рук** с разными цветами для каждого человека
- **Автоматический выход** если нет людей в кадре 10 секунд
- **Адаптивная производительность** в зависимости от скорости движения
- **Настройка отображения** через `.env` файл

### Отображение:
- **Зеленый скелет** - первое тело
- **Красный скелет** - второе тело  
- **Синий скелет** - третье тело
- **Цветные точки** для лица и рук каждого человека
- **Счетчик людей** в кадре
- **Предупреждения** о выходе из игры

## Требования

- Python 3.11.11
- OpenCV 4.9.x
- MediaPipe 0.10.x
- NumPy < 2.0
- Веб-камера

## Устранение неполадок

### Проблемы с зависимостями:
Если возникают конфликты версий, переустановите зависимости:
```bash
pip install "opencv-python<4.10" "numpy<2" mediapipe python-dotenv
```

### Проблемы с камерой:
- Проверьте индекс камеры в `.env` файле
- Убедитесь, что камера не используется другими приложениями
- Попробуйте изменить `CAMERA_INDEX` на 1, 2 и т.д.

### Проблемы с pyenv:
Убедитесь, что pyenv настроен правильно:
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### Проблемы с правами доступа:
Если возникают ошибки с правами доступа к `.python-version`, используйте:
```bash
# Вместо pyenv local используйте pyenv shell
pyenv shell 3.11.11
python main.py
```

### Ошибки MediaPipe:
Если возникают ошибки с типами MediaPipe, убедитесь, что используется совместимая версия:
```bash
pip install mediapipe==0.10.21
```

### Проблемы с производительностью:
Если изображение лагает, попробуйте следующие решения:

1. **Отключите детекцию позы:**
   ```bash
   ENABLE_POSE_DETECTION=false
   ```

2. **Уменьшите разрешение камеры:**
   ```bash
   CAMERA_WIDTH=640
   CAMERA_HEIGHT=480
   ```

3. **Настройте FPS (если нужно):**
   ```bash
   CAMERA_FPS=30  # Для 640x480
   CAMERA_FPS=15  # Для 1280x720
   ```

4. **Диагностика камеры:**
   ```bash
   # Простой тест камеры
   python tests/test.py
   
   # Минимальный тест
   python tests/minimal_test.py
   
   # Диагностика всех камер
   python tests/simple_test.py
   
   # Проверка камеры
   python tests/camera_check.py
   ```

5. **Проверьте поддержку MJPEG:**
   Убедитесь, что `USE_MJPEG_CODEC=true` в `.env` файле.

### Проблемы с Wayland:
Если видите предупреждение "Ignoring XDG_SESSION_TYPE=wayland", это означает, что система использует Wayland вместо X11. Исправление уже включено в код, но можно также установить переменную окружения:
```bash
export QT_QPA_PLATFORM=xcb
```


