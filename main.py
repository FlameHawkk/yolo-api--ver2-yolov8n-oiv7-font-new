from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image, ImageDraw, ImageFont
import json
import csv
import os
from datetime import datetime

# Создаем экземпляр FastAPI приложения
app = FastAPI(title="YOLO API Service")

# Настройка CORS (Cross-Origin Resource Sharing) для работы с фронтендом и мобильными приложениями
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# КОНФИГУРАЦИЯ АННОТАЦИЙ - ВСЕ ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ ЗДЕСЬ
ANNOTATION_CONFIG = {
    # Коэффициенты для расчета толщины рамки
    'line_thickness_base': 5,           # Базовая толщина для изображения высотой 800px
    'line_thickness_min': 2,            # Минимальная толщина
    'line_thickness_max': 8,            # Максимальная толщина
    
    # Коэффициенты для расчета размера шрифта
    'font_size_base': 30,               # Базовый размер для изображения высотой 800px
    'font_size_min': 15,                # Минимальный размер шрифта
    'font_size_max': 60,                # Максимальный размер шрифта
    
    # Коэффициенты для отступов текста
    'text_padding': 2,                  # Отступ текста от краев подложки
    'text_offset': 1,                   # Отступ текста от bounding box
    
    # Порог яркости для выбора цвета текста
    'brightness_threshold': 128,        # Если яркость > 128 - черный текст, иначе белый
}

# Глобальные переменные
current_model = None # Модель
translation_dict = {} # Словарь переводов
model_config = {} # Конфигурация
current_font = None # Шрифт

def load_model_config():
    """
    Загрузка конфигурации модели из JSON файла model_config.json
    Функция читает настройки и сохраняет их в глобальную переменную model_config
    """
    global model_config
    try:
        # Открываем и читаем JSON файл с конфигурацией
        with open('model_config.json', 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        print(f"Конфигурация модели загружена: {model_config}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки конфигурации модели: {e}")
        return False

def load_translations(translate_name):
    """
    Загрузка переводов классов из CSV файла
    
    Args:
        translate_name (str): Имя файла с переводами (например, "OpenImagesV7.csv")
    
    Returns:
        bool: True если загрузка успешна, False в случае ошибки
    """
    global translation_dict
    try:
        # Формируем путь к файлу переводов в папке translations
        translation_file = f'translations/{translate_name}'
        translation_dict = {}
        
        # Открываем CSV файл и читаем построчно
        with open(translation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Извлекаем данные из строки CSV
                english_name = row['english']
                russian_name = row['russian']
                
                # Создаем запись в словаре переводов
                # Ключ - английское название, значение - словарь с переводом и номером класса
                translation_dict[english_name] = {
                    'russian': russian_name,
                    'class_number': int(row['class_number'])
                }
        
        print(f"Переводы загружены из файла: {translate_name}")
        print(f"Всего классов в словаре переводов: {len(translation_dict)}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки переводов: {e}")
        return False

def load_font(font_file_name):
    """Загрузка шрифта из папки fonts"""
    global current_font
    
    if not font_file_name:
        print("❌ Имя файла шрифта не указано в конфиге")
        return None
    
    font_paths = [
        f"fonts/{font_file_name}",
        f"./fonts/{font_file_name}",
        font_file_name,
        f"/opt/render/project/src/fonts/{font_file_name}"
    ]
    
    base_font_size = ANNOTATION_CONFIG['font_size_base']
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, base_font_size)
                current_font = font
                print(f"✅ Шрифт загружен: {font_path}")
                return font
            else:
                print(f"⚠️ Файл шрифта не найден: {font_path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки шрифта {font_path}: {e}")
    
    fallback_fonts = [
        "arial.ttf", "arialbd.ttf", "DejaVuSans.ttf", 
        "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf"
    ]
    
    for font_name in fallback_fonts:
        try:
            font = ImageFont.truetype(font_name, base_font_size)
            current_font = font
            print(f"✅ Используем fallback шрифт: {font_name}")
            return font
        except:
            continue
    
    try:
        font = ImageFont.load_default()
        current_font = font
        print("⚠️ Используем стандартный шрифт")
        return font
    except Exception as e:
        print(f"❌ Не удалось загрузить ни один шрифт: {e}")
        return None

def load_model():
    """
    Загрузка модели YOLO из папки models
    
    Returns:
        bool: True если загрузка успешна, False в случае ошибки
    """
    global current_model
    try:
        # Формируем полный путь к файлу модели
        model_path = f'models/{model_config["model_name"]}'
        
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Загружаем модель с помощью Ultralytics
        current_model = YOLO(model_path)
        # Перемещаем модель на CPU
        current_model.to('cpu')
        print(f"Модель успешно загружена: {model_config['model_name']}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False

def initialize_app():
    """
    Основная функция инициализации приложения
    Выполняет загрузку конфигурации, модели и переводов
    
    Returns:
        bool: True если все компоненты загружены успешно
    """

    # Загружаем конфигурацию модели
    if not load_model_config():
        print("❌ Ошибка: Не удалось загрузить конфигурацию модели")
        return False
    
    # Загружаем модель YOLO
    if not load_model():
        print("❌ Ошибка: Не удалось загрузить модель")
        return False
    
    # Загружаем переводы классов
    if not load_translations(model_config["translate_name"]):
        print("❌ Ошибка: Не удалось загрузить переводы")
        return False
    
    # Загружаем шрифт
    font_file = model_config.get("font_file")
    if font_file:
        load_font(font_file)
    else:
        print("⚠️ Файл шрифта не указан в конфигурации")
    
    # Успех
    print("✅ Все компоненты приложения успешно инициализированы")
    return True

def get_label_translation(label, language):
    """
    Получение перевода метки класса на указанный язык
    
    Args:
        label (str): Исходная метка на английском языке
        lang (str): Язык для перевода ('en' или 'ru')
    
    Returns:
        str: Переведенная метка на выбранном языке
    """

    # Если запрошен английский или метки нет в словаре, возвращаем оригинал
    if language == 'en' or label not in translation_dict:
        return label
    
    # Если запрошен русский и перевод есть, возвращаем русскую версию
    if language == 'ru':
        return translation_dict[label]['russian']
    
    # Для неподдерживаемых языков возвращаем английскую метку
    return label


def get_color_for_class(class_id):
    """
    Генерирует цвет для класса на основе его ID
    Расширенная палитра с 40 цветами, похожими на оригинальные YOLO
    """
    colors = [
        # Основные яркие цвета (первые 10)
        (255, 0, 0),      # Красный
        (0, 255, 0),      # Зеленый
        (0, 0, 255),      # Синий
        (255, 255, 0),    # Желтый
        (255, 0, 255),    # Пурпурный
        (0, 255, 255),    # Голубой
        (255, 128, 0),    # Оранжевый
        (128, 255, 0),    # Лаймовый
        (0, 128, 255),    # Голубой (темнее)
        (255, 0, 128),    # Розовый
        
        # Дополнительные цвета (11-20)
        (128, 0, 255),    # Фиолетовый
        (0, 255, 128),    # Весенний зеленый
        (255, 128, 128),  # Светло-красный
        (128, 255, 128),  # Светло-зеленый
        (128, 128, 255),  # Светло-синий
        (255, 255, 128),  # Светло-желтый
        (255, 128, 255),  # Светло-пурпурный
        (128, 255, 255),  # Светло-голубой
        (192, 192, 192),  # Серебряный
        (128, 128, 128),  # Серый
        
        # Теплые цвета (21-30)
        (255, 165, 0),    # Ярко-оранжевый
        (255, 140, 0),    # Темно-оранжевый
        (255, 99, 71),    # Томатный
        (255, 69, 0),     # Красно-оранжевый
        (255, 215, 0),    # Золотой
        (218, 165, 32),   # Золотистый
        (210, 105, 30),   # Шоколадный
        (139, 69, 19),    # Седло-коричневый
        (160, 82, 45),    # Сиена
        (205, 133, 63),   # Перу
        
        # Холодные цвета (31-40)
        (70, 130, 180),   # Стальной синий
        (100, 149, 237),  # Васильковый
        (30, 144, 255),   # Синий Доджер
        (0, 191, 255),    # Глубокий небесно-голубой
        (72, 209, 204),   # Средний бирюзовый
        (32, 178, 170),   # Светло-морской
        (0, 139, 139),    # Темный бирюзовый
        (0, 128, 128),    # Бирюзовый
        (47, 79, 79),     # Темный аспидно-серый
        (95, 158, 160),   # Кадетский синий
    ]
    return colors[class_id % len(colors)]

def get_contrast_text_color(background_color):
    """
    ОПРЕДЕЛЕНИЕ КОНТРАСТНОГО ЦВЕТА ТЕКСТА
    
    ПРИНЦИП РАБОТЫ:
    1. Берем цвет фона (подложки) в формате (R, G, B)
    2. Вычисляем яркость фона по формуле восприятия человеческим глазом:
       Яркость = 0.299*R + 0.587*G + 0.114*B
    3. Сравниваем яркость с пороговым значением (по умолчанию 128)
    4. Если яркость > порога - фон СВЕТЛЫЙ, используем ЧЕРНЫЙ текст
       Если яркость < порога - фон ТЕМНЫЙ, используем БЕЛЫЙ текст
    
    Эта формула учитывает, что человеческий глаз по-разному воспринимает цвета:
    - Наиболее чувствителен к зеленому (коэффициент 0.587)
    - Менее чувствителен к красному (коэффициент 0.299)  
    - Наименее чувствителен к синему (коэффициент 0.114)
    """
    r, g, b = background_color
    
    # Формула расчета относительной яркости (стандарт W3C для доступности)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
    
    # Получаем порог из конфигурации
    threshold = ANNOTATION_CONFIG['brightness_threshold']
    
    # Выбираем цвет текста на основе яркости фона
    if brightness > threshold:
        return (4, 28, 85)  # Черный (темный) текст для светлого фона
    else:
        return (255, 255, 255)  # Белый текст для темного фона

def calculate_font_size(image_height):
    """
    Вычисляет размер шрифта на основе высоты изображения
    с использованием конфигурационных параметров
    """
    config = ANNOTATION_CONFIG
    base_height = 800
    
    font_size = max(int(config['font_size_base'] * (image_height / base_height)), 
                    int(config['font_size_min']))  # Приводим min к int
    font_size = min(font_size, int(config['font_size_max']))  # Приводим max к int
    
    return font_size

def calculate_line_thickness(image_height):
    """
    Вычисляет толщину линий на основе высоты изображения
    с использованием конфигурационных параметров
    """
    config = ANNOTATION_CONFIG
    base_height = 800
    
    thickness = max(int(config['line_thickness_base'] * (image_height / base_height)), 
                    int(config['line_thickness_min']))  # Приводим min к int
    thickness = min(thickness, int(config['line_thickness_max']))  # Приводим max к int
    
    return thickness

def create_custom_annotated_image(image, results, detections, language):
    """
    Создание аннотированного изображения с переведенными метками
    """
    # Получаем конфигурационные параметры
    config = ANNOTATION_CONFIG
    
    # ШАГ 1: ПОДГОТОВКА ИЗОБРАЖЕНИЯ
    # Конвертируем numpy array в PIL Image
    pil_image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(pil_image)
    
    # Получаем размеры изображения для масштабирования
    image_width, image_height = pil_image.size
    
    # ШАГ 2: НАСТРОЙКА ШРИФТА И ПАРАМЕТРОВ
    # Вычисляем
    font_size = int(calculate_font_size(image_height))  # Приводим к int
    line_thickness = int(calculate_line_thickness(image_height))  # Приводим к int
    padding = config['text_padding']
    text_offset = config['text_offset']
    
    # Загружаем шрифт с правильным размером
    font = None
    if current_font:
        try:
            font_path = getattr(current_font, 'path', None)
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # ШАГ 3: ОБРАБОТКА КАЖДОГО BOUNDING BOX
    boxes = results[0].boxes
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            # Координаты bounding box (приводим к int)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf) # Уверенность
            class_id = int(box.cls) # ID класса
            
            # Получаем переведенную метку из наших детекций
            if i < len(detections):
                display_label = detections[i]['label']
            else:
                # Используем оригинальную метку
                original_label = current_model.names[class_id]
                display_label = get_label_translation(original_label, language)
            
            # Формируем текст для отображения
            label_text = f"{display_label} {confidence:.2f}"
            
            # Получаем цвет и контрастный текст
            box_color = get_color_for_class(class_id)
            text_color = get_contrast_text_color(box_color)
            
            # Рисуем bounding box с настроенной толщиной
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=box_color, width=line_thickness)
            
            # Расчет размера текста
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label_text) * font_size // 2
                text_height = font_size
            
            # Размеры подложки с учетом отступов
            total_text_width = text_width + padding * 2
            total_text_height = text_height + padding * 2
            
            # УМНОЕ РАЗМЕЩЕНИЕ ТЕКСТА
            if y1 - total_text_height - text_offset >= 0:
                # Если места сверху достаточно - над bounding box
                text_x = x1 + padding
                text_y = y1 - text_height - padding - text_offset
                background_rect = [
                    int(x1), 
                    int(y1 - total_text_height - text_offset), 
                    int(x1 + total_text_width), 
                    int(y1)
                ]
            else:
                # Еси места сверху нет - внутри bounding box
                text_x = x1 + padding
                text_y = y1 + padding
                background_rect = [
                    int(x1), 
                    int(y1), 
                    int(x1 + total_text_width), 
                    int(y1 + total_text_height)
                ]
            
            # Защита от выхода за правую границу
            if background_rect[2] > image_width:
                overflow = background_rect[2] - image_width
                background_rect[0] = max(0, background_rect[0] - overflow)
                background_rect[2] = image_width
                text_x = background_rect[0] + padding
            
            # Рисуем подложку и текст
            draw.rectangle(background_rect, fill=box_color)
            draw.text((int(text_x), int(text_y)), label_text, fill=text_color, font=font)
    
    return np.array(pil_image)

@app.on_event("startup")
async def startup_event():
    """
    Событие, выполняемое при запуске сервера
    Инициализирует все необходимые компоненты приложения
    """
    print("🚀 Запуск YOLO API сервера...")
    
    # Выполняем инициализацию приложения
    if initialize_app():
        print("✅ Сервер успешно запущен")
        print(f"📁 Используемая модель: {model_config['model_name']}")
        print(f"📄 Файл переводов: {model_config['translate_name']}")
        print(f"🔤 Загружено переводов: {len(translation_dict)} классов")
        print(f"🔠 Используемый шрифт: {model_config.get('font_file', 'не указан')}")
    else:
        print("❌ Не удалось инициализировать приложение")
        # Прерываем запуск сервера при ошибке инициализации
        raise RuntimeError("Не удалось инициализировать приложение")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    language: str = Form("en")
):
    """
    Основной endpoint для выполнения предсказания на изображении
    
    Args:
        file: Загружаемое изображение (обязательный параметр)
        confidence: Порог уверенности для детекции (по умолчанию 0.5)
        language: Язык возвращаемых меток ('en' или 'ru', по умолчанию 'en')
    
    Returns:
        dict: Результаты детекции с переведенными метками
    """
    try:
        print(f"🎯 Начало обработки запроса: confidence={confidence}, language={language}")
        
        # Проверяем, что модель загружена
        if current_model is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        # Проверяем корректность указанного языка
        if language not in ['en', 'ru']:
            raise HTTPException(
                status_code=400, 
                detail="Неподдерживаемый язык. Используйте 'en' или 'ru'"
            )

        # Проверяем корректность порога уверенности (от 0 до 1) 
        if confidence < 0 or confidence > 1:
            raise HTTPException(
                status_code=400,
                detail="Порог уверенности должен быть между 0 и 1"
            )
        
        # Проверяем, что загружен файл изображения
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        # Читаем данные изображения из запроса
        image_data = await file.read()
        file_size = len(image_data)
        print(f"📁 Получено изображение: {file.filename}, размер: {file_size} байт")
        
        # Открываем изображение с помощью PIL
        image = Image.open(io.BytesIO(image_data))

        # Конвертируем в RGB если нужно (для PNG с альфа-каналом)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
            print("🔄 Конвертирован в RGB")
      
        image_array = np.array(image)
        print(f"🖼️ Размер изображения: {image_array.shape}")
        
        # Выполняем предсказание с помощью YOLO модели
        print(f"🔍 Выполнение предсказания YOLO с уверенностью {confidence}...")
        # примечание: используем встроенную фильтрацию YOLO       
        results = current_model(image_array, conf=confidence, verbose=True)
        
        print(f"📊 YOLO обнаружено результатов: {len(results)}")
        
        # Обрабатываем результаты (YOLO уже отфильтровал по confidence)
        detections = []
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                print(f"📦 Результат: {len(boxes)} боксов")
                for j, box in enumerate(boxes):
                    box_confidence = float(box.conf)
                    class_id = int(box.cls)
                    original_label = current_model.names[class_id]
                    
                    # Получаем перевод названия класса на запрошенный язык
                    translated_label = get_label_translation(original_label, language)
                    
                    print(f"  🏷️ Бокс {j}: {original_label} -> {translated_label} (ID: {class_id}), уверенность: {box_confidence:.3f}")
                    
                    # Формируем информацию о детекции
                    detection = {
                        'label': translated_label,     # Переведенная метка
                        'label_en': original_label,    # Оригинальная английская метка
                        'confidence': box_confidence,  # Уверенность предсказания
                        'bbox': box.xyxy[0].tolist(),  # Координаты bounding box [x1, y1, x2, y2]
                        'class_id': class_id           # ID класса
                    }
                    detections.append(detection)
            else:
                print(f"❌ Результат {i}: нет боксов")
        
        print(f"✅ Обработано детекций: {len(detections)}")
        
        # Сортируем по уверенности (от высокой к низкой)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Создаем аннотированное изображение с переведенными метками
        print("🖌️ Создаем аннотированное изображение с переведенными метками...")
        annotated_image = create_custom_annotated_image(
            image_array, results, detections, language
        )
        
        # Конвертируем изображение в base64 для передачи в ответе
        annotated_pil = Image.fromarray(annotated_image)
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG", quality=95)
        
        import base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"🎉 Успешно завершено. Возвращаем {len(detections)} детекций")
        
        # Формируем и возвращаем ответ
        return {
            "success": True,
            "detections": detections,
            "annotated_image": image_base64,
            "model_used": model_config["model_name"],
            "translate_file": model_config["translate_name"],
            "language": language,
            "confidence_threshold": confidence,
            "total_detections": len(detections),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Обрабатываем ошибки
        print(f"❌ Критическая ошибка предсказания: {str(e)}")
        import traceback
        print(f"🔍 Трассировка ошибки: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """
    Endpoint для проверки работоспособности сервера
    Используется для мониторинга и проверки состояния API
    """

    # Определяем статус сервера на основе загрузки модели
    status = "healthy" if current_model is not None else "degraded"
    
    return {
        "status": status,
        "current_model": model_config.get("model_name", "none"),
        "translate_file": model_config.get("translate_name", "none"),
        "translations_loaded": len(translation_dict),
        "font_file": model_config.get("font_file", "none"),
        "timestamp": datetime.now().isoformat()
    }

@app.api_route("/model", methods=["GET", "HEAD"])
async def list_model():
    """
    Endpoint для получения информации о текущей загруженной модели
    """
    return {
        "current_model": model_config.get("model_name", "none")        
    }

@app.api_route("/config", methods=["GET", "HEAD"])
async def get_config():
    """Endpoint для получения текущей конфигурации сервера"""
    return {
        "model_config": model_config,
        "translate_file": model_config.get("translate_name", "none"),
        "translations_loaded": len(translation_dict),
        "font": model_config.get("font_file", "none")
    }

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Корневой endpoint с основной информацией о API"""
    return {
        "message": "YOLO Object Detection API",
        "version": "1.4.0",
        "endpoints": {
            "/predict/": "POST - выполнить детекцию объектов на изображении",
            "/health": "GET - проверить состояние сервера", 
            "/model": "GET - информация о текущей модели",
            "/config": "GET - текущая конфигурация"
        }
    }
