import os
import re
import io
import tempfile
import logging
import torch
from django.conf import settings
from TTS.api import TTS
from django.conf import settings
from pydub import AudioSegment

logger = logging.getLogger(__name__)

TTS_MODEL = None


def get_model():
    global TTS_MODEL
    
    if TTS_MODEL is not None:
        return TTS_MODEL
    
    try:
        logger.info("🔄 Загрузка TTS модели...")

        model_name = settings.TTS_CONFIG.get("MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        TTS_MODEL = TTS(model_name=model_name)
        TTS_MODEL.to(device)
        logger.info(f"✅ TTS модель загружена на устройство: {device}")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации TTS модели: {e}")
        raise RuntimeError("Не удалось инициализировать TTS модель")
    
    return TTS_MODEL


def split_text_into_sentences(text):
    # Разбиваем на предложения по точке, восклицательному и вопросительному знакам
    sentences = re.split(r'([.!?]+)', text)
    
    result = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if not sentence:
            continue
        
        full_sentence = sentence + punctuation
        
        # Разбиваем длинные предложения по запятым
        if ',' in sentence:
            parts = sentence.split(',')
            for j, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Добавляем запятую ко всем частям, кроме последней
                if j < len(parts) - 1:
                    part += ','
                # К последней части добавляем финальную пунктуацию
                else:
                    part += punctuation
                
                result.append(part)
        else:
            result.append(full_sentence)
    
    return result


def normalize_text_for_speech(text):
    # Словарь базовых сокращений (только самые необходимые)
    abbreviations = {
        'гг.': 'годов',
        'т.': 'тонн', 'т/год': 'тонн в год',
        'млн': 'миллионов', 'млрд': 'миллиардов', 'тыс.': 'тысяч',
        'км.': 'километров', 'км': 'километров', 'м.': 'метров',
        'руб.': 'рублей', 'РФ': 'Российской Федерации',
        'ПАО': 'Публичное акционерное общество', 'АО': 'Акционерное общество',
        'ООО': 'Общество с ограниченной ответственностью', 'НПЗ': 'нефтеперерабатывающий завод',
        'ИНН': 'И Н Н', 'ОГРН': 'О Г Р Н', 'ФКЦБ': 'Ф К Ц Б',
        'д.': 'дом', 'корп.': 'корпус', 'пом.': 'помещение', 'ул.': 'улица',
        'наб.': 'набережная', 'обл.': 'область', 'р-н': 'район',
        'пр-т': 'проспект', 'пер.': 'переулок', 'пл.': 'площадь',
        'оф.': 'офис', 'стр.': 'строение', 'эт.': 'этаж', 'ком.': 'комната',
        '°C': 'градусов Цельсия', '°F': 'градусов Фаренгейта',
    }

    def roman_to_arabic(match):
        roman = match.group(0).upper()
        
        # Игнорируем пустые строки
        if not roman or len(roman) == 0:
            return match.group(0)
        
        roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        # Если total = 0, значит это не римское число
        if total == 0:
            return match.group(0)
        
        return str(total)
    
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Удаляем маркеры списков
    text = re.sub(r'•\s*', ';', text)
    
    # Форматируем номера телефонов
    phone_placeholder = "☎PHONE{}☎"
    phone_storage = []
    
    def format_phone(match):
        phone = match.group(0)
        # Убираем все символы кроме цифр и плюса
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        if cleaned.startswith('+7') and len(cleaned) == 12:
            code = cleaned[2:5]
            first = cleaned[5:8]
            second = cleaned[8:10]
            third = cleaned[10:12]
            result = f"плюс 7 {code} {first} {second} {third}"
        elif cleaned.startswith('8') and len(cleaned) == 11:
            code = cleaned[1:4]
            first = cleaned[4:7]
            second = cleaned[7:9]
            third = cleaned[9:11]
            result = f"8 {code} {first} {second} {third}"
        else:
            # Если формат неизвестен, просто разбиваем по цифрам
            if cleaned.startswith('+'):
                result = 'плюс ' + ' '.join(list(cleaned[1:]))
            else:
                result = ' '.join(list(cleaned))
        
        # Сохраняем отформатированный телефон и возвращаем placeholder
        idx = len(phone_storage)
        phone_storage.append(result)
        return phone_placeholder.format(idx)
    
    # Обрабатываем телефонные номера
    text = re.sub(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]\d{2}[-\s]\d{2}', format_phone, text)
    text = re.sub(r'8[-\s]\d{3}[-\s]\d{3}[-\s]\d{2}[-\s]\d{2}', format_phone, text)
    

    # Сокращения из словаря
    text = re.sub(r'\bгг\.\s*', 'годов ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bг\.\s+(?=[А-ЯЁA-Z])', 'город ', text)
    for abbr, full in abbreviations.items():
        if abbr not in ['гг.']:
            pattern = r'\b' + re.escape(abbr) + r'\s*'
            text = re.sub(pattern, full + ' ', text, flags=re.IGNORECASE)
    
    # Обработка римских чисел
    text = re.sub(r'\b[IVXLCDM]+\b', roman_to_arabic, text, flags=re.IGNORECASE)
    
    # Словарь для порядковых числительных (родительный падеж для дат)
    ordinal_days = {
        '01': 'первого', '02': 'второго', '03': 'третьего', '04': 'четвёртого',
        '05': 'пятого', '06': 'шестого', '07': 'седьмого', '08': 'восьмого',
        '09': 'девятого', '10': 'десятого', '11': 'одиннадцатого', '12': 'двенадцатого',
        '13': 'тринадцатого', '14': 'четырнадцатого', '15': 'пятнадцатого',
        '16': 'шестнадцатого', '17': 'семнадцатого', '18': 'восемнадцатого',
        '19': 'девятнадцатого', '20': 'двадцатого', '21': 'двадцать первого',
        '22': 'двадцать второго', '23': 'двадцать третьего', '24': 'двадцать четвёртого',
        '25': 'двадцать пятого', '26': 'двадцать шестого', '27': 'двадцать седьмого',
        '28': 'двадцать восьмого', '29': 'двадцать девятого', '30': 'тридцатого',
        '31': 'тридцать первого'
    }
    
    ordinal_months = {
        '01': 'января', '02': 'февраля', '03': 'марта', '04': 'апреля',
        '05': 'мая', '06': 'июня', '07': 'июля', '08': 'августа',
        '09': 'сентября', '10': 'октября', '11': 'ноября', '12': 'декабря'
    }
    
    ordinal_months_no_zero = {
        '1': 'января', '2': 'февраля', '3': 'марта', '4': 'апреля',
        '5': 'мая', '6': 'июня', '7': 'июля', '8': 'августа',
        '9': 'сентября', '10': 'октября', '11': 'ноября', '12': 'декабря'
    }
    
    def day_to_ordinal(day_str):
        day = int(day_str)
        
        formatted_day = f"{day:02d}"
        return ordinal_days.get(formatted_day, day_str)
    
    def year_to_words(year_str):
        year = int(year_str)
        
        if year < 1000 or year > 9999:
            return year_str
        
        thousands = year // 1000
        hundreds = (year % 1000) // 100
        tens = (year % 100) // 10
        ones = year % 10
        
        result = []
        
        # Тысячи
        thousands_words = {
            1: 'одна тысяча', 2: 'две тысячи', 3: 'три тысячи',
            4: 'четыре тысячи', 5: 'пять тысяч', 6: 'шесть тысяч',
            7: 'семь тысяч', 8: 'восемь тысяч', 9: 'девять тысяч'
        }
        if thousands > 0:
            result.append(thousands_words[thousands])
        
        # Сотни
        hundreds_words = {
            1: 'сто', 2: 'двести', 3: 'триста', 4: 'четыреста', 5: 'пятьсот',
            6: 'шестьсот', 7: 'семьсот', 8: 'восемьсот', 9: 'девятьсот'
        }
        if hundreds > 0:
            result.append(hundreds_words[hundreds])
        
        # Десятки и единицы
        last_two = year % 100
        
        if 10 <= last_two <= 19:
            teens = {
                10: 'десятого', 11: 'одиннадцатого', 12: 'двенадцатого',
                13: 'тринадцатого', 14: 'четырнадцатого', 15: 'пятнадцатого',
                16: 'шестнадцатого', 17: 'семнадцатого', 18: 'восемнадцатого',
                19: 'девятнадцатого'
            }
            result.append(teens[last_two])
        else:
            tens_words = {
                2: 'двадцать', 3: 'тридцать', 4: 'сорок', 5: 'пятьдесят',
                6: 'шестьдесят', 7: 'семьдесят', 8: 'восемьдесят', 9: 'девяносто'
            }
            if tens > 1:
                result.append(tens_words[tens])
            
            ones_ordinal = {
                1: 'первого', 2: 'второго', 3: 'третьего', 4: 'четвёртого',
                5: 'пятого', 6: 'шестого', 7: 'седьмого', 8: 'восьмого', 9: 'девятого'
            }
            if ones > 0:
                result.append(ones_ordinal[ones])
            elif tens == 0 and hundreds == 0:
                result.append('го')
        
        return ' '.join(result)
    
    def format_date(match):
        day, month, year = match.groups()
        
        day_word = day_to_ordinal(day)
        month_word = ordinal_months.get(month) or ordinal_months_no_zero.get(month.lstrip('0'), month)
        year_word = year_to_words(year)
        
        return f", {day_word} {month_word} {year_word} года,"
    
    # Обрабатываем даты с разными форматами
    text = re.sub(r'(\d{1,2})\.(\d{1,2})\.(\d{4})\b', format_date, text)
    
    def format_short_date(match):
        day, month, year = match.groups()
        full_year = f"20{year}" if int(year) < 50 else f"19{year}"
        
        day_word = day_to_ordinal(day)
        month_word = ordinal_months.get(month) or ordinal_months_no_zero.get(month.lstrip('0'), month)
        year_word = year_to_words(full_year)
        
        return f", {day_word} {month_word} {year_word} года,"
    
    text = re.sub(r'(\d{1,2})\.(\d{1,2})\.(\d{2})(?!\d)', format_short_date, text)
    
    def format_email(match):
        email = match.group(0)
        email = email.replace('@', ' собака ')
        email = email.replace('.', ' точка ')
        return email
    
    # Форматируем email
    text = re.sub(r'\S+@\S+\.\S+', format_email, text)
    
    # Обработка процентов перед остальными числами
    text = re.sub(r'(\d+)\s*%', r'\1 процентов', text)
    
    # Обработка ДЛИННЫХ чисел БЕЗ пробелов (7+ цифр)
    long_number_placeholder = "🔢NUM{}🔢"
    long_number_storage = []
    
    def format_long_number(match):
        number_str = match.group(0)
        length = len(number_str)
        
        if length < 7:
            return number_str
        
        parts = []
        
        while len(number_str) > 0:
            if len(number_str) <= 2:
                parts.insert(0, number_str)
                break
            elif len(number_str) == 3:
                parts.insert(0, number_str)
                break
            else:
                parts.insert(0, number_str[-3:])
                number_str = number_str[:-3]
        
        result = ' '.join(parts)
        
        idx = len(long_number_storage)
        long_number_storage.append(result)
        return long_number_placeholder.format(idx)
    
    # Форматируем длинные числа БЕЗ пробелов
    text = re.sub(r'\b\d{7,}\b', format_long_number, text)
    
    def format_large_number(match):
        number_str = match.group(0).replace(' ', '')
        
        if len(number_str) > 15:
            return match.group(0)
        
        try:
            number = int(number_str)
        except ValueError:
            return match.group(0)
        
        if number >= 1000000000:
            billions = number // 1000000000
            remainder = number % 1000000000
            result = f"{billions} миллиардов"
            if remainder > 0:
                result += f" {remainder}"
            return result
        elif number >= 1000000:
            millions = number // 1000000
            remainder = number % 1000000
            result = f"{millions} миллионов"
            if remainder > 0:
                result += f" {remainder}"
            return result
        elif number >= 1000:
            thousands = number // 1000
            remainder = number % 1000
            result = f"{thousands} тысяч"
            if remainder > 0:
                result += f" {remainder}"
            return result
        
        return match.group(0)
    
    # Форматируем ТОЛЬКО числа с пробелами (минимум один пробел внутри)
    text = re.sub(r'\d+(?:\s+\d+)+', format_large_number, text)
    
    def format_decimal(match):
        whole = match.group(1)
        fractional = match.group(2)
        
        whole_words = {
            '0': 'ноль', '1': 'одна', '2': 'две', '3': 'три',
            '4': 'четыре', '5': 'пять', '6': 'шесть', '7': 'семь',
            '8': 'восемь', '9': 'девять', '10': 'десять'
        }
        
        if len(fractional) == 1:
            frac_words = {
                '1': 'одна десятая', '2': 'две десятых', '3': 'три десятых',
                '4': 'четыре десятых', '5': 'пять десятых', '6': 'шесть десятых',
                '7': 'семь десятых', '8': 'восемь десятых', '9': 'девять десятых',
            }
            return f"{whole_words.get(whole, whole)} целых {frac_words.get(fractional, fractional + ' десятых')}"
        elif len(fractional) == 2:
            frac_int = int(fractional)
            if frac_int == 1:
                return f"{whole_words.get(whole, whole)} целых одна сотая"
            elif frac_int == 2:
                return f"{whole_words.get(whole, whole)} целых две сотых"
            elif 3 <= frac_int <= 4:
                return f"{whole_words.get(whole, whole)} целых {fractional} сотых"
            elif 5 <= frac_int <= 20:
                return f"{whole_words.get(whole, whole)} целых {fractional} сотых"
            elif frac_int == 21:
                return f"{whole_words.get(whole, whole)} целых двадцать одна сотая"
            elif frac_int == 22:
                return f"{whole_words.get(whole, whole)} целых двадцать две сотых"
            else:
                return f"{whole_words.get(whole, whole)} целых {fractional} сотых"
        elif len(fractional) == 3:
            frac_int = int(fractional)
            if frac_int == 1:
                return f"{whole_words.get(whole, whole)} целых одна тысячная"
            elif frac_int == 2:
                return f"{whole_words.get(whole, whole)} целых две тысячных"
            else:
                return f"{whole_words.get(whole, whole)} целых {fractional} тысячных"
        
        return match.group(0)
    
    # Форматируем десятичные дроби с запятой
    text = re.sub(r'(\d+),(\d+)', format_decimal, text)
    
    # Заменяем номера с годом в конце
    text = re.sub(r'\bг\.\s*(?!\s*[А-ЯЁA-Z])', 'года ', text, flags=re.IGNORECASE)

    # Обработка номеров и параграфов
    text = text.replace('№', 'номер ')
    text = text.replace('§', 'параграф ')
    
    for symbol in '"«»\'`—–':
        text = text.replace(symbol, '')

    text = text.replace(';', ',')
    text = text.replace('(', ', ')
    text = text.replace(')', ', ')
    
    # Нормализуем знаки препинания
    text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)

    # Удаляем множественные запятые
    text = re.sub(r',\s*,+', ',', text)
    
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Восстанавливаем длинные числа
    for idx, num in enumerate(long_number_storage):
        text = text.replace(long_number_placeholder.format(idx), num)
    
    # Восстанавливаем телефонные номера
    for idx, phone in enumerate(phone_storage):
        text = text.replace(phone_placeholder.format(idx), phone)
    
    def replace_leading_zeros(match):
        number = match.group(0)
        if number.startswith('0'):
            result = []
            for char in number:
                if char == '0':
                    result.append('ноль')
                else:
                    remaining = number[len(result):]
                    result.append(remaining)
                    break
            else:
                return ' '.join(result)
            
            return ' '.join(result)
        return number
    
    text = re.sub(r'\b0\d*\b', replace_leading_zeros, text)
    
    return text


def text_to_speech_streaming(text):
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")
    
    # Нормализация текста
    normalized_text = normalize_text_for_speech(text)
    
    # Получение модели
    model = get_model()
    
    # Разбиваем на предложения
    sentences = split_text_into_sentences(normalized_text)
    
    # Генерируем аудио для каждого предложения
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            temp_path = tmp.name
        
        try:
            # Генерируем аудио
            reference_wav_path = settings.TTS_CONFIG.get("REFERENCE_WAV_PATH", os.path.abspath(os.path.join(settings.BASE_DIR, 'docs', 'audio.wav')))
            speaker = settings.TTS_CONFIG.get("SPEAKER", "Aaron Dreschner")
            language = settings.TTS_CONFIG.get("LANGUAGE", "ru")

            if not os.path.exists(reference_wav_path):
                model.tts_to_file(
                    text=sentence,
                    file_path=temp_path,
                    speaker=speaker,
                    language=language,
                )
            else:
                model.tts_to_file(
                    text=sentence,
                    file_path=temp_path,
                    speaker_wav=reference_wav_path,
                    language=language,
                )
            
            # Читаем аудио
            audio = AudioSegment.from_wav(temp_path)
            
            # Сохраняем обработанное аудио во временный буфер
            buffer = io.BytesIO()
            audio.export(buffer, format='wav')
            audio_data = buffer.getvalue()
            
            # Отправляем аудио данные
            yield audio_data
            
        finally:
            # Удаляем временный файл
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {e}")
