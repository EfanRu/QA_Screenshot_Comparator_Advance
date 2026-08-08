import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from config import Config
from catboost import CatBoostClassifier
import pandas as pd
from skimage.transform import resize
from skimage.measure import shannon_entropy
from skimage import measure


class ImageUtils:
    """Utility functions for image processing"""
    IMG_SIZE = (512, 256)

    @staticmethod
    def take_screenshot(driver, name):
        """Take screenshot and save with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.png"

        # Take full page screenshot
        original_size = driver.get_window_size()
        required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(required_width, required_height)

        screenshot = driver.get_screenshot_as_png()
        driver.set_window_size(original_size['width'], original_size['height'])

        return screenshot


    @staticmethod
    def take_page_screenshot_parts(driver, name, overlap_percent=10):
        """
        Take multiple screenshots of the page during scrolling and return collection of images.

        Args:
            driver: WebDriver instance
            name: base name for screenshots
            overlap_percent: percentage of overlap between screenshots (to avoid missing content at borders)

        Returns:
            List of PIL Image objects
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_parts = []

        # Получаем высоту всей страницы и видимой области
        total_height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
        viewport_height = driver.get_window_size()['height']

        # Рассчитываем перекрытие
        overlap_pixels = int(viewport_height * overlap_percent / 100)
        step_height = viewport_height - overlap_pixels

        current_position = 0
        part_number = 1

        while current_position < total_height:
            # Прокручиваем к текущей позиции
            driver.execute_script(f"window.scrollTo(0, {current_position})")

            # Ждём стабилизации страницы
            time.sleep(1.3)

            # Делаем скриншот видимой области
            screenshot_data = driver.get_screenshot_as_png()
            screenshot = Image.open(BytesIO(screenshot_data))

            # Добавляем метаданные о позиции
            screenshot.info['scroll_position'] = current_position
            screenshot.info['part_number'] = part_number
            screenshot.info['timestamp'] = timestamp
            screenshot.info['name'] = name

            screenshot_parts.append(screenshot)

            current_position += step_height
            part_number += 1

        return screenshot_parts


    @staticmethod
    def save_image(image_data, path):
        """Save image from bytes or numpy array"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(image_data, bytes):
            with open(path, 'wb') as f:
                f.write(image_data)
        elif isinstance(image_data, np.ndarray):
            cv2.imwrite(str(path), image_data)

        return path

    @staticmethod
    def load_image(path):
        """Load image from path"""
        return cv2.imread(str(path))


    @staticmethod
    def crop_to_content_with_coords(image, threshold=0.01):
        """
        Обрезает изображение до области с информацией и возвращает координаты.

        Returns:
            cropped: обрезанное изображение
            top_left: кортеж (y, x) — координаты верхнего левого угла обрезанной области
                      относительно исходного изображения (до применения padding)
        """
        mask = image > threshold

        if not mask.any():
            return np.zeros((10, 10)), (0, 0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Сохраняем координаты ДО добавления padding
        top_left_y = rmin
        top_left_x = cmin

        padding = 10
        rmin = max(0, rmin - padding)
        rmax = min(image.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image.shape[1], cmax + padding)

        cropped = image[rmin:rmax, cmin:cmax]
        top_left = (top_left_y, top_left_x)

        return cropped, top_left


    @staticmethod
    def extract_features(diff_image, top_left):
        """
        Извлекает признаки из diff-области.

        Parameters:
            diff_image: обрезанное diff-изображение
            top_left: кортеж (y, x) координат относительно исходного скриншота
        """
        features = {}

        h, w = diff_image.shape

        # 1. Геометрические признаки (bounding box)
        features['width'] = w
        features['height'] = h
        features['area_bbox'] = w * h
        features['aspect_ratio'] = w / h if h > 0 else 0
        features['relative_area'] = (w * h) / (ImageUtils.IMG_SIZE[0] * ImageUtils.IMG_SIZE[1])

        # 2. Позиционные признаки
        features['top_left_y'] = top_left[0]
        features['top_left_x'] = top_left[1]
        features['center_y_norm'] = (top_left[0] + h / 2) / ImageUtils.IMG_SIZE[1]
        features['center_x_norm'] = (top_left[1] + w / 2) / ImageUtils.IMG_SIZE[0]

        # 3. Признаки интенсивности
        features['mean_intensity'] = np.mean(diff_image)
        features['std_intensity'] = np.std(diff_image)
        features['max_intensity'] = np.max(diff_image)
        features['sum_intensity'] = np.sum(diff_image)
        features['nonzero_ratio'] = np.count_nonzero(diff_image) / diff_image.size

        # 4. Морфологические признаки (связные компоненты)
        binary = (diff_image > 0.01).astype(np.uint8)
        labeled = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labeled)

        features['num_components'] = len(regions)

        if len(regions) > 0:
            areas = [r.area for r in regions]
            features['largest_component_area'] = max(areas)
            features['largest_component_ratio'] = max(areas) / (np.count_nonzero(binary) + 1)
            features['mean_component_area'] = np.mean(areas)
            features['std_component_area'] = np.std(areas) if len(areas) > 1 else 0

            # Характеристики крупнейшей компоненты
            largest_region = max(regions, key=lambda r: r.area)
            features['largest_compactness'] = (largest_region.perimeter ** 2) / (4 * np.pi * largest_region.area + 1)
            features['largest_eccentricity'] = largest_region.eccentricity
            features['largest_solidity'] = largest_region.solidity  # area / convex_area
            features['largest_extent'] = largest_region.extent  # area / bbox_area
            features['largest_orientation'] = largest_region.orientation
        else:
            features['largest_component_area'] = 0
            features['largest_component_ratio'] = 0
            features['mean_component_area'] = 0
            features['std_component_area'] = 0
            features['largest_compactness'] = 0
            features['largest_eccentricity'] = 0
            features['largest_solidity'] = 0
            features['largest_extent'] = 0
            features['largest_orientation'] = 0

        # 5. Текстурные признаки
        # Приводим к одному размеру для текстурного анализа
        resized = resize(diff_image, (32, 32), anti_aliasing=True)

        features['entropy'] = shannon_entropy(resized + 1e-10)

        # Статистики проекций
        row_proj = np.sum(resized, axis=1)
        col_proj = np.sum(resized, axis=0)
        features['std_row_projection'] = np.std(row_proj)
        features['std_col_projection'] = np.std(col_proj)
        features['max_row_projection'] = np.max(row_proj)
        features['max_col_projection'] = np.max(col_proj)

        return features


    @staticmethod
    def create_feature_dataset(X_diff, X_coords):
        """
        Создает датафрейм с признаками для всех образцов.

        Parameters:
            X_diff: список обрезанных diff-изображений
            X_coords: список кортежей (y, x) верхнего левого угла
        """
        all_features = []

        for i, (diff_img, coords) in enumerate(zip(X_diff, X_coords)):
            features = ImageUtils.extract_features(diff_img, coords)
            features['sample_id'] = i
            all_features.append(features)

        df = pd.DataFrame(all_features)
        return df


    @staticmethod
    def compute_symmetry_score_enhanced(diff_image):
        """
        Делит diff-область пополам по вертикали и горизонтали,
        сравнивает половинки через среднюю абсолютную разницу.
        Возвращает минимальную разницу (0 = полная симметрия).
        Улучшенная версия с нормализацией и поиском симметрии с небольшим сдвигом
        """

        h, w = diff_image.shape

        # Нормализация для устойчивости
        if diff_image.max() > 0:
            diff_norm = diff_image / diff_image.max()
        else:
            return 0

        best_score = float('inf')

        # Пробуем разные точки разделения для поиска наилучшей симметрии
        for offset in range(-2, 3):  # Небольшой сдвиг
            mid_w = (w // 2) + offset
            if mid_w <= 0 or mid_w >= w:
                continue

            left = diff_norm[:, :mid_w]
            right = np.fliplr(diff_norm[:, mid_w:2 * mid_w]) if 2 * mid_w <= w else np.fliplr(
                diff_norm[:, w - mid_w:])

            # Приводим к одному размеру
            min_w = min(left.shape[1], right.shape[1])
            if min_w > 0:
                vert_diff = np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))
                best_score = min(best_score, vert_diff)

        return best_score


    @staticmethod
    def detect_duplicated_pattern(diff_image):
        """
        Ищет повторяющиеся вертикальные паттерны в diff-изображении.
        Возвращает score (0 = идеальное повторение, т.е. явный сдвиг)
        """
        h, w = diff_image.shape

        if h < 10 or w < 10:
            return 1.0

        best_score = 1.0
        best_shift = 0

        # Ищем повторяющиеся колонки/блоки
        # Пробуем разные сдвиги (половина ширины и другие)
        for shift_ratio in [0.5, 0.33, 0.25]:  # Проверяем разные соотношения
            shift = int(w * shift_ratio)
            if shift < 10:
                continue

            # Сравниваем левую часть с правой (со сдвигом)
            for start in range(0, w - shift, max(1, shift // 4)):
                left_block = diff_image[:, start:start + shift]
                right_block = diff_image[:, start + shift:start + 2 * shift]

                if right_block.shape[1] >= shift * 0.8:
                    # Обрезаем до одинакового размера
                    min_w = min(left_block.shape[1], right_block.shape[1])
                    if min_w > 5:
                        # Нормализуем
                        left_norm = left_block[:, :min_w].astype(float)
                        right_norm = right_block[:, :min_w].astype(float)

                        if left_norm.max() > 0:
                            left_norm /= left_norm.max()
                        if right_norm.max() > 0:
                            right_norm /= right_norm.max()

                        similarity = np.mean(np.abs(left_norm - right_norm))

                        if similarity < best_score:
                            best_score = similarity
                            best_shift = shift

        return best_score, best_shift


    @staticmethod
    def compare_images(img1, img2, is_ml_bug_enabled=False, threshold=30):
        """
        Compare two images using OpenCV
        Returns: (are_similar, diff_image, difference_percent, pixel_diff_count)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Check if images are the same size
        if gray1.shape != gray2.shape:
            # Resize second image to match first
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Apply threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Count different pixels
        diff_pixels = np.sum(thresh == 255)
        total_pixels = gray1.shape[0] * gray1.shape[1]
        diff_percent = (diff_pixels / total_pixels) * 100

        # Create colored diff image
        colored_diff = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        colored_diff[thresh == 255] = [0, 0, 255]  # Mark differences in red

        # Highlight differences in original image
        img1_with_diff = img1.copy()
        img1_with_diff[thresh == 255] = [0, 0, 255]  # Red color for diff

        are_similar = diff_percent <= (1 - Config.SIMILARITY_THRESHOLD) * 100

        if not are_similar and is_ml_bug_enabled:
            img = Image.fromarray(cv2.cvtColor(img1_with_diff, cv2.COLOR_BGR2RGB)).convert('L')  # конвертируем в grayscale
            img = img.resize(ImageUtils.IMG_SIZE)  # сжимаем
            img = np.array(img) / 255.0  # нормализация

            img_cropped, top_left = ImageUtils.crop_to_content_with_coords(img)

            img_features = ImageUtils.create_feature_dataset([img_cropped], [top_left])
            img_features['symmetry_score'] = ImageUtils.compute_symmetry_score_enhanced(img_cropped)
            best_score, best_shift = ImageUtils.detect_duplicated_pattern(img_cropped)
            img_features['duplicated_pattern_best_score'] = best_score
            img_features['duplicated_pattern_best_shift'] = best_shift

            loaded_model = CatBoostClassifier()
            loaded_model.load_model('catboost_model.cbm')
            new_predictions = loaded_model.predict(img_features)
            print(f"Вызов ML модели для детекции багов! Результат анализа: {new_predictions}")
            are_similar = new_predictions == 0

        return are_similar, img1_with_diff, diff_percent, diff_pixels


    @staticmethod
    def get_screenshot_hash(image):
        """Generate perceptual hash for image"""
        # Resize to 8x8 for simple hashing
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Calculate mean and create hash
        mean = np.mean(gray)
        hash_string = ''.join(['1' if pixel > mean else '0' for row in gray for pixel in row])

        return hash_string