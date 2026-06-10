import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from config import Config


class ImageUtils:
    """Utility functions for image processing"""

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
    def compare_images(img1, img2, threshold=30):
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