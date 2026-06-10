import os
import shutil
from pathlib import Path

import cv2
from config import Config
from utils.image_utils import ImageUtils


class ScreenshotComparator:
    """Main screenshot comparison engine"""


    def __init__(self):
        Config.create_dirs()
        self.results = []

    def compare_screenshots(self, actual_image, expected_name, test_name):
        """Compare actual screenshot with expected one"""

        # Save actual screenshot
        actual_path = Config.ACTUAL_DIR / f"{test_name}_{expected_name}_actual.png"
        ImageUtils.save_image(actual_image, actual_path)

        # Check if expected screenshot exists
        expected_path = Config.EXPECTED_DIR / f"{expected_name}.png"

        if not expected_path.exists():
            # Save as new baseline
            ImageUtils.save_image(actual_image, expected_path)
            result = {
                'test': test_name,
                'status': 'BASELINE_CREATED',
                'message': f'New baseline created: {expected_name}',
                'diff_percent': 0,
                'actual': actual_path,
                'expected': expected_path,
                'diff': None
            }
            self.results.append(result)
            return result

        # Load expected screenshot
        expected_image = ImageUtils.load_image(expected_path)

        # Compare images
        if actual_image.shape != expected_image.shape:
            expected_image = cv2.resize(expected_image,
                                        (actual_image.shape[1], actual_image.shape[0]))

        are_similar, diff_image, diff_percent, diff_pixels = ImageUtils.compare_images(
            actual_image, expected_image
        )

        # Save diff image if not similar
        diff_path = None
        if not are_similar:
            diff_name = f"{test_name}_{expected_name}_diff.png"
            diff_path = Config.DIFF_DIR / diff_name
            ImageUtils.save_image(diff_image, diff_path)

        result = {
            'test': test_name,
            'status': 'PASS' if are_similar else 'FAIL',
            'message': f'Similarity: {100 - diff_percent:.2f}%',
            'diff_percent': round(diff_percent, 2),
            'diff_pixels': diff_pixels,
            'actual': actual_path,
            'expected': expected_path,
            'diff': diff_path
        }

        self.results.append(result)
        return result


    def compare_and_sort_screenshots_for_thesis(self):
        """
        Сравнивает скриншоты попарно и сортирует их по папкам.

        Args:
            actual_dir: путь к папке с скриншотами без задержки
            expected_dir: путь к папке со скриншотами с задержкой 1 сек
            compare_true_dir: папка для совпадающих скриншотов
            compare_false_dir: папка для различающихся скриншотов
            diff_dir: папка для изображений разницы
        """
        expected_dir = Config.THESIS_COMPARE_ACTUAL_DIR
        actual_dir = Config.THESIS_COMPARE_ACTUAL_DIR_WITH_DELAY
        compare_true_dir = Config.THESIS_COMPARE_TRUE
        compare_false_dir = Config.THESIS_COMPARE_FALSE
        diff_dir = Config.THESIS_COMPARE_DIFF

        # Создаём необходимые папки
        Path(compare_true_dir).mkdir(parents=True, exist_ok=True)
        Path(compare_false_dir).mkdir(parents=True, exist_ok=True)
        Path(diff_dir).mkdir(parents=True, exist_ok=True)

        # Получаем списки файлов
        actual_files = sorted([f for f in os.listdir(actual_dir) if f.endswith('.png')])
        expected_files = sorted([f for f in os.listdir(expected_dir) if f.endswith('.png')])

        results = []

        for expected_file in expected_files:
            # Формируем имя соответствующего файла в папке с задержкой
            base_name = expected_file.replace('_expected_', '').split('_')[0]
            actual_file = expected_file.replace('_expected_', '_expected_with_1_sec_wait_')

            if actual_file not in actual_files:
                print(f"WARNING: Expected file not found: {actual_file}")
                continue

            # Полные пути к файлам
            actual_path = Path(actual_dir) / actual_file
            expected_path = Path(expected_dir) / expected_file

            # Загружаем изображения
            actual_image = ImageUtils.load_image(actual_path)
            expected_image = ImageUtils.load_image(expected_path)

            test_name = base_name
            expected_name = actual_file.replace('.png', '')

            # Сравниваем изображения
            are_similar, diff_image, diff_percent, diff_pixels = ImageUtils.compare_images(
                actual_image, expected_image
            )

            # Сохраняем изображение разницы, если не похожи
            diff_path = None
            if not are_similar:
                diff_name = f"{test_name}_{expected_name}_diff.png"
                diff_path = Path(diff_dir) / diff_name
                ImageUtils.save_image(diff_image, diff_path)

            # Определяем целевую папку и копируем файлы
            if are_similar:
                target_dir = Path(compare_true_dir)
            else:
                target_dir = Path(compare_false_dir)

            # Копируем оба скриншота в соответствующую папку
            shutil.copy2(actual_path, target_dir / expected_file)
            shutil.copy2(expected_path, target_dir / actual_file)

            # Формируем результат
            result = {
                'test': test_name,
                'status': 'PASS' if are_similar else 'FAIL',
                'message': f'Similarity: {100 - diff_percent:.2f}%',
                'diff_percent': round(diff_percent, 2),
                'diff_pixels': diff_pixels,
                'actual': str(actual_path),
                'expected': str(expected_path),
                'diff': str(diff_path) if diff_path else None
            }
            results.append(result)

        return results


    def compare_screenshots_from_paths(self, expected_dir, actual_dir, diff_dir):
        """
        Сравнивает скриншоты попарно и сортирует их по папкам.

        Args:
            actual_dir: путь к папке с скриншотами без задержки
            expected_dir: путь к папке со скриншотами с задержкой 1 сек
            diff_dir: папка для изображений разницы
        """

        # Создаём необходимые папки
        Path(diff_dir).mkdir(parents=True, exist_ok=True)

        # Получаем списки файлов
        expected_files = sorted([f for f in os.listdir(expected_dir) if f.endswith('.png')])

        results = []

        for expected_file in expected_files:
            # Полные пути к файлам
            actual_path = Path(actual_dir) / expected_file
            expected_path = Path(expected_dir) / expected_file

            # Загружаем изображения
            actual_image = ImageUtils.load_image(actual_path)
            expected_image = ImageUtils.load_image(expected_path)

            # Сравниваем изображения
            are_similar, diff_image, diff_percent, diff_pixels = ImageUtils.compare_images(
                actual_image, expected_image
            )

            print(f"Изображения {expected_path} одинаковы {are_similar} процент расхождения {diff_percent}")

            # Сохраняем изображение разницы, если не похожи
            diff_path = None
            if not are_similar:
                diff_name = f"{expected_file}_diff.png"
                diff_path = Path(diff_dir) / diff_name
                ImageUtils.save_image(diff_image, diff_path)

            # Формируем результат
            result = {
                'test': expected_file,
                'status': 'PASS' if are_similar else 'FAIL',
                'message': f'Similarity: {100 - diff_percent:.2f}%',
                'diff_percent': round(diff_percent, 2),
                'diff_pixels': diff_pixels,
                'actual': str(actual_path),
                'expected': str(expected_path),
                'diff': str(diff_path) if diff_path else None
            }
            results.append(result)

        return results


    def generate_report(self):
        """Generate HTML report"""
        report_path = Config.BASE_DIR / 'reports' / 'screenshot_test_report.html'

        html = """
        <html>
        <head>
            <title>Screenshot Comparison Report</title>
            <style>
                body { font-family: Arial; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                .pass { background-color: #dff0d8; }
                .fail { background-color: #f2dede; }
                .baseline { background-color: #d9edf7; }
                img { max-width: 200px; max-height: 150px; }
            </style>
        </head>
        <body>
            <h1>Screenshot Comparison Test Report</h1>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>Diff %</th>
                    <th>Actual</th>
                    <th>Expected</th>
                    <th>Diff</th>
                </tr>
        """

        for result in self.results:
            status_class = {
                'PASS': 'pass',
                'FAIL': 'fail',
                'BASELINE_CREATED': 'baseline'
            }.get(result['status'], '')

            diff_display = f'<a href="{result["diff"]}">View Diff</a>' if result['diff'] else 'N/A'

            html += f"""
                <tr class="{status_class}">
                    <td>{result['test']}</td>
                    <td>{result['status']}</td>
                    <td>{result['message']}</td>
                    <td>{result.get('diff_percent', 'N/A')}%</td>
                    <td><a href="{result['actual']}">View</a></td>
                    <td><a href="{result['expected']}">View</a></td>
                    <td>{diff_display}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Report generated: {report_path}")
        return report_path