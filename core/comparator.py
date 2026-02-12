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