import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent

    # Screenshot paths
    SCREENSHOTS_DIR = BASE_DIR / 'screenshots'
    ACTUAL_DIR = SCREENSHOTS_DIR / 'actual'
    EXPECTED_DIR = SCREENSHOTS_DIR / 'expected'
    DIFF_DIR = BASE_DIR / 'reports' / 'diffs'

    # Browser settings
    BROWSER = 'chrome'
    HEADLESS = False
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1080
    WEBDRIVER_PATH = '/home/slava/Documents/chrome_driver/chromedriver'

    # Image comparison settings
    SIMILARITY_THRESHOLD = 0.95  # 95% similarity required
    PIXEL_DIFF_THRESHOLD = 0  # Count pixels with difference > 0

    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.ACTUAL_DIR, cls.EXPECTED_DIR, cls.DIFF_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)