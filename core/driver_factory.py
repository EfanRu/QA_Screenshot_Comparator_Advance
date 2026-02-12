from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from config import Config


class DriverFactory:
    """Factory for creating WebDriver instances"""

    @staticmethod
    def get_driver():
        if Config.BROWSER.lower() == 'chrome':
            return DriverFactory._create_chrome_driver()
        else:
            raise ValueError(f"Unsupported browser: {Config.BROWSER}")

    @staticmethod
    def _create_chrome_driver():
        chrome_options = Options()

        if Config.HEADLESS:
            chrome_options.add_argument('--headless=new')

        chrome_options.add_argument(f'--window-size={Config.WINDOW_WIDTH},{Config.WINDOW_HEIGHT}')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-extensions')

        # Performance optimizations
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_argument('--hide-scrollbars')

        # service = Service(ChromeDriverManager().install())
        driver_path = Config.WEBDRIVER_PATH
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        return driver