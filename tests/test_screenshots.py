import pytest

from config import Config
from core.driver_factory import DriverFactory
from utils.image_utils import ImageUtils
from core.comparator import ScreenshotComparator
import cv2
import numpy as np


class TestScreenshots:
    """Test suite for screenshot comparison"""

    @pytest.fixture(scope="function")
    def setup(self):
        """Setup test environment"""
        self.driver = DriverFactory.get_driver()
        self.comparator = ScreenshotComparator()
        yield
        self.driver.quit()

    @pytest.mark.skip
    def test_google_homepage(self, setup):
        """Test Google homepage screenshot"""
        self.driver.get("https://www.google.com")

        # Take screenshot
        screenshot_data = ImageUtils.take_screenshot(self.driver, "google_homepage")
        screenshot = cv2.imdecode(
            np.frombuffer(screenshot_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Compare with baseline
        result = self.comparator.compare_screenshots(
            screenshot,
            "google_homepage",
            "test_google_homepage"
        )

        assert result['status'] in ['PASS', 'BASELINE_CREATED'], \
            f"Screenshot comparison failed: {result['message']}"

    @pytest.mark.skip
    def test_github_homepage(self, setup):
        """Test GitHub homepage screenshot"""
        self.driver.get("https://github.com")

        # Take screenshot
        screenshot_data = ImageUtils.take_screenshot(self.driver, "github_homepage")
        screenshot = cv2.imdecode(
            np.frombuffer(screenshot_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Compare with baseline
        result = self.comparator.compare_screenshots(
            screenshot,
            "github_homepage",
            "test_github_homepage"
        )

        assert result['status'] in ['PASS', 'BASELINE_CREATED'], \
            f"Screenshot comparison failed: {result['message']}"


    def test_save_pages(self, setup):
        """Test for save pages screenshot for data set"""

        # Compare with baseline
        self.save_page("https://netology.ru", "netology_main")
        self.save_page("https://netology.ru/degree", "netology_degree")
        self.save_page("https://netology.ru/programs/sysadmin", "netology_sysadmin")
        self.save_page("https://netology.ru/programs/developer1c_ultimate", "netology_developer1c_ultimate")
        self.save_page("https://netology.ru/programs/graphic-design-ultimate", "netology_graphic-design-ultimate")
        self.save_page("https://netology.ru/programs/dizajner-intererov", "netology_dizajner-intererov")
        self.save_page("https://netology.ru/programs/specialist-po-iskusstvennomu-intellektu", "netology_specialist-po-iskusstvennomu-intellektu")
        self.save_page("https://netology.ru/programs/automation-engineer", "netology_automation-engineer")
        self.save_page("https://netology.ru/programs/python", "netology_python")
        self.save_page("https://netology.ru/programs/designer-communication", "netology_designer-communication")
        self.save_page("https://netology.ru/programs/fullstack-devops", "netology_fullstack-devops")
        self.save_page("https://netology.ru/programs/qa-middle", "netology_qa-middle")
        self.save_page("https://netology.ru/programs/informationsecurity", "netology_informationsecurity")
        self.save_page("https://netology.ru/programs/1c-analitik-s-nulya-do-middle", "netology_1c-analitik-s-nulya-do-middle")
        self.save_page("https://netology.ru/programs/fullstack-python-dev", "netology_fullstack-python-dev")
        self.save_page("https://netology.ru/programs/accountant", "netology_accountant")
        self.save_page("https://netology.ru/programs/analytics-dwh", "netology_analytics-dwh")
        self.save_page("https://netology.ru/programs/professiya-menezher-marketplejsov", "netology_professiya-menezher-marketplejsov")
        self.save_page("https://netology.ru/programs/hr-manager", "netology_hr-manager")
        self.save_page("https://netology.ru/programs/menedger-marketplace", "netology_menedger-marketplace")
        self.save_page("https://netology.ru/programs/nutritionist", "netology_nutritionist")
        self.save_page("https://netology.ru/programs/data_analyst_ultimate", "netology_data_analyst_ultimate")
        self.save_page("https://netology.ru/programs/product-ultimate", "netology_product-ultimate")
        self.save_page("https://netology.ru/programs/web-designer", "netology_web-designer")
        self.save_page("https://netology.ru/programs/landshaftnyj-dizajner-ultimate", "netology_landshaftnyj-dizajner-ultimate")
        self.save_page("https://netology.ru/programs/illustration-ultimate", "netology_illustration-ultimate")
        self.save_page("https://netology.ru/programs/menedzher-avito", "netology_menedzher-avito")
        self.save_page("https://netology.ru/programs/product-design", "netology_product-design")
        self.save_page("https://netology.ru/programs/devops", "netology_devops")
        self.save_page("https://netology.ru/programs/professiya-psykholog", "netology_professiya-psykholog")
        self.save_page("https://netology.ru/programs/distance-course-internet-marketing", "netology_distance-course-internet-marketing")
        self.save_page("https://netology.ru/programs/senior-internet-marketer", "netology_senior-internet-marketer")
        self.save_page("https://netology.ru/programs/prodatascience", "netology_prodatascience")
        self.save_page("https://netology.ru/programs/biohaking", "netology_biohaking")
        self.save_page("https://netology.ru/programs/network-engineer", "netology_network-engineer")
        self.save_page("https://netology.ru/programs/scenarnoye-masterstvo", "netology_")
        self.save_page("https://netology.ru/programs/professiya_financial_analyst", "netology_")
        self.save_page("https://netology.ru/programs/metodolog-obrazovatelnyh-programm-sinhron", "netology_")
        self.save_page("https://netology.ru/programs/qa", "netology_")
        self.save_page("https://netology.ru/programs/gamedesigner", "netology_")
        self.save_page("https://netology.ru/programs/java-developer", "netology_")
        self.save_page("https://netology.ru/programs/unity-developer", "netology_")
        self.save_page("https://netology.ru/programs/target-smm-full", "netology_")
        self.save_page("https://netology.ru/programs/developer1c", "netology_")
        self.save_page("https://netology.ru/programs/data-scientist", "netology_")
        self.save_page("https://netology.ru/programs/dizayn_sredy", "netology_")
        self.save_page("https://netology.ru/programs/video-editing", "netology_")
        self.save_page("https://netology.ru/programs/systems-analyst", "netology_")
        self.save_page("https://netology.ru/programs/veb-razrabotchik-s-nulya-professiya-s-vyborom-specializacii", "netology_")
        self.save_page("https://netology.ru/programs/business-analytics-online", "netology_")
        self.save_page("https://netology.ru/programs/sales-manager-online", "netology_")


    def save_page(self, url, name):
        self.driver.get(url)

        # Take screenshot
        screenshot_data = ImageUtils.take_screenshot(self.driver, name)
        screenshot = cv2.imdecode(
            np.frombuffer(screenshot_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Save actual screenshot
        path = Config.ACTUAL_DIR / f"{name}_actual.png"
        ImageUtils.save_image(screenshot, path)


    @pytest.fixture(scope="session", autouse=True)
    def generate_final_report(self, request):
        """Generate report after all tests"""
        yield
        # Get the test instance and generate report
        test_instance = None
        for item in request.session.items:
            if hasattr(item, 'instance') and hasattr(item.instance, 'comparator'):
                test_instance = item.instance
                break

        if test_instance and hasattr(test_instance, 'comparator'):
            test_instance.comparator.generate_report()
