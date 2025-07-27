"""
Prints all URLs to visit in the Concert API docs - use these to navigate required URLs to fetch information
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- CONFIG ----------
BASE_URL = "https://developer.ibm.com/apis/catalog/concert--ibm-concert-api/api/API--concert--ibm-concert-api-v1-1-0"
DOMAIN_PREFIX = "https://developer.ibm.com/apis/catalog/concert--ibm-concert-api/"

# Sidebar navigation selector - the main ul container with all the divs
SIDEBAR_SELECTOR = "#root > div > div.overview-page.hide-scrollbar.grid-width > div.flex-grid.cds--grid > div.custom-grid-1 > div > div.side-nav-column.pr-0.cds--col-sm-4.cds--col-md-2.cds--col-lg-3.cds--col-xlg-3.cds--col-max-3 > div.side-nav > nav > ul > div:nth-child(3) > li > ul"

# ---------- SETUP ----------
os.makedirs("concert_api_docs/meta", exist_ok=True)

# Initialize Chrome with options
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")  # Use new headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

visited = set()
all_links = set()

# ---------- HELPERS ----------

def extract_sidebar_links(driver, base_url):
    """Extract all navigation links from the sidebar"""
    links = set()
    
    # Wait for the sidebar to load
    wait = WebDriverWait(driver, 10)
    sidebar = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, SIDEBAR_SELECTOR)))
    
    # Find all div children under the main ul
    divs = sidebar.find_elements(By.XPATH, "./div")
    print(f"Found {len(divs)} main navigation sections")
    
    for i, div in enumerate(divs, 1):
        try:
            # Look for nested ul > li > ul structure within each div
            nested_uls = div.find_elements(By.XPATH, ".//ul")
            for ul in nested_uls:
                # Get all anchor tags within this ul
                anchors = ul.find_elements(By.TAG_NAME, "a")
                for anchor in anchors:
                    href = anchor.get_attribute("href")
                    if href and href.startswith(DOMAIN_PREFIX):
                        links.add(href)
                        print(f"Found link: {href}")
                    elif href and href.startswith("#"):
                        # Handle fragment-only URLs
                        full_url = base_url + href
                        links.add(full_url)
                        print(f"Found fragment link: {full_url}")
        except Exception as e:
            print(f"Error processing div {i}: {e}")
            continue
    
    return list(links)

# ---------- MAIN EXECUTION ----------

try:
    # First, visit the base URL and extract all sidebar links
    print(f"Visiting base URL: {BASE_URL}")
    driver.get(BASE_URL)
    time.sleep(3)  # Wait for page to load completely
    
    # Extract all navigation links from sidebar
    all_links = extract_sidebar_links(driver, BASE_URL)
    print(f"\nFound {len(all_links)} total links")
    
except Exception as e:
    print(f"[CRITICAL ERROR]: {e}")

finally:
    driver.quit()