"""
Visits all pages under Concert API docs to webscrape information using the Chrome driver. 
Stores each page as a chunk into ChromaDB Concert collection.
"""

import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------- CONFIG ----------
prefix = "https://developer.ibm.com/apis/catalog/concert--ibm-concert-api/api/API--concert--ibm-concert-api-v1-1-0#"
# suffixes = ["post_login", "delete_apikey", "get_apikey"]

suffixes = [
    "post_login",
    "delete_apikey",
    "get_apikey",
    "post_apikey",
    "list_users_and_groups",
    "list_all_applications",
    "create_applications",
    "delete_application",
    "get_application_details",
    "update_applications",
    "list_applications_vulnerability_details",
    "get_application_package_trends",
    "list_environment_certificates",
    "get_certificate_details",
    "get_certificate_issuers",
    "update_certificates_data",
    "delete_certificates_data",
    "list_postures",
    "get_posture_by_id",
    "create_postures_update",
    "create_posture",
    "list_catalogs",
    "create_catalog",
    "delete_catalog",
    "get_catalog_by_id",
    "import_catalogs",
    "list_all_profiles",
    "create_profile",
    "delete_profile",
    "get_profile_by_id",
    "replace_profile",
    "import_profiles",
    "get_all_risks",
    "upsert_risk",
    "delete_risk",
    "create_delete_component_definitions",
    "import_component_definitions",
    "upload_component_definitions",
    "list_concert_workflows",
    "create_components",
    "list_concertdef_sboms",
    "validate_concertdef_sbom",
    "concertdef_sbom_schema",
    "delete_credentials",
    "list_credentials",
    "create_credential",
    "get_credential",
    "update_credential",
    "validate_credential",
    "delete_entity_access_users",
    "list_entity_access_users",
    "update_entity_access_users",
    "post_entity_access_users",
    "delete_entity_settings",
    "get_entity_settings",
    "update_entity_settings",
    "create_entity_settings",
    "list_environments",
    "create_environment",
    "delete_environment",
    "get_environment",
    "update_environment",
    "list_runtimes",
    "get_runtime",
    "get_el_object",
    "add_el_object",
    "create_applications_export",
    "download_export_jobs",
    "create_environments_export",
    "export_el_object",
    "export_el_files",
    "create_export_catalog",
    "create_export_profile",
    "create_export_assessment",
    "delete_handler_open",
    "get_one_handler_open",
    "create_handler_open",
    "replace_handler_description_open",
    "replace_handler_protocol_open",
    "replace_handler_hostname_open",
    "replace_handler_port_open",
    "replace_handler_prefix_open",
    "replace_handler_auth_open",
    "replace_handler_default_headers_open",
    "list_handlers_open",
    "delete_contract_open",
    "get_contract_information_open",
    "create_contract_open",
    "replace_contract_open",
    "ingest_upload_files",
    "ingest_load_data",
    "import_files",
    "list_ingestion_jobs",
    "create_ingestion_job",
    "delete_ingestion_job",
    "get_ingestion_job",
    "update_ingestion_job",
    "collect_data",
    "list_images",
    "list_repositories",
    "list_instance_settings",
    "update_instance_setting",
    "list_build_artifacts",
    "create_build_artifacts",
    "delete_build_artifact",
    "get_artifact",
    "update_build_artifact",
    "get_artifact_packages",
    "get_artifact_cves",
    "get_artifact_blast_radius",
    "list_build_artifacts_vulnerability_details",
    "list_build_artifacts_history",
    "list_build_artifact_package_trends",
    "list_source_repos",
    "create_source_repos",
    "delete_source_repos",
    "get_source_repository_details",
    "update_source_repos",
    "get_source_repo_blast_radius",
    "list_sourcerepo_history",
    "list_sourcerepo_package_trends",
    "list_accesspoints",
    "get_accesspoint_details",
    "get_accesspoint_blast_radius",
    "create_manual_assessments",
    "create_manual_assessment_update",
    "create_job",
    "get_job_by_id",
    "update_cancel_job",
    "list_packages",
    "list_packages_vulnerability_details",
    "list_recommendations",
    "get_recommendation",
    "get_sca_package_highlights",
    "get_sca_sbom_highlights",
    "get_sca_package",
    "list_cyclonedx_sboms",
    "delete_sbom",
    "get_cyclonedx_sbom",
    "get_sbom_blast_radius",
    "get_sbom_openssf_scorecard",
    "get_package_blast_radius",
    "get_package_associations",
    "list_sbom_history",
    "get_tocs_lz_filters",
    "search_tocs_lz",
    "highlight_tocs_lz",
    "search_tocs_el",
    "upload_evidence",
    "archive_record",
    "list_all_tags",
    "create_tags",
    "delete_tag",
    "update_tags",
    "get_user_preferences",
    "replace_user_preferences",
    "get_table_preferences",
    "replace_table_preferences",
    "list_vulnerabilities",
    "get_vulnerability",
    "update_vulnerabilities",
    "list_vulnerability_scans",
    "list_package_vulnerabilities",
    "list_dast_and_sast_exposures",
    "get_vulnerability_blast_radius",
    "get_exposure",
    "list_vulnerability_assessments",
    "update_vulnerability_assessment",
    "get_vulnerability_assessment_by_id",
    "update_cve_assessment",
    "list_all_vulnerability_assessments",
    "delete_vulnerabilities",
    "delete_exposures",
    "create_remediation_action",
    "update_decision_remediation_action",
    "update_remediation_action",
    "list_remediation_actions",
    "get_remediation_action",
    "search_remediation_actions",
    "update_exposure_assessment",
    "get_exposure_trends",
    "get_vulnerability_trends",
    "delete_automations",
    "list_automations",
    "create_automation",
    "get_automation",
    "update_automation_rule",
    "ticket_create",
    "get_labels",
    "triggerEvaluation",
    "refreshPackageData"
]

DEBUG = False  # Set to False to store in ChromaDB

chroma_path = "/data/wikidata/chromadb" # "concert_docs/chroma_db" 
collection_name = "concert_api"
embedding_fn = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

# ---------- SETUP ----------
if not DEBUG:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

# Initialize Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

def wait_for_main_content(driver):
    """Wait for the main content container to load."""
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div[3]/div[2]/div[2]/div/div[2]/div/div/div[1]/div"))
        )
        time.sleep(2)
        return True
    except Exception:
        return False

def extract_main_content(driver):
    """Extract main page content."""
    try:
        container = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[2]/div[2]/div/div[2]/div/div/div[1]/div")
        html_content = container.get_attribute("innerHTML")
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        clean_text = ' '.join(clean_text.split()).strip()
        return clean_text
    except Exception:
        return None

def find_and_expand_arrows(driver, container, max_depth=10, current_depth=0):
    """Recursively find and expand arrow elements to extract hidden content."""
    if current_depth >= max_depth:
        return []
    
    extracted_contents = []
    
    try:
        arrow_selector = ".//div[contains(@class, 'arrow') and contains(@class, 'arrow-text') and not(contains(@class, 'arrow-down'))]"
        arrows = container.find_elements(By.XPATH, arrow_selector)
        
        for arrow in arrows:
            try:
                current_classes = arrow.get_attribute("class")
                if "arrow-down" in current_classes:
                    continue
                
                arrow_text = arrow.text.strip()
                
                # Find parent element
                try:
                    parent_element = arrow.find_element(By.XPATH, "./ancestor::li[1]")
                except NoSuchElementException:
                    try:
                        parent_element = arrow.find_element(By.XPATH, "./ancestor::div[contains(@class, 'parameter')][1]")
                    except NoSuchElementException:
                        parent_element = arrow.find_element(By.XPATH, "./..")
                
                initial_text_length = len(parent_element.text)
                
                # Click arrow
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", arrow)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", arrow)
                time.sleep(1.5)
                
                # Check if expanded
                updated_classes = arrow.get_attribute("class")
                if "arrow-down" in updated_classes:
                    updated_text_length = len(parent_element.text)
                    content_growth = updated_text_length - initial_text_length
                    
                    if content_growth > 10:
                        updated_content = parent_element.get_attribute("innerHTML")
                        clean_content = re.sub(r'<[^>]+>', ' ', updated_content)
                        clean_content = ' '.join(clean_content.split()).strip()
                        
                        content_info = {
                            'arrow_text': arrow_text,
                            'content': clean_content
                        }
                        extracted_contents.append(content_info)
                    
                    # Recursively process nested arrows
                    time.sleep(1)
                    nested_contents = find_and_expand_arrows(driver, parent_element, max_depth, current_depth + 1)
                    extracted_contents.extend(nested_contents)
                
            except Exception:
                continue
                
    except Exception:
        pass
    
    return extracted_contents

def find_target_section(driver):
    """Find the specific section containing expandable arrows."""
    try:
        return driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[2]/div[2]/div/div[2]/div/div/div[1]/div[1]/div[2]/ul")
    except NoSuchElementException:
        return None

# ---------- MAIN PROCESSING ----------

for i, suffix in enumerate(suffixes):
    url = prefix + suffix
    print(f"\nProcessing: {url} ({suffix})")
    
    try:
        # Navigate to page
        driver.get("about:blank")
        driver.get(url)
        time.sleep(2)
        
        if not wait_for_main_content(driver):
            continue
        
        # Extract main content
        main_content = extract_main_content(driver)
        if not main_content:
            continue
        
        # Extract arrow-expanded content
        expanded_content = []
        target_section = find_target_section(driver)
        if target_section:
            expanded_content = find_and_expand_arrows(driver, target_section)
        
        # Combine all content
        combined_content = f"MAIN CONTENT:\n{main_content}\n\n"
        
        if expanded_content:
            combined_content += "EXPANDED CONTENT:\n"
            for content_info in expanded_content:
                combined_content += f"\n--- {content_info['arrow_text']} ---\n"
                combined_content += content_info['content'] + "\n"
        
        # Output extracted content
        print("Extracted content:")
        print(combined_content)
        print("-" * 80)
        
        # Store in ChromaDB or debug
        if DEBUG:
            # Just print for debugging
            pass
        else:
            chunk_id = f"{i}_{suffix}"
            collection.upsert(
                documents=[combined_content],
                ids=[chunk_id],
                metadatas=[{
                    "source": url,
                    "endpoint": suffix
                }]
            )
        
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Clean up
driver.quit()

if not DEBUG:
    print(f"\n✅ Indexed {len(suffixes)} API endpoints into ChromaDB.")
else:
    print(f"\n✅ Debug mode: Processed {len(suffixes)} API endpoints")



## TEST THE CREATED CHROMADB 

# import chromadb
# from chromadb.config import Settings

# chroma_path = "/data/wikidata/chromadb" 
# collection_name = "concert_api"
# client = chromadb.PersistentClient(path=chroma_path)
# collection = client.get_collection(name=collection_name)
# collections = client.list_collections()
# print("\n🗂 Collections in ChromaDB:")
# for col in collections:
#     print(f"- {col.name}")

# # Load your specific collection (e.g., "turbonomic")
# collection = client.get_collection("concert_api") 

# # TO PRINT ALL ENTRIES
# print(f"\n📦 Collection '{collection.name}' has {collection.count()} documents.\n")
# results = collection.get(limit=3)
# print(results)

# TO DELETE A SPECIFIC ENTRY
# bad_id = "032e0fd1d8031a86ec7298652bc62e6b"
# # Delete the entry
# collection.delete(ids=[bad_id])
# print(f"✅ Deleted entry with ID: {bad_id}")

# TO DELETE THE COLLECTION
# client.delete_collection(name="concert_api")