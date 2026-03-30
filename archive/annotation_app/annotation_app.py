"""
annotation_app.py

Streamlit application for annotating schema mapping predictions.
Collects human judgments on whether predicted mappings are correct.
Shows agreement across methods to help annotators.
"""

import streamlit as st
import json
import os
import hashlib
import re
import secrets
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="Schema Mapping",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .mapping-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .source-field {
        color: #0066cc;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .target-field {
        color: #28a745;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .method-tag {
        background-color: #6c757d;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-right: 5px;
        display: inline-block;
    }
    .agreement-high {
        background-color: #28a745;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .agreement-medium {
        background-color: #ffc107;
        color: black;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .agreement-low {
        background-color: #dc3545;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .annotation-row {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .api-name {
        font-size: 1.3em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        color: #333;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Data directory paths
DATA_DIR = Path("data")
RESULTS_DIRS = {
    "map_reflect": Path("map_reflect/results"),
    "anytool": Path("anytool/results"),
    "react": Path("react/results"),
    "smt": Path("smt/results"),
    "gnn": Path("gnn/results"),
    "rematch": Path("rematch/results"),
    "match_maker": Path("match_maker/results"),
}
ANNOTATIONS_FILE = DATA_DIR / "mapping_annotations.json"
USERS_FILE = DATA_DIR / "annotation_users.json"
PROPERTY_LOOKUP_FILE = DATA_DIR / "itopsgraph_api_lookup.json"


# ============== User Management ==============

def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> Dict:
    """Load users from file."""
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {"users": {}}


def save_users(users: Dict):
    """Save users to file."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, email: str, display_name: str = "") -> Tuple[bool, str]:
    """Register a new user."""
    users = load_users()

    if username.lower() in [u.lower() for u in users.get("users", {}).keys()]:
        return False, "Username already exists"

    if not username or not password:
        return False, "Username and password are required"

    if len(password) < 4:
        return False, "Password must be at least 4 characters"

    users["users"][username] = {
        "password_hash": hash_password(password),
        "email": email,
        "display_name": display_name or username,
        "created_at": datetime.now().isoformat(),
    }

    save_users(users)
    return True, "Account created successfully"


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user."""
    users = load_users()
    user = users.get("users", {}).get(username)

    if user and user.get("password_hash") == hash_password(password):
        return {
            "username": username,
            "email": user.get("email", ""),
            "display_name": user.get("display_name", username),
        }
    return None


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user info by username."""
    users = load_users()
    user = users.get("users", {}).get(username)
    if user:
        return {
            "username": username,
            "email": user.get("email", ""),
            "display_name": user.get("display_name", username),
        }
    return None


def reset_user_password(username: str, new_password: str) -> Tuple[bool, str]:
    """Reset a user's password."""
    users = load_users()
    if username not in users.get("users", {}):
        return False, "User not found"

    users["users"][username]["password_hash"] = hash_password(new_password)
    save_users(users)
    return True, "Password reset successfully"


def generate_temp_password(length: int = 12) -> str:
    """Generate a random temporary password."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def send_password_reset_email(to_email: str, username: str, new_password: str) -> Tuple[bool, str]:
    """
    Send password reset email.

    Requires SMTP settings in environment or .env:
    - SMTP_SERVER: SMTP server address
    - SMTP_PORT: SMTP port (default 25 for internal relays, 587 for external)
    - SMTP_USERNAME: SMTP username (optional for internal relays)
    - SMTP_PASSWORD: SMTP password (optional for internal relays)
    - SMTP_FROM_EMAIL: From email address
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "25"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL")

    if not smtp_server or not from_email:
        return False, "Email not configured. Please contact administrator."

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = "Schema Mapping - Password Reset"

        body = f"""
Hello {username},

Your password has been reset. Here is your new temporary password:

    {new_password}

Please log in and change your password as soon as possible.

If you did not request this password reset, please contact the administrator.

Best regards,
Schema Mapping Team
"""
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # Only use TLS and auth if credentials are provided
            if smtp_username and smtp_password:
                server.starttls()
                server.login(smtp_username, smtp_password)
            server.send_message(msg)

        return True, "Password reset email sent successfully"
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False, f"Failed to send email: {str(e)}"


# ============== Data Loading ==============

def load_property_lookup() -> Dict:
    """Load property lookup for field descriptions."""
    if PROPERTY_LOOKUP_FILE.exists():
        with open(PROPERTY_LOOKUP_FILE) as f:
            return json.load(f)
    return {}


def load_schema_descriptions() -> Dict[str, Dict]:
    """Load type and field descriptions from GraphQL schema files."""
    schema_dir = DATA_DIR / "itopsgraph_graphql"
    descriptions = {}

    if not schema_dir.exists():
        return descriptions

    for schema_file in schema_dir.glob("*.graphql"):
        try:
            content = schema_file.read_text()
            # Parse the schema file name to get product_type
            filename = schema_file.stem  # e.g., "turbonomic_Action"

            # Extract type description (triple-quoted string before "type")
            type_desc = ""
            type_desc_match = re.search(r'"""([^"]+)"""\s*type\s+\w+', content)
            if type_desc_match:
                type_desc = type_desc_match.group(1).strip()

            # Extract field descriptions
            fields = {}
            # Match field descriptions: """description""" followed by field_name:
            field_pattern = r'"""([^"]+)"""\s*(\w+):'
            for match in re.finditer(field_pattern, content):
                field_desc = match.group(1).strip()
                field_name = match.group(2).strip()
                fields[field_name] = field_desc

            # Also extract fields without descriptions
            field_only_pattern = r'^\s+(\w+):\s*(\[?\w+\]?!?)'
            for match in re.finditer(field_only_pattern, content, re.MULTILINE):
                field_name = match.group(1).strip()
                if field_name not in fields:
                    fields[field_name] = ""

            descriptions[filename] = {
                "type_description": type_desc,
                "fields": fields,
            }

        except Exception:
            continue

    return descriptions


def get_field_info(field_id: str, lookup: Dict) -> Dict:
    """Get field information from lookup."""
    if field_id in lookup:
        return lookup[field_id]
    return {"description": "No description available", "field_type": "unknown"}


def get_api_info(api_name: str, lookup: Dict, schema_descs: Dict = None) -> Dict:
    """Get API-level information by aggregating field info.

    Args:
        api_name: API name like 'turbonomic.Action' or 'turbonomic_Action'
        lookup: Property lookup dictionary
        schema_descs: Schema descriptions from GraphQL files

    Returns:
        Dict with type_description, product_name, type_name, and list of fields
    """
    # Normalize the API name (could be product.Type or product_Type)
    normalized = api_name.replace(".", "_")

    # Find all fields for this API from lookup
    fields = []
    type_description = ""
    product_name = ""
    type_name = ""

    for key, info in lookup.items():
        if key.startswith(normalized + "."):
            fields.append({
                "name": info.get("field_name", ""),
                "type": info.get("field_type", ""),
                "description": info.get("description", ""),
                "is_required": info.get("is_required", False),
            })
            if not type_description and info.get("type_description"):
                type_description = info["type_description"]
            if not product_name:
                product_name = info.get("product_name", "")
            if not type_name:
                type_name = info.get("type_name", "")

    # If no fields found, try to extract from the field mappings directly
    if not fields:
        parts = normalized.split("_")
        if len(parts) >= 2:
            type_part = "_".join(parts[1:])
            for key, info in lookup.items():
                key_type = info.get("type_name", "")
                if key_type == type_part:
                    fields.append({
                        "name": info.get("field_name", ""),
                        "type": info.get("field_type", ""),
                        "description": info.get("description", ""),
                        "is_required": info.get("is_required", False),
                    })
                    if not type_description and info.get("type_description"):
                        type_description = info["type_description"]
                    if not product_name:
                        product_name = info.get("product_name", "")
                    if not type_name:
                        type_name = info.get("type_name", "")

    # Try to get description from schema files if not found in lookup
    if schema_descs and normalized in schema_descs:
        schema_info = schema_descs[normalized]
        if not type_description and schema_info.get("type_description"):
            type_description = schema_info["type_description"]
        # Add field descriptions from schema if missing
        schema_fields = schema_info.get("fields", {})
        if not fields and schema_fields:
            fields = [{"name": name, "type": "", "description": desc, "is_required": False}
                     for name, desc in schema_fields.items()]
        elif fields:
            # Enhance existing fields with schema descriptions
            for f in fields:
                if not f.get("description") and f["name"] in schema_fields:
                    f["description"] = schema_fields[f["name"]]

    return {
        "api_name": api_name,
        "product_name": product_name,
        "type_name": type_name,
        "type_description": type_description,
        "fields": fields,
        "field_count": len(fields),
    }


def load_all_mappings() -> Tuple[List[Dict], Dict[str, List[str]]]:
    """Load all mapping predictions from result files.

    Only loads ITOpsGraph-related files.
    """
    all_mappings = []
    seen_keys = set()
    agreement_tracker = defaultdict(list)

    for method_name, results_dir in RESULTS_DIRS.items():
        if not results_dir.exists():
            continue

        for result_file in results_dir.glob("*.json"):
            if "checkpoint" in result_file.name or "annotations" in result_file.name:
                continue
            # Skip dual_context files
            if "dual_context" in result_file.name:
                continue
            # Only load ITOpsGraph-related files
            filename_lower = result_file.name.lower()
            if "itops" not in filename_lower:
                continue

            try:
                with open(result_file) as f:
                    data = json.load(f)

                filename = result_file.stem

                if "hierarchical" in filename.lower():
                    method = "hierarchical"
                elif "ue_reflect" in filename.lower():
                    method = "ue_reflect"
                else:
                    method = method_name

                results = data.get("results", [])

                for result in results:
                    sample_id = result.get("sample_id", "")
                    mappings = extract_mappings_from_result(result, method, filename)

                    for mapping in mappings:
                        source = mapping["source_field"]
                        target = mapping["predicted_target"]
                        agreement_key = f"{source}|{target}"
                        agreement_tracker[agreement_key].append(method)

                        key = f"{source}|{target}|{method}|{filename}"
                        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]

                        if key_hash not in seen_keys:
                            seen_keys.add(key_hash)
                            mapping["id"] = key_hash
                            mapping["method"] = method
                            mapping["model"] = filename
                            mapping["sample_id"] = sample_id
                            mapping["agreement_key"] = agreement_key
                            all_mappings.append(mapping)

            except Exception:
                continue

    return all_mappings, dict(agreement_tracker)


def extract_mappings_from_result(result: Dict, method: str, filename: str) -> List[Dict]:
    """Extract mappings from different result formats."""
    mappings = []

    if "source_field" in result and "prediction" in result:
        if result.get("prediction"):
            mappings.append({
                "source_field": result["source_field"],
                "predicted_target": result["prediction"],
                "expected_target": result.get("expected_target", ""),
                "justification": result.get("mapping_result", [{}])[0].get("justification", "") if result.get("mapping_result") else "",
                "confidence": result.get("uncertainty_info", {}).get("score", 0),
                "auto_correct": result.get("correct", None),
            })

    elif "predicted_mapping" in result:
        pred_mapping = result.get("predicted_mapping", {})
        gt_mapping = result.get("ground_truth_mapping", {})

        for source_prop, target_info in pred_mapping.items():
            target = None
            justification = ""

            if isinstance(target_info, dict):
                target = target_info.get("target_property", "") or target_info.get("mapProperty", "")
                justification = target_info.get("justification", "")
            elif isinstance(target_info, list) and target_info:
                first_item = target_info[0]
                if isinstance(first_item, dict):
                    target = first_item.get("mapProperty", "") or first_item.get("target_property", "")
                    justification = first_item.get("justification", "")
                elif isinstance(first_item, str):
                    target = first_item
            elif isinstance(target_info, str):
                target = target_info

            if target:
                expected = ""
                if gt_mapping:
                    for src, tgt in gt_mapping.items():
                        expected = tgt
                        break

                mappings.append({
                    "source_field": source_prop,
                    "predicted_target": target,
                    "expected_target": expected,
                    "justification": justification,
                    "confidence": 0,
                    "auto_correct": None,
                })

    return mappings


def extract_api_from_field(field: str) -> Tuple[str, str]:
    """Extract API (product.Type) and field name from a full field path.

    Example: 'turbonomic.Action.details' -> ('turbonomic.Action', 'details')
    """
    if not field:
        return "", ""

    parts = field.split(".")
    if len(parts) >= 3:
        # product.Type.field format
        api = ".".join(parts[:2])
        field_name = ".".join(parts[2:])
        return api, field_name
    elif len(parts) == 2:
        # Could be product.Type or Type.field
        return parts[0], parts[1]
    else:
        return field, ""


def is_valid_mapping(source_field: str, target_field: str, lookup: Dict) -> bool:
    """Check if a mapping has valid source and target fields."""
    # Filter out mappings with unknown/invalid source or target
    if not source_field or source_field.lower() == "unknown":
        return False
    if not target_field or target_field.lower() == "unknown":
        return False

    # Filter out mappings where source and target are the same
    if source_field == target_field:
        return False

    # Check if we have field info for at least one of them
    source_info = lookup.get(source_field, {})
    target_info = lookup.get(target_field, {})

    # If both have unknown type and no description, it's likely invalid
    source_type = source_info.get("field_type", "unknown")
    target_type = target_info.get("field_type", "unknown")

    if source_type == "unknown" and target_type == "unknown":
        # Check if we have any description
        if not source_info.get("description") and not target_info.get("description"):
            return False

    return True


def group_by_api_mapping(mappings: List[Dict], agreement: Dict, lookup: Dict = None) -> Dict[str, Dict]:
    """Group field mappings by API (type) level mapping.

    Groups all field mappings between source_api and target_api together.
    Tracks which methods predicted each field mapping.
    """
    api_grouped = defaultdict(lambda: {
        "methods": set(),
        "field_mappings": {},  # Dict mapping field_key -> {info, methods}
        "source_api": "",
        "target_api": "",
        "justifications": [],
    })

    for mapping in mappings:
        source_field = mapping.get("source_field", "")
        target_field = mapping.get("predicted_target", "")

        # Skip invalid mappings
        if lookup and not is_valid_mapping(source_field, target_field, lookup):
            continue

        # Extract API level
        source_api, source_field_name = extract_api_from_field(source_field)
        target_api, target_field_name = extract_api_from_field(target_field)

        if not source_api or not target_api:
            continue

        # Skip if source and target APIs are the same
        if source_api == target_api:
            continue

        # Create API-level key
        api_key = f"{source_api}|{target_api}"
        method = mapping["method"]

        api_grouped[api_key]["methods"].add(method)
        api_grouped[api_key]["source_api"] = source_api
        api_grouped[api_key]["target_api"] = target_api

        # Track field mapping with methods
        field_key = f"{source_field_name}|{target_field_name}"
        if field_key not in api_grouped[api_key]["field_mappings"]:
            api_grouped[api_key]["field_mappings"][field_key] = {
                "source_field": source_field_name or source_field,
                "target_field": target_field_name or target_field,
                "full_source": source_field,
                "full_target": target_field,
                "methods": set(),
            }
        api_grouped[api_key]["field_mappings"][field_key]["methods"].add(method)

        if mapping.get("justification"):
            api_grouped[api_key]["justifications"].append({
                "method": method,
                "field_key": field_key,
                "text": mapping["justification"]
            })

    # Filter out GNN-only mappings (low quality type-based matching)
    GNN_ONLY_METHODS = {"gnn"}

    keys_to_remove = []
    for key in api_grouped:
        # Filter out field mappings that are ONLY predicted by GNN
        filtered_field_mappings = {}
        for field_key, field_info in api_grouped[key]["field_mappings"].items():
            # Keep mapping if it has at least one non-GNN method
            if field_info["methods"] - GNN_ONLY_METHODS:
                filtered_field_mappings[field_key] = field_info

        api_grouped[key]["field_mappings"] = filtered_field_mappings

        # Update methods to only include methods that have surviving field mappings
        surviving_methods = set()
        for field_info in filtered_field_mappings.values():
            surviving_methods.update(field_info["methods"])
        api_grouped[key]["methods"] = surviving_methods

        # Mark for removal if no field mappings left
        if not filtered_field_mappings:
            keys_to_remove.append(key)

    # Remove API mappings with no surviving field mappings
    for key in keys_to_remove:
        del api_grouped[key]

    # Convert sets to sorted lists
    for key in api_grouped:
        api_grouped[key]["methods"] = sorted(api_grouped[key]["methods"])
        api_grouped[key]["count"] = len(api_grouped[key]["methods"])

        # Convert field_mappings dict to list with methods as sorted list
        field_list = []
        for field_key, field_info in api_grouped[key]["field_mappings"].items():
            field_info["methods"] = sorted(field_info["methods"])
            field_info["field_key"] = field_key
            field_list.append(field_info)
        # Sort by number of methods (descending)
        field_list.sort(key=lambda x: len(x["methods"]), reverse=True)
        api_grouped[key]["field_mappings"] = field_list

    return dict(api_grouped)


# ============== Annotations ==============

def load_annotations() -> Dict:
    """Load existing annotations."""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {"annotations": {}, "metadata": {"created": datetime.now().isoformat()}}


def save_annotations(annotations: Dict):
    """Save annotations to file."""
    annotations["metadata"]["last_modified"] = datetime.now().isoformat()
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=2)


def save_api_annotation(api_key: str, source_api: str, target_api: str, mapping_type: str,
                        field_annotations: Dict[str, str], comment: str = ""):
    """Save a two-level annotation for an API mapping.

    Args:
        api_key: The API mapping key (source_api|target_api)
        source_api: Source API name
        target_api: Target API name
        mapping_type: Type of mapping - "similar", "joinable", or "invalid"
        field_annotations: Dict of field_key -> "yes"/"no"/"maybe"/None for each field mapping
        comment: Optional annotator comment
    """
    if "annotations" not in st.session_state.annotations:
        st.session_state.annotations["annotations"] = {}

    username = st.session_state.current_user.get("username", "unknown")

    # Separate field mappings by annotation
    valid_fields = [k for k, v in field_annotations.items() if v == "yes"]
    invalid_fields = [k for k, v in field_annotations.items() if v == "no"]
    maybe_fields = [k for k, v in field_annotations.items() if v == "maybe"]

    # Backwards compatibility: also set api_valid based on mapping_type
    api_valid = mapping_type in ("similar", "joinable")

    st.session_state.annotations["annotations"][api_key] = {
        "source_api": source_api,
        "target_api": target_api,
        "mapping_type": mapping_type,  # "similar", "joinable", or "invalid"
        "api_valid": api_valid,  # backwards compatibility
        "valid_field_mappings": valid_fields,
        "invalid_field_mappings": invalid_fields,
        "maybe_field_mappings": maybe_fields,
        "comment": comment,
        "annotator": username,
        "timestamp": datetime.now().isoformat(),
    }

    save_annotations(st.session_state.annotations)


def clear_annotation(key: str):
    """Clear an annotation."""
    if key in st.session_state.annotations.get("annotations", {}):
        del st.session_state.annotations["annotations"][key]
        save_annotations(st.session_state.annotations)


# ============== Session State ==============

def init_session_state():
    """Initialize session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "login_mode" not in st.session_state:
        st.session_state.login_mode = "signin"
    if "forgot_password_mode" not in st.session_state:
        st.session_state.forgot_password_mode = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "annotate"
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    if "mappings" not in st.session_state:
        st.session_state.mappings = None
    if "agreement" not in st.session_state:
        st.session_state.agreement = None
    if "property_lookup" not in st.session_state:
        st.session_state.property_lookup = load_property_lookup()
    if "schema_descriptions" not in st.session_state:
        st.session_state.schema_descriptions = load_schema_descriptions()
    if "filter_method" not in st.session_state:
        st.session_state.filter_method = "All"
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = "All"
    if "filter_agreement" not in st.session_state:
        st.session_state.filter_agreement = "All"
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "filter_product" not in st.session_state:
        st.session_state.filter_product = "All"


# ============== Login/Signup Page ==============

def render_login_page():
    """Render login/signup page."""
    st.title("Schema Mapping")

    # Forgot password mode
    if st.session_state.forgot_password_mode:
        render_forgot_password_form()
        return

    if st.session_state.login_mode == "signup":
        render_signup_form()
    else:
        render_signin_form()


def render_forgot_password_form():
    """Render forgot password form."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Reset Password")
        reset_username = st.text_input("Enter your username", key="reset_username")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset Password", type="primary", use_container_width=True):
                if not reset_username:
                    st.error("Please enter your username")
                else:
                    user = get_user_by_username(reset_username)
                    if not user:
                        st.error("Invalid credentials")
                    else:
                        user_email = user.get("email")
                        if not user_email:
                            st.error("No email address registered for this account. Please contact administrator.")
                        else:
                            # Generate new password
                            new_password = generate_temp_password()

                            # Reset password
                            success, message = reset_user_password(reset_username, new_password)
                            if success:
                                # Send email
                                email_success, email_message = send_password_reset_email(
                                    user_email, reset_username, new_password
                                )
                                if email_success:
                                    st.success(f"Password reset email sent to {user_email[:3]}***{user_email[-10:]}")
                                    st.info("Your new password has been sent. Please check your email.")
                                else:
                                    st.error(email_message)
                            else:
                                st.error(message)

        with col_b:
            if st.button("Back to Sign In", use_container_width=True):
                st.session_state.forgot_password_mode = False
                st.rerun()


def render_signin_form():
    """Render sign in form."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Sign In")

        with st.form("signin_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            col_a, col_b = st.columns(2)
            with col_a:
                signin_submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            with col_b:
                signup_clicked = st.form_submit_button("Sign Up", use_container_width=True)

            if signin_submitted:
                if not username or not password:
                    st.error("Please enter username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            if signup_clicked:
                st.session_state.login_mode = "signup"
                st.rerun()

        # Forgot password link outside the form
        if st.button("Forgot Password?", key="forgot_password_link"):
            st.session_state.forgot_password_mode = True
            st.rerun()


def render_signup_form():
    """Render sign up form."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Create Account")

        with st.form("signup_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            display_name = st.text_input("Display Name (optional)")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            col_a, col_b = st.columns(2)
            with col_a:
                create_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            with col_b:
                back_clicked = st.form_submit_button("Back to Sign In", use_container_width=True)

            if create_submitted:
                if not username or not password:
                    st.error("Username and password are required")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    success, message = register_user(username, password, email, display_name)
                    if success:
                        st.success(message)
                        # Auto sign in
                        st.session_state.authenticated = True
                        st.session_state.current_user = {
                            "username": username,
                            "email": email,
                            "display_name": display_name or username,
                        }
                        st.session_state.login_mode = "signin"
                        st.rerun()
                    else:
                        st.error(message)

            if back_clicked:
                st.session_state.login_mode = "signin"
                st.rerun()


# ============== Sidebar ==============

def render_annotate_sidebar_content():
    """Render sidebar for annotation page."""
    # Filters
    st.markdown("### Filters")

    if st.session_state.mappings:
        methods = ["All"] + sorted(set(m.get("method", "") for m in st.session_state.mappings))
        st.session_state.filter_method = st.selectbox("Method", methods)

        st.session_state.filter_status = st.selectbox(
            "Annotation Status",
            ["All", "Not Annotated", "Annotated"]
        )

        st.session_state.filter_agreement = st.selectbox(
            "Method Agreement",
            ["All", "High (3+)", "Medium (2)", "Low (1)"]
        )

    st.markdown("---")

    # Statistics
    st.markdown("### Statistics")

    annotations = st.session_state.annotations.get("annotations", {})

    if st.session_state.mappings and st.session_state.agreement:
        lookup = st.session_state.property_lookup
        api_mappings = group_by_api_mapping(st.session_state.mappings, st.session_state.agreement, lookup)
        total_mappings = len(api_mappings)
    else:
        total_mappings = 0

    annotated = len(annotations)

    similar_count = sum(1 for a in annotations.values() if a.get("mapping_type") == "similar")
    joinable_count = sum(1 for a in annotations.values() if a.get("mapping_type") == "joinable")
    invalid_count = sum(1 for a in annotations.values() if a.get("mapping_type") == "invalid")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total APIs", total_mappings)
        st.metric("Similar", similar_count)
        st.metric("Invalid", invalid_count)
    with col2:
        st.metric("Annotated", annotated)
        st.metric("Joinable", joinable_count)

    if total_mappings > 0:
        progress = min(1.0, annotated / total_mappings)
        st.progress(progress, text=f"{progress*100:.1f}% complete")

    st.markdown("---")

    # Methods breakdown
    if st.session_state.mappings:
        st.markdown("### Methods")
        method_counts = defaultdict(int)
        for m in st.session_state.mappings:
            method_counts[m.get("method", "unknown")] += 1
        for method, count in sorted(method_counts.items()):
            st.caption(f"{method}: {count}")

    st.markdown("---")

    # Navigation
    if st.button("View Annotations", use_container_width=True):
        st.session_state.current_page = "view_annotations"
        st.rerun()

    # Export
    export_to_json()


def render_view_annotations_sidebar_content():
    """Render sidebar for view annotations page."""
    st.markdown("### Filter by Annotator")

    annotations = st.session_state.annotations.get("annotations", {})
    annotators = sorted(set(a.get("annotator", "unknown") for a in annotations.values()))

    if annotators:
        selected = st.selectbox("Annotator", ["All"] + annotators)
        st.session_state.filter_annotator = selected
    else:
        st.caption("No annotations yet")
        st.session_state.filter_annotator = "All"

    st.markdown("---")

    st.markdown("### Filter by Type")
    st.session_state.filter_label = st.selectbox("Mapping Type", ["All", "Similar", "Joinable", "Invalid"])

    st.markdown("---")

    # Summary stats
    st.markdown("### Summary")

    if annotations:
        annotator_stats = defaultdict(lambda: {"similar": 0, "joinable": 0, "invalid": 0, "total": 0})
        for a in annotations.values():
            annotator = a.get("annotator", "unknown")
            mapping_type = a.get("mapping_type", "")
            if mapping_type == "similar":
                annotator_stats[annotator]["similar"] += 1
            elif mapping_type == "joinable":
                annotator_stats[annotator]["joinable"] += 1
            elif mapping_type == "invalid":
                annotator_stats[annotator]["invalid"] += 1
            annotator_stats[annotator]["total"] += 1

        for annotator, stats in sorted(annotator_stats.items()):
            st.markdown(f"**{annotator}**")
            st.caption(f"Total: {stats['total']} | Similar: {stats['similar']} | Joinable: {stats['joinable']} | Invalid: {stats['invalid']}")


def export_to_json():
    """Export annotations to JSON."""
    annotations = st.session_state.annotations.get("annotations", {})

    # Structure the export by mapping type
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "total_annotations": len(annotations),
        },
        "similar_api_mappings": [],  # APIs representing same concept
        "joinable_api_mappings": [],  # APIs that can be linked
        "invalid_api_mappings": [],  # No valid mapping
    }

    for api_key, ann in annotations.items():
        api_entry = {
            "source_api": ann.get("source_api", ""),
            "target_api": ann.get("target_api", ""),
            "mapping_type": ann.get("mapping_type", ""),
            "valid_field_mappings": ann.get("valid_field_mappings", []),
            "invalid_field_mappings": ann.get("invalid_field_mappings", []),
            "maybe_field_mappings": ann.get("maybe_field_mappings", []),
            "comment": ann.get("comment", ""),
            "annotator": ann.get("annotator", ""),
            "timestamp": ann.get("timestamp", ""),
        }

        mapping_type = ann.get("mapping_type", "")
        if mapping_type == "similar":
            export_data["similar_api_mappings"].append(api_entry)
        elif mapping_type == "joinable":
            export_data["joinable_api_mappings"].append(api_entry)
        else:
            export_data["invalid_api_mappings"].append(api_entry)

    json_content = json.dumps(export_data, indent=2)

    st.download_button(
        label="Download Annotations",
        data=json_content,
        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )


# ============== Annotation Page ==============

def render_method_tags(methods: List[str]):
    """Render method tags showing which methods agree."""
    tags_html = ""
    for method in methods:
        tags_html += f'<span class="method-tag">{method}</span>'
    st.markdown(tags_html, unsafe_allow_html=True)


def render_api_mapping_card(api_key: str, mapping_info: Dict):
    """Render an API-level mapping card with two-level annotation."""
    source_api = mapping_info["source_api"]
    target_api = mapping_info["target_api"]
    methods = mapping_info["methods"]
    field_mappings = mapping_info["field_mappings"]
    justifications = mapping_info.get("justifications", [])

    annotations = st.session_state.annotations.get("annotations", {})
    current_annotation = annotations.get(api_key, {})
    mapping_type = current_annotation.get("mapping_type", None)
    valid_fields = set(current_annotation.get("valid_field_mappings", []))
    invalid_fields = set(current_annotation.get("invalid_field_mappings", []))
    maybe_fields = set(current_annotation.get("maybe_field_mappings", []))

    # Color coding by mapping type
    if mapping_type == "similar":
        border_color = "#28a745"  # Green for similar
    elif mapping_type == "joinable":
        border_color = "#17a2b8"  # Blue for joinable
    elif mapping_type == "invalid":
        border_color = "#dc3545"  # Red for invalid
    else:
        border_color = "#dee2e6"  # Gray for unannotated

    lookup = st.session_state.property_lookup

    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
        """, unsafe_allow_html=True)

        # Source API -> Target API
        st.markdown(f'<div class="api-name">{source_api} &rarr; {target_api}</div>', unsafe_allow_html=True)

        # Method tags - show which methods agree on API level
        render_method_tags(methods)

        # API Documentation - get fields from the field_mappings if lookup doesn't have them
        schema_descs = st.session_state.get("schema_descriptions", {})
        source_info = get_api_info(source_api, lookup, schema_descs)
        target_info = get_api_info(target_api, lookup, schema_descs)

        # If lookup didn't find fields, extract from the field mappings
        if source_info["field_count"] == 0 and field_mappings:
            source_fields = set(fm["source_field"] for fm in field_mappings)
            source_info["fields"] = [{"name": f, "type": "", "description": ""} for f in sorted(source_fields)]
            source_info["field_count"] = len(source_fields)

        if target_info["field_count"] == 0 and field_mappings:
            target_fields = set(fm["target_field"] for fm in field_mappings)
            target_info["fields"] = [{"name": f, "type": "", "description": ""} for f in sorted(target_fields)]
            target_info["field_count"] = len(target_fields)

        with st.expander("API Documentation", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Source: {source_api}**")
                if source_info["type_description"]:
                    st.markdown(f"_{source_info['type_description'][:300]}_")
                st.caption(f"{source_info['field_count']} fields")
                if source_info["fields"]:
                    field_list = ", ".join([f["name"] for f in source_info["fields"][:15]])
                    if len(source_info["fields"]) > 15:
                        field_list += f" ... +{len(source_info['fields']) - 15} more"
                    st.caption(f"Fields: {field_list}")

            with col2:
                st.markdown(f"**Target: {target_api}**")
                if target_info["type_description"]:
                    st.markdown(f"_{target_info['type_description'][:300]}_")
                st.caption(f"{target_info['field_count']} fields")
                if target_info["fields"]:
                    field_list = ", ".join([f["name"] for f in target_info["fields"][:15]])
                    if len(target_info["fields"]) > 15:
                        field_list += f" ... +{len(target_info['fields']) - 15} more"
                    st.caption(f"Fields: {field_list}")

        # API-level justification summary
        if justifications:
            # Get unique justifications for API-level summary (first one from each method)
            method_justifications = {}
            for j in justifications:
                if isinstance(j, dict):
                    method = j.get("method", "")
                    if method and method not in method_justifications:
                        method_justifications[method] = j.get("text", "")[:200]

            if method_justifications:
                with st.expander(f"API Mapping Rationale ({len(method_justifications)} methods)", expanded=False):
                    for method, text in method_justifications.items():
                        st.caption(f"**{method}:** {text}...")

        st.markdown("---")

        # Field mappings with per-field annotation
        st.markdown(f"**Field Mappings ({len(field_mappings)} fields)**")
        st.caption("For each field mapping, select Yes (valid), No (invalid), or Maybe (unsure)")

        # Initialize field annotations in session state if not present
        field_state_key = f"fields_{api_key}"
        if field_state_key not in st.session_state:
            # Initialize from saved annotation
            st.session_state[field_state_key] = {}
            for fm in field_mappings:
                fk = fm["field_key"]
                if fk in valid_fields:
                    st.session_state[field_state_key][fk] = "yes"
                elif fk in invalid_fields:
                    st.session_state[field_state_key][fk] = "no"
                elif fk in maybe_fields:
                    st.session_state[field_state_key][fk] = "maybe"
                else:
                    st.session_state[field_state_key][fk] = None

        # Build a lookup of justifications by field_key
        field_justifications = defaultdict(list)
        for j in justifications:
            if isinstance(j, dict):
                fk = j.get("field_key", "")
                if fk:
                    field_justifications[fk].append(j)

        for fm in field_mappings:
            source_field = fm["source_field"]
            target_field = fm["target_field"]
            full_source = fm.get("full_source", source_field)
            full_target = fm.get("full_target", target_field)
            field_methods = fm.get("methods", [])
            field_key = fm["field_key"]

            # Get field info
            source_info = get_field_info(full_source, lookup)
            target_info = get_field_info(full_target, lookup)

            source_type = source_info.get("field_type", "")
            target_type = target_info.get("field_type", "")

            type_info = ""
            if source_type and target_type and source_type != "unknown" and target_type != "unknown":
                type_info = f" ({source_type} → {target_type})"

            # Method tags for this field
            method_tags = " ".join([f'<span class="method-tag" style="font-size:0.7em;padding:1px 4px;">{m}</span>' for m in field_methods])

            # Current annotation for this field
            current_val = st.session_state[field_state_key].get(field_key, None)

            col1, col2, col3, col4 = st.columns([3, 0.5, 0.5, 0.5])

            with col1:
                st.markdown(f"`{source_field}` → `{target_field}`{type_info} {method_tags}", unsafe_allow_html=True)

                # Show field descriptions if available
                source_desc = source_info.get("description", "")
                target_desc = target_info.get("description", "")
                if source_desc or target_desc:
                    desc_parts = []
                    if source_desc and source_desc != source_field:
                        desc_parts.append(f"**{source_field}**: {source_desc[:100]}")
                    if target_desc and target_desc != target_field:
                        desc_parts.append(f"**{target_field}**: {target_desc[:100]}")
                    if desc_parts:
                        st.caption(" | ".join(desc_parts))

                # Show justifications for this field mapping
                field_justs = field_justifications.get(field_key, [])
                if field_justs:
                    for j in field_justs[:2]:
                        method = j.get("method", "")
                        text = j.get("text", "")[:200]
                        st.caption(f"_{method}: {text}..._")

            with col2:
                if st.button("Y", key=f"y_{api_key}_{field_key}",
                            type="primary" if current_val == "yes" else "secondary",
                            help="Yes - Valid mapping"):
                    st.session_state[field_state_key][field_key] = "yes"
                    st.rerun()

            with col3:
                if st.button("N", key=f"n_{api_key}_{field_key}",
                            type="primary" if current_val == "no" else "secondary",
                            help="No - Invalid mapping"):
                    st.session_state[field_state_key][field_key] = "no"
                    st.rerun()

            with col4:
                if st.button("?", key=f"m_{api_key}_{field_key}",
                            type="primary" if current_val == "maybe" else "secondary",
                            help="Maybe - Unsure"):
                    st.session_state[field_state_key][field_key] = "maybe"
                    st.rerun()

        # Comment box
        current_comment = current_annotation.get("comment", "")
        comment = st.text_input(
            "Comment (optional)",
            value=current_comment,
            key=f"comment_{api_key}",
            placeholder="Add a note about this API mapping..."
        )

        st.markdown("---")

        # API-level annotation with mapping type
        st.markdown("**What type of mapping is this?**")
        st.caption("**Similar**: APIs represent the same concept | **Joinable**: APIs can be linked via relationship | **Invalid**: No valid mapping")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("Similar", key=f"similar_{api_key}",
                        type="primary" if mapping_type == "similar" else "secondary",
                        use_container_width=True):
                field_annotations = st.session_state.get(field_state_key, {})
                save_api_annotation(api_key, source_api, target_api, "similar", field_annotations, comment)
                st.rerun()

        with col2:
            if st.button("Joinable", key=f"joinable_{api_key}",
                        type="primary" if mapping_type == "joinable" else "secondary",
                        use_container_width=True):
                field_annotations = st.session_state.get(field_state_key, {})
                save_api_annotation(api_key, source_api, target_api, "joinable", field_annotations, comment)
                st.rerun()

        with col3:
            if st.button("Invalid", key=f"invalid_{api_key}",
                        type="primary" if mapping_type == "invalid" else "secondary",
                        use_container_width=True):
                field_annotations = st.session_state.get(field_state_key, {})
                save_api_annotation(api_key, source_api, target_api, "invalid", field_annotations, comment)
                st.rerun()

        with col4:
            if st.button("Skip", key=f"skip_{api_key}", use_container_width=True):
                st.session_state.current_index = min(
                    st.session_state.current_index + 1,
                    st.session_state.get("total_mappings", 0) - 1
                )
                st.rerun()

        with col5:
            if mapping_type is not None:
                if st.button("Clear", key=f"clear_{api_key}", use_container_width=True):
                    clear_annotation(api_key)
                    if field_state_key in st.session_state:
                        del st.session_state[field_state_key]
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def filter_api_mappings(api_mappings: Dict) -> Dict:
    """Apply filters to API-level mappings."""
    filtered = {}
    annotations = st.session_state.annotations.get("annotations", {})

    for key, info in api_mappings.items():
        if st.session_state.filter_method != "All":
            if st.session_state.filter_method not in info["methods"]:
                continue

        if st.session_state.filter_status == "Not Annotated":
            if key in annotations:
                continue
        elif st.session_state.filter_status == "Annotated":
            if key not in annotations:
                continue

        count = info["count"]
        if st.session_state.filter_agreement == "High (3+)":
            if count < 3:
                continue
        elif st.session_state.filter_agreement == "Medium (2)":
            if count != 2:
                continue
        elif st.session_state.filter_agreement == "Low (1)":
            if count != 1:
                continue

        filtered[key] = info

    return filtered


def get_valid_products(lookup: Dict) -> set:
    """Get set of valid product names from the lookup."""
    products = set()
    for v in lookup.values():
        prod = v.get('product_name', '')
        if prod:
            products.add(prod)
    return products


# Product aliases for short forms used in data
PRODUCT_ALIASES = {
    'turbo': 'turbonomic',
    'omop': 'omop',
    'synthea': 'synthea',
    'nl2gql': 'nl2gql',
}


def get_product_from_api(api_name: str, valid_products: set = None) -> str:
    """Extract product name from API name like 'turbonomic_Action' -> 'turbonomic'.

    Uses valid_products set to correctly handle products with hyphens/underscores.
    Handles aliases like 'turbo' -> 'turbonomic'.
    """
    # Handle nested APIs like 'concert.packages.recommendation' - take first part
    if '.' in api_name and '_' not in api_name.split('.')[0]:
        api_name = api_name.split('.')[0]

    normalized = api_name.replace(".", "_")

    if valid_products:
        # Try to match against known products (longest match first)
        for product in sorted(valid_products, key=len, reverse=True):
            if normalized.startswith(product + "_") or normalized == product:
                return product

    # Extract first part before underscore
    parts = normalized.split("_")
    first_part = parts[0] if parts else ""

    # Check for aliases
    if first_part in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[first_part]

    # Filter out non-product names (likely type names or other artifacts)
    non_products = {'recommendation', 'packages', 'edges', 'nodes', 'items', 'tags',
                   'filter', 'input', 'exposures', 'environments', 'openissues',
                   'severitybreakdown', 'agreements', 'assessment', 'documents',
                   'transactions', 'vehicles', 'students', 'student', 'research',
                   'journalist', 'furniture', 'financial', 'degree', 'cars',
                   'accelerator', 'acceptance', 'residents', 'document',
                   'exposurescollection', 'openissues'}

    if first_part.lower() in non_products:
        return ""

    # Only return if it looks like a valid product name (not a random word)
    if first_part and first_part[0].islower():
        return first_part

    return ""


def render_annotate_page():
    """Render main annotation page."""
    st.title("Schema Mapping")

    if st.session_state.mappings is None:
        with st.spinner("Loading mappings from all methods..."):
            st.session_state.mappings, st.session_state.agreement = load_all_mappings()

    mappings = st.session_state.mappings
    agreement = st.session_state.agreement

    if not mappings:
        st.warning("No mappings found. Check that result files exist.")
        return

    lookup = st.session_state.property_lookup
    api_mappings = group_by_api_mapping(mappings, agreement, lookup)

    # Get valid products from lookup
    valid_products = get_valid_products(lookup)

    # Get unique source products from mappings
    source_products = set()
    for key, info in api_mappings.items():
        product = get_product_from_api(info["source_api"], valid_products)
        if product:
            source_products.add(product)
    source_products = sorted(source_products)

    # Product filter dropdown
    col_filter, col_spacer = st.columns([2, 3])
    with col_filter:
        selected_product = st.selectbox(
            "Filter by Source Product",
            ["All"] + source_products,
            index=(["All"] + source_products).index(st.session_state.filter_product) if st.session_state.filter_product in ["All"] + source_products else 0,
            key="product_filter"
        )
        if selected_product != st.session_state.filter_product:
            st.session_state.filter_product = selected_product
            st.session_state.current_index = 0  # Reset to first item when filter changes
            st.rerun()

    # Apply product filter
    if st.session_state.filter_product != "All":
        api_mappings = {k: v for k, v in api_mappings.items()
                       if get_product_from_api(v["source_api"], valid_products) == st.session_state.filter_product}

    filtered_mappings = filter_api_mappings(api_mappings)

    # Sort by method count (high to low), then by number of field mappings
    sorted_keys = sorted(filtered_mappings.keys(),
                        key=lambda k: (filtered_mappings[k]["count"], len(filtered_mappings[k]["field_mappings"])),
                        reverse=True)

    total_count = len(sorted_keys)
    st.session_state.total_mappings = total_count

    if total_count == 0:
        st.info("No API mappings match the current filters.")
        return

    # Ensure current_index is valid
    if st.session_state.current_index >= total_count:
        st.session_state.current_index = 0
    if st.session_state.current_index < 0:
        st.session_state.current_index = 0

    current_idx = st.session_state.current_index

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("< Prev", use_container_width=True, disabled=(current_idx == 0)):
            st.session_state.current_index = max(0, current_idx - 1)
            st.rerun()

    with col2:
        if st.button("Next >", use_container_width=True, disabled=(current_idx >= total_count - 1)):
            st.session_state.current_index = min(total_count - 1, current_idx + 1)
            st.rerun()

    with col3:
        st.markdown(f"**{current_idx + 1} of {total_count}** API mappings")

    # Show current API mapping
    current_key = sorted_keys[current_idx]
    render_api_mapping_card(current_key, filtered_mappings[current_key])


# ============== View Annotations Page ==============

def render_view_annotations_page():
    """Render view annotations page."""
    st.title("View Annotations")

    # Back button
    if st.button("< Back to Annotate"):
        st.session_state.current_page = "annotate"
        st.rerun()

    st.markdown("---")

    annotations = st.session_state.annotations.get("annotations", {})

    if not annotations:
        st.info("No annotations yet. Go to 'Annotate' to start annotating mappings.")
        return

    # Apply filters
    filter_annotator = getattr(st.session_state, 'filter_annotator', 'All')
    filter_label = getattr(st.session_state, 'filter_label', 'All')

    filtered = {}
    for key, ann in annotations.items():
        if filter_annotator != "All" and ann.get("annotator") != filter_annotator:
            continue
        # Filter by mapping type
        mapping_type = ann.get("mapping_type", "")
        if filter_label == "Similar":
            if mapping_type != "similar":
                continue
        elif filter_label == "Joinable":
            if mapping_type != "joinable":
                continue
        elif filter_label == "Invalid":
            if mapping_type != "invalid":
                continue
        filtered[key] = ann

    st.markdown(f"**Showing {len(filtered)} of {len(annotations)} annotations**")

    # Sort by timestamp (newest first)
    sorted_annotations = sorted(filtered.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True)

    for key, ann in sorted_annotations:
        mapping_type = ann.get("mapping_type", "")
        if mapping_type == "similar":
            label = "SIMILAR"
            label_color = "#28a745"
            label_bg = "#d4edda"
        elif mapping_type == "joinable":
            label = "JOINABLE"
            label_color = "#17a2b8"
            label_bg = "#d1ecf1"
        elif mapping_type == "invalid":
            label = "INVALID"
            label_color = "#dc3545"
            label_bg = "#f8d7da"
        else:
            label = "UNKNOWN"
            label_color = "#856404"
            label_bg = "#fff3cd"

        valid_fields = ann.get('valid_field_mappings', [])
        invalid_fields = ann.get('invalid_field_mappings', [])
        maybe_fields = ann.get('maybe_field_mappings', [])

        comment = ann.get('comment', '')
        comment_html = f'<br/><em style="color: #6c757d;">"{comment}"</em>' if comment else ''

        st.markdown(f"""
        <div class="annotation-row">
            <span style="background-color: {label_bg}; color: {label_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold;">
                {label}
            </span>
            <strong style="margin-left: 10px;">{ann.get('annotator', 'unknown')}</strong>
            <span style="color: #6c757d; margin-left: 10px;">{ann.get('timestamp', '')[:19]}</span>
            <br/>
            <code>{ann.get('source_api', '')}</code> &rarr; <code>{ann.get('target_api', '')}</code>
            <br/>
            <small style="color: #28a745;">Valid: {len(valid_fields)}</small> |
            <small style="color: #dc3545;">Invalid: {len(invalid_fields)}</small> |
            <small style="color: #ffc107;">Maybe: {len(maybe_fields)}</small>
            {comment_html}
        </div>
        """, unsafe_allow_html=True)


# ============== Main ==============

def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar is always visible
    with st.sidebar:
        if st.session_state.authenticated:
            # Show current user name and buttons
            user = st.session_state.current_user
            st.markdown(f"**{user.get('display_name', user.get('username'))}**")
            st.caption(user.get("email", ""))

            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.session_state.mappings = None
                st.rerun()

            st.markdown("---")

            # Show filters and stats when authenticated
            if st.session_state.current_page == "annotate":
                render_annotate_sidebar_content()
            else:
                render_view_annotations_sidebar_content()
        else:
            # Show branding when not authenticated
            st.markdown("**Schema Mapping**")
            st.caption("Please sign in to continue")
            st.markdown("---")
            st.info("Sign in to annotate schema mappings")

    # Main content
    if not st.session_state.authenticated:
        render_login_page()
    else:
        if st.session_state.current_page == "annotate":
            render_annotate_page()
        else:
            render_view_annotations_page()


if __name__ == "__main__":
    main()
