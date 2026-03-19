#!/usr/bin/env python3
"""
Build the ITOM-Hashicorp schema mapping dataset.

Maps conceptual ITOM entities (from entity_model.json) to actual GraphQL/protobuf
types (from itopsgraph_api_lookup.json), then creates ground truth mappings between
ITOM and Hashicorp schema types.

Output:
  data/itom_hashicorp_api_lookup.json   - Combined API lookup for both sides
  data/itom_hashicorp_dataset.jsonl     - Evaluation utterances with ground truth
  data/itom_hashicorp_entity_mapping.json - Entity-to-schema-type mapping (audit)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    with open(DATA_DIR / "itom_hashicorp_entity_model.json") as f:
        entity_model = json.load(f)
    with open(DATA_DIR / "itopsgraph_api_lookup.json") as f:
        api_lookup = json.load(f)
    return entity_model, api_lookup


def build_type_index(api_lookup):
    """Build product -> type_name -> {fields, descriptions} index."""
    index = {}
    for field_key, meta in api_lookup.items():
        product = meta.get("product_name", "")
        type_name = meta.get("type_name", "")
        if not product or not type_name:
            continue
        if product not in index:
            index[product] = {}
        if type_name not in index[product]:
            index[product][type_name] = {
                "fields": [],
                "field_count": 0,
                "descriptions": set(),
            }
        index[product][type_name]["fields"].append(field_key)
        index[product][type_name]["field_count"] += 1
        desc = meta.get("description", "")
        if desc:
            index[product][type_name]["descriptions"].add(desc)
    return index


def camel_words(name):
    """Split CamelCase and underscore names into lowercase words."""
    # Split camelCase
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", name)
    # Also split on underscores
    result = set()
    for p in parts:
        result.add(p.lower())
    for p in name.split("_"):
        if p:
            result.add(p.lower())
            # Also split camelCase within underscore parts
            for w in re.findall(r"[A-Z][a-z]+|[a-z]+", p):
                result.add(w.lower())
    return result


def is_top_level_type(type_name):
    """Filter out nested query/mutation response types — keep top-level schema types."""
    # Skip deeply nested response types like "query_getTrace_rootSpan_..."
    if type_name.count("_") > 2:
        return False
    # Skip mutation/query input/response types
    skip_prefixes = [
        "query_", "mutation_", "mutationInput_", "_query_",
        "queryInput_", "create", "update", "delete", "get",
        "find", "add", "remove", "set", "list",
    ]
    lower = type_name.lower()
    for prefix in skip_prefixes:
        if lower.startswith(prefix.lower()):
            return False
    # Skip types ending in Input, Response, etc.
    skip_suffixes = [
        "_Input", "_input", "_response", "Input", "Response",
        "_items", "Collection", "Connection", "Edge",
    ]
    for suffix in skip_suffixes:
        if type_name.endswith(suffix):
            return False
    return True


def map_entities_to_types(entity_model, type_index):
    """Map each conceptual entity to the best matching actual schema type."""
    mapping = {}

    # Manual overrides for known mappings
    overrides = {
        # Instana — map to actual GraphQL types
        ("instana", "Application"): [("instana", "ApplicationPerspective")],
        ("instana", "Service"): [("instana", "ApplicationService")],
        ("instana", "Metrics"): [("instana", "Entity")],  # Entity carries metric references
        ("instana", "Traces"): [("instana", "Event")],  # Events/traces are the closest top-level type
        ("instana", "Logs"): [],  # No direct top-level type
        ("instana", "Latency"): [],
        ("instana", "Throughput"): [],
        ("instana", "Synthetics"): [],
        # Concert — map to actual GraphQL types
        ("concert", "Application"): [("concert", "Application")],
        ("concert", "Environment"): [("concert", "Environment"), ("concert", "AppEnvironment")],
        ("concert", "AccessPoints"): [("concert", "Access_points")],
        ("concert", "BuildArtifact"): [("concert", "BuildArtifact")],
        ("concert", "CVEVulnerability"): [("concert", "Vulnerability")],
        ("concert", "Patch"): [("concert", "VulnerabilityPatchInput")],
        ("concert", "ResilienceProfile"): [("concert", "Profile")],
        ("concert", "SourceCode"): [("concert", "Applications_source_repo")],
        ("concert", "CodeRepos"): [("concert", "Images_repositories")],
        ("concert", "CICDPipelines"): [("concert", "Container_builds")],
        ("concert", "ITSMData"): [("concert", "Recommendations")],
        ("concert", "IncidentData"): [("concert", "Entity_settings")],
        ("concert", "ResilienceMetrics"): [("concert", "Topology_compliance_view")],
        ("concert", "ComplianceApplication"): [("concert", "Topology_compliance_view")],
        ("concert", "ComplianceReports"): [],
        # Turbonomic — map to actual GraphQL types
        ("turbonomic", "TurboApplication"): [("turbonomic", "Entity")],
        ("turbonomic", "ResourceUsages"): [("turbonomic", "StatsEntry")],
        ("turbonomic", "ServiceMetrics"): [("turbonomic", "Statistics")],
        ("turbonomic", "OptimizationPolicy"): [("turbonomic", "Template")],
        ("turbonomic", "OptimizationAction"): [("turbonomic", "Action")],
        # HCP Terraform
        ("hcp_terraform", "TerraformTemplate"): [("hcp_terraform", "HcpTerraformWorkspace")],
        # Core infrastructure -> infragraph/cloud types
        ("core_infrastructure", "ContainerImage"): [("infragraph", "BaseVirtualMachineImage")],
        ("core_infrastructure", "ContainerWorkload"): [("infragraph", "BaseVirtualMachine")],
        ("core_infrastructure", "Service"): [("infragraph", "BaseServerlessFunction")],
        ("core_infrastructure", "Cluster"): [("aws", "AwsEcsCluster"), ("turbonomic", "ContainerClusterEntity")],
        ("core_infrastructure", "Node"): [("infragraph", "NodeIdentifier")],
        ("core_infrastructure", "Pod"): [("turbonomic", "WorkloadControllerEntity")],
        ("core_infrastructure", "Namespace"): [("turbonomic", "NamespaceEntity")],
        ("core_infrastructure", "Deployment"): [],
        ("core_infrastructure", "Volume"): [("infragraph", "BaseVolume"), ("aws", "AwsEc2Volume")],
        ("core_infrastructure", "AccessPoint"): [("concert", "Access_points")],
    }

    for product, product_data in entity_model.items():
        for entity_name, entity_data in product_data.get("entities", {}).items():
            key = (product, entity_name)
            if key in overrides:
                matched = []
                for p, t in overrides[key]:
                    if p in type_index and t in type_index[p]:
                        matched.append({
                            "product": p,
                            "type_name": t,
                            "field_count": type_index[p][t]["field_count"],
                            "match_method": "manual_override",
                        })
                mapping[key] = {
                    "entity": entity_data,
                    "matched_types": matched,
                }
            else:
                # Automatic matching as fallback
                mapping[key] = {
                    "entity": entity_data,
                    "matched_types": [],
                }

    return mapping


def build_cross_product_ground_truth(entity_mapping, entity_model):
    """Build type-level ground truth mappings between ITOM and Hashicorp."""
    itom_products = {"instana", "concert", "turbonomic", "hcp_terraform"}
    hashi_products = {"infragraph", "packer", "hcp_terraform", "aws", "azure", "gcp", "hcp"}

    # Semantic equivalences from the entity model (cross-product relationships)
    # These define which conceptual entities map across products
    semantic_pairs = [
        # --- ITOM cross-product: Application triple ---
        ("concert", "Application", "instana", "Application", "semantic_equivalent",
         "Both represent an application — Concert for governance, Instana for observability"),
        ("concert", "Application", "turbonomic", "TurboApplication", "semantic_equivalent",
         "Both represent an application — Concert for governance, Turbonomic for optimization"),
        ("instana", "Application", "turbonomic", "TurboApplication", "semantic_equivalent",
         "Both represent an application — Instana for observability, Turbonomic for optimization"),

        # --- ITOM cross-product: Metrics/observability ---
        ("turbonomic", "ServiceMetrics", "instana", "Metrics", "semantic_equivalent",
         "Both capture service performance metrics — Turbonomic for optimization, Instana for observability"),
        ("turbonomic", "ResourceUsages", "instana", "Metrics", "semantic_equivalent",
         "Both capture resource utilization — Turbonomic for optimization, Instana for observability"),
        ("concert", "ResilienceMetrics", "instana", "Metrics", "derived_from",
         "Concert resilience metrics are derived from Instana observability metrics"),
        ("concert", "ResilienceMetrics", "instana", "Traces", "derived_from",
         "Concert resilience metrics incorporate Instana distributed traces"),

        # --- ITOM cross-product: Incident/Action ---
        ("concert", "IncidentData", "instana", "Traces", "semantic_equivalent",
         "Concert incidents correlate with Instana trace anomalies"),
        ("concert", "ITSMData", "turbonomic", "OptimizationAction", "semantic_equivalent",
         "Concert ITSM change records track Turbonomic optimization actions"),
        ("concert", "CVEVulnerability", "instana", "Application", "references",
         "Concert CVEs affect applications monitored by Instana"),

        # --- ITOM -> core infrastructure projections ---
        ("instana", "Service", "core_infrastructure", "Service", "projection",
         "Instana surfaces core infrastructure Service with observability attributes"),
        ("concert", "BuildArtifact", "core_infrastructure", "ContainerImage", "supply_chain",
         "Concert build artifacts packaged into container images"),
        ("concert", "AccessPoints", "core_infrastructure", "AccessPoint", "projection",
         "Concert environment access points map to core infrastructure"),
        ("turbonomic", "OptimizationAction", "core_infrastructure", "ContainerWorkload", "acts_on",
         "Turbonomic optimization actions target container workloads"),
        ("turbonomic", "TurboApplication", "core_infrastructure", "Service", "composed_of",
         "Turbonomic application is composed of core infrastructure services"),
        ("instana", "Application", "core_infrastructure", "Service", "composed_of",
         "Instana application is composed of core infrastructure services"),
        ("concert", "CICDPipelines", "core_infrastructure", "ContainerImage", "produces",
         "Concert CI/CD pipelines produce container images"),

        # --- Hashicorp (Terraform) -> core infrastructure ---
        ("hcp_terraform", "TerraformTemplate", "core_infrastructure", "Cluster", "governs",
         "Terraform templates govern cluster provisioning"),
        ("hcp_terraform", "TerraformTemplate", "core_infrastructure", "Node", "governs",
         "Terraform templates govern node provisioning"),
        ("hcp_terraform", "TerraformTemplate", "core_infrastructure", "Service", "governs",
         "Terraform templates govern service definitions"),
        ("hcp_terraform", "TerraformTemplate", "core_infrastructure", "ContainerWorkload", "governs",
         "Terraform templates govern workload specifications"),
        ("hcp_terraform", "TerraformTemplate", "core_infrastructure", "Volume", "governs",
         "Terraform templates provision volumes"),
    ]

    # Resolve conceptual entities to actual schema types and build ground truth
    ground_truth = []
    for src_prod, src_entity, tgt_prod, tgt_entity, rel_type, description in semantic_pairs:
        src_key = (src_prod, src_entity)
        tgt_key = (tgt_prod, tgt_entity)

        src_types = entity_mapping.get(src_key, {}).get("matched_types", [])
        tgt_types = entity_mapping.get(tgt_key, {}).get("matched_types", [])

        if not src_types or not tgt_types:
            continue

        for st in src_types:
            for tt in tgt_types:
                ground_truth.append({
                    "source_product": st["product"],
                    "source_type": st["type_name"],
                    "target_product": tt["product"],
                    "target_type": tt["type_name"],
                    "relationship": rel_type,
                    "description": description,
                    "source_entity": src_entity,
                    "target_entity": tgt_entity,
                })

    # Also add direct Hashicorp internal mappings (infragraph Base* -> aws/azure/gcp)
    hashi_mappings = [
        ("infragraph", "BaseVirtualMachine", "aws", "AwsEc2Instance",
         "Infragraph base VM abstraction maps to AWS EC2 instance"),
        ("infragraph", "BaseVirtualMachine", "azure", "AzureVirtualMachine",
         "Infragraph base VM abstraction maps to Azure VM"),
        ("infragraph", "BaseVirtualMachine", "gcp", "GcpComputeInstance",
         "Infragraph base VM abstraction maps to GCP compute instance"),
        ("infragraph", "BaseAccount", "aws", "AwsAccount",
         "Infragraph base account maps to AWS account"),
        ("infragraph", "BaseSecurityGroup", "aws", "AwsSecurityGroup",
         "Infragraph base security group maps to AWS security group"),
        ("infragraph", "BaseVolume", "aws", "AwsEc2Volume",
         "Infragraph base volume maps to AWS EBS volume"),
        ("infragraph", "BaseStorageBucket", "aws", "AwsS3Bucket",
         "Infragraph base storage bucket maps to AWS S3 bucket"),
        ("infragraph", "BaseServerlessFunction", "aws", "AwsLambdaFunction",
         "Infragraph base serverless function maps to AWS Lambda"),
        ("infragraph", "BaseRole", "aws", "AwsIamRole",
         "Infragraph base role maps to AWS IAM role"),
        ("infragraph", "BaseUser", "aws", "AwsIamUser",
         "Infragraph base user maps to AWS IAM user"),
        ("infragraph", "BasePolicy", "aws", "AwsIamPolicy",
         "Infragraph base policy maps to AWS IAM policy"),
        ("infragraph", "BaseSubnet", "aws", "AwsSubnet",
         "Infragraph base subnet maps to AWS subnet"),
        ("infragraph", "BaseVpc", "aws", "AwsVpc",
         "Infragraph base VPC maps to AWS VPC"),
        ("infragraph", "BaseRegion", "aws", "AwsRegion",
         "Infragraph base region maps to AWS region"),
        ("infragraph", "BaseNetworkInterface", "aws", "AwsNetworkInterface",
         "Infragraph base network interface maps to AWS ENI"),
        ("infragraph", "BaseSecurityRule", "aws", "AwsSecurityGroupRule",
         "Infragraph base security rule maps to AWS SG rule"),
        ("infragraph", "BaseZone", "aws", "AwsAvailabilityZone",
         "Infragraph base zone maps to AWS availability zone"),
        ("infragraph", "BaseVirtualMachineImage", "aws", "AwsAmi",
         "Infragraph base VM image maps to AWS AMI"),
        ("infragraph", "BaseVirtualMachineType", "aws", "AwsEc2InstanceType",
         "Infragraph base VM type maps to AWS instance type"),
        # Packer -> infragraph
        ("packer", "PackerBuild", "infragraph", "PackerBuild",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerVersion", "infragraph", "PackerVersion",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerArtifact", "infragraph", "PackerArtifact",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerChannel", "infragraph", "PackerChannel",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerBucket", "infragraph", "PackerBucket",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerRegistry", "infragraph", "PackerRegistry",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerAncestry", "infragraph", "PackerAncestry",
         "Same entity in both packer and infragraph schemas"),
        ("packer", "PackerSbom", "infragraph", "PackerSbom",
         "Same entity in both packer and infragraph schemas"),
        # HCP -> infragraph/hcp_terraform
        ("hcp", "HcpTerraformWorkspace", "hcp_terraform", "HcpTerraformWorkspace",
         "Same entity across HCP and HCP Terraform schemas"),
        ("hcp", "HcpTerraformRun", "hcp_terraform", "HcpTerraformRun",
         "Same entity across HCP and HCP Terraform schemas"),
        ("hcp", "HcpTerraformStateVersion", "hcp_terraform", "HcpTerraformStateVersion",
         "Same entity across HCP and HCP Terraform schemas"),
        ("hcp", "HcpTerraformProject", "hcp_terraform", "HcpTerraformProject",
         "Same entity across HCP and HCP Terraform schemas"),
        ("hcp", "HcpTerraformOrganization", "hcp_terraform", "HcpTerraformOrganization",
         "Same entity across HCP and HCP Terraform schemas"),
        ("hcp", "HcpTerraformAgentPool", "hcp_terraform", "HcpTerraformAgentPool",
         "Same entity across HCP and HCP Terraform schemas"),
        # ITOM -> Hashicorp cross-product (via infragraph Base types)
        ("turbonomic", "VirtualDisksEntry", "infragraph", "BaseVolume",
         "Turbonomic virtual disk maps to infragraph base volume"),
        ("turbonomic", "CloudType", "infragraph", "Source",
         "Turbonomic cloud type maps to infragraph source"),
        ("turbonomic", "CloudAspect", "aws", "AwsAccount",
         "Turbonomic cloud aspect maps to AWS account"),
        ("turbonomic", "CloudAspect", "aws", "AwsEcsCluster",
         "Turbonomic cloud aspect maps to AWS ECS cluster"),
    ]

    for src_prod, src_type, tgt_prod, tgt_type, desc in hashi_mappings:
        ground_truth.append({
            "source_product": src_prod,
            "source_type": src_type,
            "target_product": tgt_prod,
            "target_type": tgt_type,
            "relationship": "schema_mapping",
            "description": desc,
            "source_entity": src_type,
            "target_entity": tgt_type,
        })

    return ground_truth


def build_api_lookup(api_lookup, ground_truth):
    """Build combined API lookup for ITOM + Hashicorp products."""
    itom_products = {"instana", "concert", "turbonomic", "hcp_terraform"}
    hashi_products = {"infragraph", "packer", "hcp_terraform", "aws", "azure", "gcp", "hcp"}
    all_products = itom_products | hashi_products

    # Get all types mentioned in ground truth
    gt_types = set()
    for gt in ground_truth:
        gt_types.add((gt["source_product"], gt["source_type"]))
        gt_types.add((gt["target_product"], gt["target_type"]))

    # Filter api_lookup to only include fields from relevant products
    filtered = {}
    for field_key, meta in api_lookup.items():
        product = meta.get("product_name", "")
        if product in all_products:
            filtered[field_key] = meta

    return filtered


def build_dataset(ground_truth):
    """Build evaluation dataset in JSONL format."""
    samples = []
    sample_id = 0

    for gt in ground_truth:
        src = f"{gt['source_product']}_{gt['source_type']}"
        tgt = f"{gt['target_product']}_{gt['target_type']}"

        utterance = (
            f"Map {gt['source_product']} {gt['source_type']} "
            f"to {gt['target_product']} schema"
        )

        sample = {
            "id": f"itom_hashi_{sample_id:04d}",
            "utterance": utterance,
            "source_schema": gt["source_product"],
            "target_schema": gt["target_product"],
            "expected_target_type": tgt,
            "mapping": {},  # Field-level mapping (empty for now — focus is type-level)
            "metadata": {
                "source_type": gt["source_type"],
                "target_type": gt["target_type"],
                "relationship": gt["relationship"],
                "description": gt["description"],
                "source_entity": gt.get("source_entity", ""),
                "target_entity": gt.get("target_entity", ""),
            },
        }
        samples.append(sample)
        sample_id += 1

    return samples


def main():
    print("Loading data...")
    entity_model, api_lookup = load_data()
    type_index = build_type_index(api_lookup)

    print("Mapping conceptual entities to schema types...")
    entity_mapping = map_entities_to_types(entity_model, type_index)

    # Audit: print mapping
    mapped = 0
    unmapped = 0
    for (product, entity), data in sorted(entity_mapping.items()):
        types = data["matched_types"]
        if types:
            mapped += 1
            type_strs = [f"{t['product']}.{t['type_name']}" for t in types]
            print(f"  {product}.{entity} -> {', '.join(type_strs)}")
        else:
            unmapped += 1
            print(f"  {product}.{entity} -> (no schema type)")
    print(f"\nMapped: {mapped}, Unmapped: {unmapped}")

    print("\nBuilding ground truth mappings...")
    ground_truth = build_cross_product_ground_truth(entity_mapping, entity_model)
    print(f"  {len(ground_truth)} type-level mappings")

    # Count by category
    itom_prods = {"instana", "concert", "turbonomic", "hcp_terraform"}
    hashi_prods = {"infragraph", "packer", "hcp_terraform", "aws", "azure", "gcp", "hcp"}
    cross = sum(1 for g in ground_truth
                if (g["source_product"] in itom_prods and g["target_product"] in hashi_prods)
                or (g["source_product"] in hashi_prods and g["target_product"] in itom_prods))
    print(f"  Cross-product (ITOM<->Hashi): {cross}")
    print(f"  Internal: {len(ground_truth) - cross}")

    print("\nBuilding API lookup...")
    filtered_lookup = build_api_lookup(api_lookup, ground_truth)
    print(f"  {len(filtered_lookup)} fields from {len(set(v['product_name'] for v in filtered_lookup.values()))} products")

    print("\nBuilding evaluation dataset...")
    dataset = build_dataset(ground_truth)
    print(f"  {len(dataset)} evaluation samples")

    # Save outputs
    out_lookup = DATA_DIR / "itom_hashicorp_api_lookup.json"
    with open(out_lookup, "w") as f:
        json.dump(filtered_lookup, f, indent=2)
    print(f"\nSaved: {out_lookup}")

    out_dataset = DATA_DIR / "itom_hashicorp_dataset.jsonl"
    with open(out_dataset, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved: {out_dataset}")

    # Save entity mapping for audit
    out_mapping = DATA_DIR / "itom_hashicorp_entity_mapping.json"
    serializable = {}
    for (product, entity), data in entity_mapping.items():
        serializable[f"{product}.{entity}"] = {
            "description": data["entity"]["description"],
            "matched_types": data["matched_types"],
        }
    with open(out_mapping, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {out_mapping}")


if __name__ == "__main__":
    main()
