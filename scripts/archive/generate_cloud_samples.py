#!/usr/bin/env python3
"""
Generate dataset samples for AWS, Azure, and GCP products.
Creates mapping samples between cloud providers and other ITOpsGraph products.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Load the API lookup
with open(DATA_DIR / "itopsgraph_api_lookup.json") as f:
    API_LOOKUP = json.load(f)

# Organize by product and type
PRODUCT_TYPES = {}
TYPE_FIELDS = {}

for api_name, info in API_LOOKUP.items():
    product = info.get('product_name', '')
    type_name = info.get('type_name', '')
    field_name = info.get('field_name', '')
    field_type = info.get('field_type', '')

    if product and type_name:
        if product not in PRODUCT_TYPES:
            PRODUCT_TYPES[product] = set()
        PRODUCT_TYPES[product].add(type_name)

        key = f"{product}_{type_name}"
        if key not in TYPE_FIELDS:
            TYPE_FIELDS[key] = []
        TYPE_FIELDS[key].append({
            'field_name': field_name,
            'field_type': field_type,
            'api_name': api_name
        })

# Define semantic mappings between cloud types and other products
# Format: (source_product, source_type, target_product, target_type, field_mappings)
# field_mappings: list of (source_field, target_field, justification)

SEMANTIC_MAPPINGS = [
    # AWS EC2 Instance mappings
    ("aws", "AwsEc2Instance", "turbonomic", "Entity", [
        ("virtual_machine", "aspects", "Both represent the core virtual machine entity with its properties"),
        ("private_ip", "connectedEntities", "Network connectivity information"),
        ("monitoring", "aspects", "Monitoring state maps to entity aspects"),
    ]),
    ("aws", "AwsEc2Instance", "instana", "InfrastructureMetricResult", [
        ("virtual_machine", "metric", "Virtual machine metrics map to infrastructure metrics"),
        ("monitoring", "metric", "Monitoring state relates to metrics collection"),
    ]),

    # AWS Account mappings
    ("aws", "AwsAccount", "turbonomic", "BusinessAccount", [
        ("account", "accountId", "Account identifiers represent the same business entity"),
        ("aws_resource", "cloudType", "AWS resource type maps to cloud provider type"),
    ]),
    ("aws", "AwsAccount", "concert", "EntityAccessUsersCollection", [
        ("account", "users", "Account maps to user access collection"),
    ]),

    # AWS Volume mappings
    ("aws", "AwsEc2Volume", "turbonomic", "VirtualDisksEntry", [
        ("volume", "virtualDisks", "Volume storage maps to virtual disk entries"),
    ]),

    # AWS Lambda mappings
    ("aws", "AwsLambdaFunction", "turbonomic", "Entity", [
        ("serverless_function", "aspects", "Serverless function maps to entity aspects"),
        ("memory_size", "aspects", "Memory configuration is part of entity aspects"),
        ("run_time", "aspects", "Runtime configuration maps to entity aspects"),
    ]),

    # AWS IAM Role mappings
    ("aws", "AwsIamRole", "turbonomic", "Entity", [
        ("role", "aspects", "IAM role maps to entity security aspects"),
        ("description", "displayName", "Role description maps to entity display name"),
    ]),
    ("aws", "AwsIamRole", "instana", "Role", [
        ("role", "id", "Role identifier mapping"),
        ("description", "name", "Role description to name mapping"),
    ]),

    # AWS IAM Policy mappings
    ("aws", "AwsIamPolicy", "turbonomic", "ActionSchedule", [
        ("policy", "schedule", "Policy rules map to scheduled actions"),
        ("attachment_count", "schedule", "Policy usage maps to schedule configuration"),
    ]),

    # AWS IAM User mappings
    ("aws", "AwsIamUser", "instana", "User", [
        ("user", "id", "User identifier mapping"),
    ]),
    ("aws", "AwsIamUser", "concert", "Userkey", [
        ("user", "key", "User maps to user key"),
    ]),

    # AWS VPC mappings
    ("aws", "AwsVpc", "turbonomic", "Entity", [
        ("vpc", "connectedEntities", "VPC network maps to entity connections"),
    ]),

    # AWS Subnet mappings
    ("aws", "AwsSubnet", "turbonomic", "Entity", [
        ("subnet", "connectedEntities", "Subnet network maps to entity connections"),
    ]),

    # AWS Security Group mappings
    ("aws", "AwsSecurityGroup", "turbonomic", "Entity", [
        ("security_group", "aspects", "Security group maps to entity security aspects"),
    ]),

    # AWS S3 Bucket mappings
    ("aws", "AwsS3Bucket", "turbonomic", "Entity", [
        ("storage_bucket", "aspects", "Storage bucket maps to entity storage aspects"),
    ]),

    # AWS ECS Cluster mappings
    ("aws", "AwsEcsCluster", "turbonomic", "ContainerClusterEntity", [
        ("base", "aspects", "ECS cluster base maps to container cluster aspects"),
    ]),
    ("aws", "AwsEcsCluster", "instana", "KubernetesClusterInfo", [
        ("base", "id", "ECS cluster maps to Kubernetes cluster info"),
    ]),

    # AWS ECS Service mappings
    ("aws", "AwsEcsService", "turbonomic", "Entity", [
        ("base", "aspects", "ECS service maps to entity aspects"),
        ("cluster_id", "connectedEntities", "Cluster ID maps to entity connections"),
    ]),

    # AWS Region mappings
    ("aws", "AwsRegion", "turbonomic", "CurrentLocation", [
        ("region", "displayName", "Region maps to location name"),
    ]),
    ("aws", "AwsRegion", "turbonomic", "DataCenter", [
        ("region", "displayName", "Region maps to data center name"),
    ]),

    # AWS AMI mappings
    ("aws", "AwsAmi", "turbonomic", "Entity", [
        ("virtual_machine_image", "aspects", "AMI maps to entity image aspects"),
    ]),

    # Azure Virtual Machine mappings
    ("azure", "AzureVirtualMachine", "turbonomic", "Entity", [
        ("virtual_machine", "aspects", "Virtual machine maps to entity aspects"),
        ("priority", "aspects", "VM priority maps to entity aspects"),
    ]),
    ("azure", "AzureVirtualMachine", "aws", "AwsEc2Instance", [
        ("virtual_machine", "virtual_machine", "Both represent virtual machine instances"),
        ("priority", "tenancy", "VM priority relates to instance tenancy"),
    ]),
    ("azure", "AzureVirtualMachine", "instana", "InfrastructureMetricResult", [
        ("virtual_machine", "metric", "Virtual machine maps to infrastructure metrics"),
    ]),

    # Azure Resource mappings
    ("azure", "AzureResource", "aws", "AwsResource", [
        ("etag", "arn", "Resource identifier mapping (etag to ARN)"),
        ("provisioning_state", "arn", "Provisioning state relates to resource ARN"),
    ]),
    ("azure", "AzureResource", "turbonomic", "Entity", [
        ("etag", "uuid", "Resource etag maps to entity UUID"),
        ("provisioning_state", "state", "Provisioning state maps to entity state"),
    ]),

    # GCP Compute Instance mappings
    ("gcp", "GcpComputeInstance", "turbonomic", "Entity", [
        ("virtual_machine", "aspects", "GCP instance maps to entity aspects"),
    ]),
    ("gcp", "GcpComputeInstance", "aws", "AwsEc2Instance", [
        ("virtual_machine", "virtual_machine", "Both represent virtual machine instances"),
    ]),
    ("gcp", "GcpComputeInstance", "azure", "AzureVirtualMachine", [
        ("virtual_machine", "virtual_machine", "Both represent virtual machine instances"),
    ]),
    ("gcp", "GcpComputeInstance", "instana", "InfrastructureMetricResult", [
        ("virtual_machine", "metric", "Virtual machine maps to infrastructure metrics"),
    ]),
]

# Utterance templates
UTTERANCE_TEMPLATES = [
    "map {source_product} {source_type}.{source_field} to {target_product} {target_type}.{target_field}",
    "find {target_product} fields equivalent to {source_product} {source_type}.{source_field}",
    "what {target_product} field corresponds to {source_product} {source_type} {source_field}?",
    "match {source_product} {source_type}.{source_field} to {target_product} schema",
    "how does {source_product} {source_type}.{source_field} translate to {target_product}?",
    "map {source_product} {source_field} with {target_product} {target_type}.{target_field}",
    "find the {target_product} equivalent of {source_product} {source_type} {source_field}",
    "convert {source_product} {source_type}.{source_field} to {target_product} format",
]


def generate_samples():
    """Generate dataset samples from semantic mappings."""
    samples = []
    sample_id = 0

    for source_product, source_type, target_product, target_type, field_mappings in SEMANTIC_MAPPINGS:
        for source_field, target_field, justification in field_mappings:
            # Generate multiple utterances for each mapping
            for template in random.sample(UTTERANCE_TEMPLATES, min(3, len(UTTERANCE_TEMPLATES))):
                utterance = template.format(
                    source_product=source_product,
                    source_type=source_type,
                    source_field=source_field,
                    target_product=target_product,
                    target_type=target_type,
                    target_field=target_field,
                )

                sample = {
                    "id": f"itopsgraph_cloud_{sample_id:04d}",
                    "utterance": utterance,
                    "source_schema": source_product,
                    "target_schema": target_product,
                    "mapping": {
                        f"{source_product}_{source_type}.{source_field}": f"{target_product}_{target_type}.{target_field}"
                    },
                    "metadata": {
                        "confidence": 0.7,
                        "justification": justification,
                        "method": "semantic_mapping",
                        "equivalence_score": 0.0,
                        "joinability_score": 0.0,
                        "source_type": source_type,
                        "target_type": target_type
                    }
                }
                samples.append(sample)
                sample_id += 1

    return samples


def main():
    # Generate samples
    samples = generate_samples()

    # Count by source product
    source_counts = {}
    for s in samples:
        src = s['source_schema']
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"Generated {len(samples)} samples")
    print("By source product:")
    for product, count in sorted(source_counts.items()):
        print(f"  {product}: {count}")

    # Save to file
    output_file = DATA_DIR / "itopsgraph_cloud_dataset.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nSaved to {output_file}")

    # Also append to main dataset
    main_dataset = DATA_DIR / "itopsgraph_dataset.jsonl"

    # Read existing samples to get the last ID
    existing_samples = []
    with open(main_dataset) as f:
        for line in f:
            existing_samples.append(json.loads(line))

    # Find the max existing ID number
    max_id = 0
    for s in existing_samples:
        if s['id'].startswith('itopsgraph_'):
            try:
                num = int(s['id'].split('_')[-1])
                max_id = max(max_id, num)
            except:
                pass

    # Renumber cloud samples
    for i, sample in enumerate(samples):
        sample['id'] = f"itopsgraph_cloud_{max_id + 1 + i:04d}"

    # Append to main dataset
    with open(main_dataset, 'a') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Appended {len(samples)} samples to {main_dataset}")
    print(f"Total samples now: {len(existing_samples) + len(samples)}")


if __name__ == "__main__":
    main()
