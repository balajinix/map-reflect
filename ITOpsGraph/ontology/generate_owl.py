"""
Generate OWL ontology from glossary terms.

Creates a semantic hierarchy similar to the curated ontology:
- Deduplicates terms across products
- Organizes into semantic categories
- Creates deeper IS-A hierarchies
- Filters out non-concept terms (metrics, UI elements)

Usage:
    python generate_owl.py --input ../concepts/output/glossary/terms.json --output itops-ontology.owl
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


# Semantic hierarchy - maps terms to their parent class
# Format: 'normalized_term': 'ParentClass'
TERM_HIERARCHY = {
    # Infrastructure hierarchy
    'virtualmachine': 'Host',
    'vm': 'Host',
    'physicalhost': 'Host',
    'host': 'Infrastructure',
    'container': 'Infrastructure',
    'pod': 'Infrastructure',
    'cluster': 'Infrastructure',
    'node': 'Cluster',
    'zone': 'Infrastructure',
    'region': 'Infrastructure',
    'datacenter': 'Infrastructure',
    'datacentre': 'Infrastructure',
    'network': 'Infrastructure',
    'storage': 'Infrastructure',
    'volume': 'Storage',
    'disk': 'Storage',

    # Observable hierarchy
    'alert': 'Event',
    'incident': 'Event',
    'anomaly': 'Event',
    'event': 'Observable',
    'metric': 'Observable',
    'sli': 'Metric',
    'trace': 'Observable',
    'span': 'Trace',
    'log': 'Observable',
    'slo': 'Observable',

    # Workload hierarchy
    'application': 'Workload',
    'service': 'Workload',
    'microservice': 'Service',
    'job': 'Workload',
    'function': 'Workload',
    'process': 'Workload',
    'lambda': 'Function',

    # Operation hierarchy
    'action': 'Operation',
    'scalingaction': 'Action',
    'remediationaction': 'Action',
    'automation': 'Operation',
    'workflow': 'Automation',
    'runbook': 'Automation',
    'playbook': 'Automation',
    'policy': 'Operation',
    'schedule': 'Operation',

    # Agent hierarchy
    'monitoringagent': 'Agent',
    'sensor': 'Agent',
    'collector': 'Agent',

    # Resource hierarchy
    'computeresource': 'Resource',
    'storageresource': 'Resource',
    'networkresource': 'Resource',
    'cpu': 'ComputeResource',
    'memory': 'ComputeResource',

    # Cost hierarchy
    'budget': 'CostEntity',
    'costallocation': 'CostEntity',
    'reservation': 'CostEntity',
    'chargeback': 'CostEntity',
    'showback': 'CostEntity',

    # Organization hierarchy
    'team': 'Organization',
    'tenant': 'Organization',
    'account': 'Organization',
    'project': 'Organization',
    'workspace': 'Organization',

    # Integration hierarchy
    'api': 'Integration',
    'webhook': 'Integration',
    'target': 'Integration',
    'endpoint': 'Integration',

    # Deployment hierarchy
    'build': 'Deployment',
    'release': 'Deployment',
    'pipeline': 'Deployment',
    'environment': 'Deployment',
}

# Top-level categories under ITOpsEntity
CATEGORY_PATTERNS = {
    'Infrastructure': [
        'host', 'vm', 'virtual machine', 'container', 'pod', 'cluster', 'node',
        'zone', 'region', 'data center', 'datacenter', 'network', 'storage',
        'server', 'instance', 'volume', 'disk', 'namespace',
    ],
    'Observable': [
        'metric', 'trace', 'span', 'log', 'event', 'alert', 'incident',
        'slo', 'sli', 'latency', 'throughput', 'error rate', 'availability',
        'anomaly', 'baseline', 'health', 'status',
    ],
    'Workload': [
        'application', 'service', 'microservice', 'workload', 'process',
        'function', 'lambda', 'job', 'task', 'component',
    ],
    'Operation': [
        'action', 'automation', 'workflow', 'runbook', 'playbook',
        'policy', 'schedule', 'remediation', 'scaling', 'optimization',
    ],
    'Agent': [
        'agent', 'sensor', 'collector', 'monitor', 'probe', 'tracer',
    ],
    'Resource': [
        'resource', 'cpu', 'memory', 'compute', 'quota', 'limit', 'capacity',
    ],
    'CostEntity': [
        'cost', 'budget', 'spend', 'charge', 'billing', 'allocation',
        'chargeback', 'showback', 'reservation', 'savings', 'price',
    ],
    'Organization': [
        'organization', 'team', 'user', 'group', 'account', 'tenant',
        'project', 'workspace', 'business unit', 'department',
    ],
    'Integration': [
        'integration', 'api', 'webhook', 'target', 'endpoint', 'connection',
        'channel', 'notification',
    ],
    'Deployment': [
        'deployment', 'build', 'release', 'pipeline', 'environment',
        'version', 'artifact', 'package',
    ],
}

# Terms to skip (not good ontology classes)
SKIP_PATTERNS = [
    # Metrics and measurements
    r'cpu\s*(total|utilization|usage)',
    r'gpu\s*(total|memory|utilization)',
    r'memory\s*(total|usage|utilization)',
    r'disk\s*(usage|utilization|iops)',
    r'\(\s*%\s*\)',  # Percentage indicators
    r'\(\s*max\s*\)',
    r'\(\s*min\s*\)',
    r'\(\s*avg\s*\)',

    # UI/Navigation elements
    r'^last\s+ran',
    r'^next\s+steps',
    r'^getting\s+started',
    r'^overview$',
    r'^introduction$',

    # Generic/vague terms
    r'^data$',
    r'^value$',
    r'^type$',
    r'^name$',
    r'^id$',
    r'^key$',
    r'^info$',
    r'^details$',
    r'^settings$',
    r'^configuration$',
    r'^options$',

    # Technical acronyms without context
    r'^api$',
    r'^cli$',
    r'^ssh$',
    r'^sql$',
    r'^json$',
    r'^hcl$',
    r'^rest$',
    r'^saml$',
    r'^dto$',
    r'^iam$',
    r'^idp$',
    r'^oss$',
    r'^ova$',
    r'^tfe$',
    r'^vdi$',
    r'^roi$',
    r'^nfu$',
    r'^iops$',
    r'^byol$',
    r'^sbom$',
    r'^cve$',

    # Implementation details
    r'module$',
    r'argument$',
    r'variable',
    r'expression',
    r'token',
    r'regex',
    r'backend$',
    r'directory$',
    r'debug$',
    r'profiler$',
    r'prompt$',
    r'fork$',
    r'bucket$',

    # Time/date terms
    r'^year$',
    r'^month$',
    r'^week$',
    r'^day$',
    r'^hour$',
    r'^minute$',

    # Data type terms
    r'data\s*type',
    r'primitive',
    r'composite',
    r'simple\s*data',

    # Generic dimensions/metrics
    r'dimension$',
    r'ancestors',
    r'ancestry',

    # Too generic standalone terms
    r'^run$',
    r'^note$',
    r'^apply$',
]

# Known ontology concepts - only these (and their hierarchy) will be included
KNOWN_CONCEPTS = {
    # Infrastructure
    'host', 'virtual machine', 'vm', 'container', 'pod', 'cluster', 'node',
    'zone', 'region', 'data center', 'datacenter', 'network', 'storage',
    'volume', 'disk', 'namespace', 'server', 'instance',

    # Observability
    'metric', 'trace', 'span', 'log', 'event', 'alert', 'incident',
    'anomaly', 'slo', 'sli', 'dashboard', 'monitor', 'health check',
    'baseline', 'threshold', 'notification',

    # Workload
    'application', 'service', 'microservice', 'workload', 'process',
    'function', 'job', 'task', 'component', 'endpoint', 'api',

    # Operation
    'action', 'automation', 'workflow', 'runbook', 'playbook',
    'policy', 'schedule', 'remediation', 'scaling', 'optimization',
    'deployment', 'build', 'release', 'pipeline', 'environment',

    # Agent
    'agent', 'sensor', 'collector', 'tracer', 'probe',

    # Resource
    'resource', 'cpu', 'memory', 'compute', 'quota', 'limit', 'capacity',

    # Cost
    'cost', 'budget', 'spend', 'billing', 'allocation', 'chargeback',
    'showback', 'reservation', 'savings',

    # Organization
    'organization', 'team', 'user', 'group', 'account', 'tenant',
    'project', 'workspace', 'role', 'permission',

    # Integration
    'integration', 'webhook', 'target', 'channel', 'connection',

    # Security
    'vulnerability', 'compliance', 'audit', 'credential', 'secret',
    'certificate', 'policy', 'access', 'identity',
}

# Object properties (relationships between classes)
OBJECT_PROPERTIES = [
    ('runsOn', 'Workload', 'Infrastructure', 'Indicates that a workload runs on infrastructure'),
    ('monitors', 'Agent', 'ITOpsEntity', 'Indicates monitoring relationship'),
    ('generates', 'ITOpsEntity', 'Observable', 'Indicates that an entity generates observability data'),
    ('belongsTo', 'ITOpsEntity', 'Organization', 'Indicates organizational ownership'),
    ('hasComponent', 'Workload', 'Workload', 'Indicates component relationship'),
    ('dependsOn', 'ITOpsEntity', 'ITOpsEntity', 'Indicates dependency relationship'),
    ('triggers', 'Observable', 'Operation', 'Indicates that an event triggers an operation'),
    ('incursCost', 'ITOpsEntity', 'CostEntity', 'Indicates cost association'),
    ('contains', 'Infrastructure', 'Infrastructure', 'Indicates containment (cluster contains nodes)'),
    ('hosts', 'Infrastructure', 'Workload', 'Indicates that infrastructure hosts workloads'),
    ('consumes', 'Workload', 'Resource', 'Indicates resource consumption'),
    ('deploysTo', 'Deployment', 'Infrastructure', 'Indicates deployment target'),
    ('integratesWith', 'ITOpsEntity', 'Integration', 'Indicates integration relationship'),
]

# Data properties (attributes)
DATA_PROPERTIES = [
    ('name', 'string', 'The name of the entity'),
    ('description', 'string', 'A description of the entity'),
    ('status', 'string', 'Current status of the entity'),
    ('createdAt', 'dateTime', 'When the entity was created'),
    ('updatedAt', 'dateTime', 'When the entity was last updated'),
    ('version', 'string', 'Version identifier'),
    ('identifier', 'string', 'Unique identifier'),
    ('severity', 'string', 'Severity level (for alerts/incidents)'),
    ('value', 'decimal', 'Numeric value (for metrics)'),
    ('unit', 'string', 'Unit of measurement'),
    ('threshold', 'decimal', 'Threshold value'),
    ('tags', 'string', 'Associated tags or labels'),
    ('sourceProduct', 'string', 'The product this entity originates from'),
]


@dataclass
class OntologyClass:
    """Represents an OWL class."""
    name: str
    label: str
    parent: str
    definition: str = ""
    source_products: list = field(default_factory=list)
    alt_labels: list = field(default_factory=list)


def normalize_class_name(term: str) -> str:
    """Convert a term to a valid OWL class name (PascalCase)."""
    # Remove parenthetical content
    term = re.sub(r'\([^)]*\)', '', term)
    # Split on spaces, hyphens, underscores
    words = re.split(r'[\s\-_]+', term.strip())
    # Capitalize each word and join
    return ''.join(word.capitalize() for word in words if word)


def should_skip_term(term: str, definition: str) -> bool:
    """Check if term should be skipped (not a good ontology class)."""
    text = term.lower().strip()

    # Skip very short terms
    if len(text) < 3:
        return True

    # Skip terms that are just numbers or symbols
    if re.match(r'^[\d\s\-_\.]+$', term):
        return True

    # Check skip patterns
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def is_known_concept(term: str, definition: str) -> bool:
    """Check if term is a known ontology concept or closely related."""
    text = term.lower().strip()

    # Direct match
    if text in KNOWN_CONCEPTS:
        return True

    # Check if term contains a known concept
    for concept in KNOWN_CONCEPTS:
        if concept in text or text in concept:
            return True

    # Check if definition strongly indicates a known concept
    def_lower = definition.lower()
    strong_indicators = 0
    for concept in KNOWN_CONCEPTS:
        if concept in def_lower:
            strong_indicators += 1
            if strong_indicators >= 2:
                return True

    return False


def get_parent_class(term: str, definition: str) -> str:
    """Determine the parent class for a term."""
    normalized = normalize_class_name(term).lower()

    # Check explicit hierarchy mapping first
    if normalized in TERM_HIERARCHY:
        return TERM_HIERARCHY[normalized]

    # Check category patterns
    text = f"{term} {definition}".lower()

    scores = {}
    for category, patterns in CATEGORY_PATTERNS.items():
        score = sum(1 for p in patterns if p in text)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)

    return 'ITOpsEntity'  # Default to root


def generate_owl(terms: list, output_path: Path, base_uri: str):
    """Generate OWL ontology from glossary terms."""

    classes = {}

    # Add root class
    classes['ITOpsEntity'] = OntologyClass(
        name='ITOpsEntity',
        label='IT Operations Entity',
        parent='',
        definition='Root class for all IT Operations domain concepts.',
    )

    # Add category classes
    for category in CATEGORY_PATTERNS.keys():
        classes[category] = OntologyClass(
            name=category,
            label=category,
            parent='ITOpsEntity',
            definition=f'Category for {category.lower()} related concepts.',
        )

    # Add intermediate classes from hierarchy
    intermediate_classes = set(TERM_HIERARCHY.values())
    for cls in intermediate_classes:
        if cls not in classes and cls != 'ITOpsEntity':
            # Find parent for this intermediate class
            parent = get_parent_class(cls, '')
            classes[cls] = OntologyClass(
                name=cls,
                label=cls,
                parent=parent,
                definition=f'{cls} concept.',
            )

    # Process glossary terms
    for term_data in terms:
        term = term_data.get('term', '').strip()
        definition = term_data.get('definition', '').strip()
        product = term_data.get('product', '').strip()
        aliases = term_data.get('aliases', [])

        if not term or not definition:
            continue

        # Skip non-concept terms
        if should_skip_term(term, definition):
            continue

        # Only include known ontology concepts
        if not is_known_concept(term, definition):
            continue

        class_name = normalize_class_name(term)
        if not class_name or len(class_name) < 2:
            continue

        # Handle duplicates - merge products
        if class_name in classes:
            if product and product not in classes[class_name].source_products:
                classes[class_name].source_products.append(product)
            # Keep longer definition
            if len(definition) > len(classes[class_name].definition):
                classes[class_name].definition = definition
            continue

        # Get parent class
        parent = get_parent_class(term, definition)

        classes[class_name] = OntologyClass(
            name=class_name,
            label=term,
            parent=parent,
            definition=definition,
            source_products=[product] if product else [],
            alt_labels=aliases,
        )

    # Generate OWL XML
    owl_content = generate_owl_xml(classes, base_uri)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(owl_content)

    print(f"Generated ontology with {len(classes)} classes")
    print(f"Saved to: {output_path}")

    # Print hierarchy summary
    from collections import defaultdict
    children = defaultdict(list)
    for cls in classes.values():
        children[cls.parent].append(cls.name)

    print("\nTop-level categories:")
    for cat in sorted(children.get('ITOpsEntity', [])):
        count = len(children.get(cat, []))
        print(f"  {cat}: {count} subclasses")

    return classes


def generate_owl_xml(classes: dict, base_uri: str) -> str:
    """Generate OWL/RDF-XML content."""

    lines = []

    # XML header and namespaces
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<rdf:RDF xmlns="{base_uri}#"')
    lines.append(f'     xml:base="{base_uri}"')
    lines.append('     xmlns:owl="http://www.w3.org/2002/07/owl#"')
    lines.append('     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
    lines.append('     xmlns:xml="http://www.w3.org/XML/1998/namespace"')
    lines.append('     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"')
    lines.append('     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"')
    lines.append('     xmlns:skos="http://www.w3.org/2004/02/skos/core#"')
    lines.append('     xmlns:dc="http://purl.org/dc/elements/1.1/"')
    lines.append(f'     xmlns:itops="{base_uri}#">')
    lines.append('')

    # Ontology declaration
    lines.append(f'    <owl:Ontology rdf:about="{base_uri}">')
    lines.append('        <dc:title>ITOps Domain Ontology</dc:title>')
    lines.append('        <dc:description>An ontology for IT Operations domain concepts, derived from IBM AIOps product glossaries including Instana, Turbonomic, Concert, Cloudability, and others.</dc:description>')
    lines.append('        <dc:creator>ITOpsGraph Project</dc:creator>')
    lines.append(f'        <dc:date>{datetime.now().year}</dc:date>')
    lines.append('        <owl:versionInfo>1.0.0</owl:versionInfo>')
    lines.append('    </owl:Ontology>')
    lines.append('')

    # Annotation properties
    lines.append('    <!-- Annotation Properties -->')
    lines.append('    <owl:AnnotationProperty rdf:about="http://purl.org/dc/elements/1.1/title"/>')
    lines.append('    <owl:AnnotationProperty rdf:about="http://purl.org/dc/elements/1.1/description"/>')
    lines.append('    <owl:AnnotationProperty rdf:about="http://purl.org/dc/elements/1.1/creator"/>')
    lines.append('    <owl:AnnotationProperty rdf:about="http://purl.org/dc/elements/1.1/date"/>')
    lines.append('    <owl:AnnotationProperty rdf:about="http://www.w3.org/2004/02/skos/core#definition"/>')
    lines.append('    <owl:AnnotationProperty rdf:about="http://www.w3.org/2004/02/skos/core#altLabel"/>')
    lines.append(f'    <owl:AnnotationProperty rdf:about="{base_uri}#sourceProduct">')
    lines.append('        <rdfs:comment>The product(s) from which this concept originated</rdfs:comment>')
    lines.append('    </owl:AnnotationProperty>')
    lines.append('')

    # Object properties
    lines.append('    <!-- Object Properties -->')
    for prop_name, domain, range_, comment in OBJECT_PROPERTIES:
        lines.append(f'    <owl:ObjectProperty rdf:about="{base_uri}#{prop_name}">')
        lines.append(f'        <rdfs:label>{prop_name}</rdfs:label>')
        lines.append(f'        <rdfs:comment>{escape_xml(comment)}</rdfs:comment>')
        lines.append(f'        <rdfs:domain rdf:resource="{base_uri}#{domain}"/>')
        lines.append(f'        <rdfs:range rdf:resource="{base_uri}#{range_}"/>')
        lines.append('    </owl:ObjectProperty>')
    lines.append('')

    # Data properties
    lines.append('    <!-- Data Properties -->')
    for prop_name, datatype, comment in DATA_PROPERTIES:
        lines.append(f'    <owl:DatatypeProperty rdf:about="{base_uri}#{prop_name}">')
        lines.append(f'        <rdfs:label>{prop_name}</rdfs:label>')
        lines.append(f'        <rdfs:comment>{escape_xml(comment)}</rdfs:comment>')
        lines.append(f'        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#{datatype}"/>')
        lines.append('    </owl:DatatypeProperty>')
    lines.append('')

    # Classes - sort by hierarchy level then alphabetically
    lines.append('    <!-- Classes -->')

    # Build hierarchy for sorting
    from collections import defaultdict
    children = defaultdict(list)
    for cls in classes.values():
        children[cls.parent].append(cls.name)

    def get_level(name, level=0):
        cls = classes.get(name)
        if not cls or not cls.parent:
            return level
        return get_level(cls.parent, level + 1)

    sorted_classes = sorted(
        classes.values(),
        key=lambda c: (get_level(c.name), c.name)
    )

    for cls in sorted_classes:
        lines.append(f'    <owl:Class rdf:about="{base_uri}#{cls.name}">')
        lines.append(f'        <rdfs:label>{escape_xml(cls.label)}</rdfs:label>')

        if cls.parent:
            lines.append(f'        <rdfs:subClassOf rdf:resource="{base_uri}#{cls.parent}"/>')

        if cls.definition:
            lines.append(f'        <skos:definition>{escape_xml(cls.definition)}</skos:definition>')

        for alt in cls.alt_labels:
            lines.append(f'        <skos:altLabel>{escape_xml(alt)}</skos:altLabel>')

        for product in cls.source_products:
            lines.append(f'        <itops:sourceProduct>{escape_xml(product)}</itops:sourceProduct>')

        lines.append('    </owl:Class>')

    lines.append('')
    lines.append('</rdf:RDF>')

    return '\n'.join(lines)


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))


def main():
    parser = argparse.ArgumentParser(
        description="Generate OWL ontology from glossary terms"
    )
    parser.add_argument(
        "--input", "-i",
        default="concepts/output/glossary/terms.json",
        help="Input terms.json file"
    )
    parser.add_argument(
        "--output", "-o",
        default="ontology/itops-ontology.owl",
        help="Output OWL file"
    )
    parser.add_argument(
        "--base-uri",
        default="http://ai4graphs.vpc.cloud9.ibm.com/ontology",
        help="Base URI for the ontology"
    )

    args = parser.parse_args()

    # Load terms
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    with open(input_path) as f:
        data = json.load(f)

    terms = data.get('terms', [])
    print(f"Loaded {len(terms)} terms from {input_path}")

    # Generate ontology
    output_path = Path(args.output)
    generate_owl(terms, output_path, args.base_uri)

    return 0


if __name__ == "__main__":
    exit(main())
