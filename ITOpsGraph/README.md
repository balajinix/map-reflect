# ITOpsGraph

A knowledge graph for IT Operations, integrating schema information from IBM ITOps products with documentation, glossaries, and semantic mappings.

## Components

### Graph (Neo4j)

Schema graph with types and fields from 12 products:
- Concert, Instana, Turbonomic, Cloudability, NS1, watsonx Orchestrate
- Terraform, Packer
- AWS, Azure, GCP

**Nodes:** Product, Type, Field, GlossaryTerm
**Edges:** HAS_TYPE, HAS_FIELD, TYPE_MAPPING, FIELD_MAPPING

**Access:** http://ai4graphs.vpc.cloud9.ibm.com:7474

### Wiki (MediaWiki)

Auto-generated documentation for all schema types with field tables, schema snippets, and related type mappings.

**Access:** http://ai4graphs.vpc.cloud9.ibm.com/wiki

### Glossary

442 curated terms extracted from product documentation (Instana: 64, Turbonomic: 114, Concert: 28, Terraform: 75, etc.)

**Source:** `concepts/output/glossary/terms.json`

### Ontology

OWL ontology with 64 classes for ITOps domain concepts.

**Hierarchy:**
- Infrastructure (Host, Container, Pod, Cluster, Network, Storage)
- Observable (Event, Metric, Trace, Log, SLO)
- Workload (Application, Service, Job, Process)
- Operation (Action, Automation, Policy)
- Agent, Resource, CostEntity, Organization, Integration, Deployment

**Visualization:** http://ai4graphs.vpc.cloud9.ibm.com/itops-ontology.html

**Namespace:** `http://ai4graphs.vpc.cloud9.ibm.com/ontology#`

## Links

- Wiki: http://ai4graphs.vpc.cloud9.ibm.com/wiki
- Neo4j: http://ai4graphs.vpc.cloud9.ibm.com:7474
- Ontology: http://ai4graphs.vpc.cloud9.ibm.com/itops-ontology.html
- GitHub: https://github.ibm.com/ITGraph/ITOpsGraph
