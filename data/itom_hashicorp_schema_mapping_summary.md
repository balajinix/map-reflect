# ITOM–HashiCorp Schema Mapping: Entity Extraction Summary

> Source: ITOM_Workshop_-_Data_Model.pptx (Ehningen Workshop)
> Scope: Instana, Concert, Turbonomic, HCP Terraform + shared Core Infrastructure layer
> Purpose: Foundation for Map-Reflect cross-product schema mapping (ITOM GraphQL ↔ HCP Protobuf)

---

## 1. Entity Count per Product

| Product               | Entities | Notes                                                           |
|-----------------------|----------|-----------------------------------------------------------------|
| instana               | 8        | Application, Service, Latency, Throughput, Synthetics, Logs, Metrics, Traces |
| concert               | 15       | Application, Environment, AccessPoints, SourceCode, CodeRepos, CICDPipelines, BuildArtifact, CVEVulnerability, ITSMData, IncidentData, Patch, ResilienceMetrics, ResilienceProfile, ComplianceApplication, ComplianceReports |
| turbonomic            | 5        | TurboApplication, ResourceUsages, ServiceMetrics, OptimizationPolicy, OptimizationAction |
| hcp_terraform         | 1        | TerraformTemplate (governs core infra entities via IaC)        |
| core_infrastructure   | 10       | ContainerImage, ContainerWorkload, Service, Cluster, Node, Pod, Namespace, Deployment, Volume, AccessPoint |
| **Total**             | **39**   |                                                                 |

The `core_infrastructure` product is the shared Common Data Layer (CDL) — not a standalone product but the stitching layer enabling cross-product graph traversal.

---

## 2. Cross-Product Semantically Equivalent Entity Pairs

These pairs are candidates for schema alignment in Map-Reflect and should be treated as the primary mapping anchors when generating LLM-induced reflection alignments.

| Pair                                                         | Equivalence Type   | Notes                                                                                                  |
|--------------------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------|
| `concert.Application` ≈ `instana.Application`               | Semantic equivalent | Both represent an "application" grouping services. Concert is governance-scoped; Instana is observability-scoped. |
| `concert.Application` ≈ `turbonomic.TurboApplication`       | Semantic equivalent | Same application at different product layers — Concert owns policy/compliance, Turbo owns optimization. |
| `instana.Application` ≈ `turbonomic.TurboApplication`       | Semantic equivalent | Both are product-specific application views over the same `core_infrastructure.Service` entities.      |
| `instana.Service` ≈ `core_infrastructure.Service`           | Projection          | Instana surfaces the core Service entity with observability attributes.                                |
| `turbonomic.ServiceMetrics` ≈ `instana.Metrics`             | Semantic equivalent | Both capture service-level performance data; differ by consumer (optimization vs. observability).      |
| `concert.ITSMData` + `concert.IncidentData` ≈ core ITSM node | Structural split   | The slides show a single "ITSM/Incident/Change Data" node in Concert; split for schema mapping fidelity. |
| `concert.BuildArtifact` ≈ `core_infrastructure.ContainerImage` | Supply-chain link | Build artifacts from Concert's supply chain are packaged into container images in the core layer.      |
| `concert.AccessPoints` ≈ `core_infrastructure.AccessPoint`  | Projection          | Concert's environment-scoped access points map to the core infrastructure access point entity.         |
| `concert.ResilienceMetrics` ≈ `instana.Metrics` + `instana.Traces` | Derived aggregate | Concert computes resilience scores by consuming Instana's raw observability outputs.                   |
| `hcp_terraform.TerraformTemplate` ≈ `core_infrastructure.*` (Cluster, Node, Workload, Service) | IaC governance    | Terraform provisions and governs the core infrastructure entities observed by ITOM products.            |

---

## 3. Entities Appearing in Multiple Products Under Different Names

| Canonical Concept      | Instana          | Concert                   | Turbonomic           | HCP Terraform        | Core Layer                         |
|------------------------|------------------|---------------------------|----------------------|----------------------|------------------------------------|
| Application            | `Application`    | `Application`             | `TurboApplication`   | —                    | —                                  |
| Service                | `Service`        | —                         | —                    | —                    | `Service`                          |
| Container Image        | —                | `BuildArtifact` (packaged)| —                    | `TerraformTemplate` (references) | `ContainerImage`          |
| Observability Data     | `Metrics`, `Traces`, `Logs` | `ResilienceMetrics` (derived) | `ResourceUsages`, `ServiceMetrics` | — | —                    |
| Access Endpoint        | —                | `AccessPoints`            | —                    | governs via template | `AccessPoint`                      |
| Workload               | implicit via `Service` | —                   | via `OptimizationAction` | governs via template | `ContainerWorkload`, `Pod`, `Deployment` |
| Vulnerability          | —                | `CVEVulnerability`, `Patch` | —                  | —                    | — (Concert-owned)                  |
| ITSM / Change Record   | —                | `ITSMData`, `IncidentData`| —                    | —                    | — (Concert-owned)                  |
| Optimization           | —                | —                         | `OptimizationPolicy`, `OptimizationAction` | — | —                  |
| Compliance             | —                | `ComplianceApplication`, `ComplianceReports` | — | —             | —                                  |

---

## 4. Cross-Product Relationship Graph (Core Traversal Paths)

The following traversal chains are extracted directly from the use case flow diagrams (slides 10, 12, 13, 14, 35, 36, 37) and represent the primary API call sequences for schema mapping.

### Use Case 1: Application Vulnerability Discovery
```
instana.Application
  → (traverses via core) → core.Service
  → core.ContainerWorkload
  → core.ContainerImage
  → concert.CVEVulnerability
```
API surface: Instana GraphQL → Core GraphQL → Concert GraphQL

### Use Case 2: Application-Scoped RBAC
```
instana.Application
  → core.Service            → turbonomic (apply scope on services)
  → core.ContainerImage     → concert    (apply scope on images)
```
API surface: Instana GraphQL → Core GraphQL → Turbonomic GraphQL + Concert GraphQL

### Use Case 3: Observability-Enhanced Resilience
```
concert.Application
  → concert.BuildArtifact / concert.Environment
  → core.ContainerImage
  → core.ContainerWorkload / core.Service
  → instana.Metrics + instana.Traces
  → concert.ResilienceMetrics
  → concert.ResilienceProfile
```
API surface: Concert GraphQL → Core GraphQL → Instana GraphQL → Concert GraphQL (write)

---

## 5. ITOM–HashiCorp Schema Mapping Architecture

The ITOM → HashiCorp mapping is anchored through:

```
hcp_terraform.TerraformTemplate  (Protobuf API)
  --governs-->  core_infrastructure.Cluster
  --governs-->  core_infrastructure.Node
  --governs-->  core_infrastructure.ContainerWorkload
  --governs-->  core_infrastructure.Service
                    ^             ^              ^
                Instana       Turbonomic       Concert
              (GraphQL)       (GraphQL)       (GraphQL)
```

### API Format Boundary

| Side        | Format   | Key entities exposed                                     |
|-------------|----------|----------------------------------------------------------|
| ITOM (IBM)  | GraphQL  | Application, Service, Metrics, Traces, CVE, ResilienceProfile, OptimizationAction |
| HashiCorp   | Protobuf | Workspace, Run, StateVersion, Module, Variable, ResourceInstance |

### Proposed Mapping Anchors for Map-Reflect

| HCP Terraform (Protobuf)   | ITOM Core / Product entity        | Mapping confidence |
|----------------------------|------------------------------------|--------------------|
| `Workspace`                | `hcp_terraform.TerraformTemplate`  | Direct (1:1)       |
| `ResourceInstance`         | `core_infrastructure.Cluster` / `core_infrastructure.Node` | High |
| `ResourceInstance` (pod)   | `core_infrastructure.ContainerWorkload` | High          |
| `ResourceInstance` (svc)   | `core_infrastructure.Service`      | High               |
| `StateVersion`             | Snapshot of `TerraformTemplate` at a point in time | Medium |
| `Run`                      | Trigger event → `concert.CICDPipelines` | Medium        |
| `Module`                   | Sub-graph of `TerraformTemplate`   | Medium             |
| `Variable`                 | Attribute-level mapping (no direct entity) | Low        |

---

## 6. Notes for Map-Reflect Paper

- The `core_infrastructure` layer is the **schema mapping bridge**: it is the only layer both ITOM products (as consumers) and Terraform (as provisioner) reference simultaneously.
- Cross-product schema mapping should be modeled as a **two-hop alignment**: HCP Protobuf → Core GraphQL Schema → Product GraphQL Schema, rather than direct p2p mapping.
- `concert.ResilienceMetrics` is the richest cross-product entity, drawing from Instana, ITSM, and Concert-internal data — it is the best candidate for demonstrating multi-source schema alignment in Map-Reflect evaluation.
- `instana.Application`, `concert.Application`, and `turbonomic.TurboApplication` are the highest-priority semantic equivalent triple for the paper's equivalence detection evaluation since they represent the same real-world concept (a business application) expressed through three different GraphQL schemas with distinct attribute sets and relationship topologies.
