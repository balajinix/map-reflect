# Solis GraphQL Schemas

This directory contains the official Solis GraphQL schemas for IBM's IT Operations (ITOps) product suite. These are production-ready schemas sourced from the `solis-graphql-schemas` repository.

---

## Source Repository

- **Source:** `github.ibm.com/solis-dal/solis-graphql-schemas`
- **Branch:** `dev/products/`
- **Extraction Date:** 2026-01-11

---

## Products Overview

| Product | Files | Lines | Description |
|---------|-------|-------|-------------|
| [Apptio/Cloudability](#apptio-cloudability) | 37 | 6,470 | Cloud cost management & FinOps |
| [Concert](#concert) | 15 | 18,017 | Security & vulnerability management |
| [Instana](#instana) | 19 | 44,643 | Application performance monitoring |
| [NS1](#ns1) | 6 | 571 | DNS & traffic management |
| [Turbonomic](#turbo) | 7 | 1,526 | Application resource management |
| [Watson Orchestrate](#watsonx-orchestrate) | 9 | 1,026 | AI-powered workflow automation |

**Total: 93 files, 72,253 lines**

---

## Schema Format

These schemas use **StepZen GraphQL** format with custom directives:

### Custom Directives

| Directive | Purpose |
|-----------|---------|
| `@sdl` | Schema Definition Language - references other files to include |
| `@rest` | Maps GraphQL fields to REST API endpoints |
| `@sequence` | Defines multi-step query execution (e.g., auth then data) |
| `@tool` | Defines LLM tool metadata for AI integrations |
| `@graphql` | References other GraphQL endpoints |

### Schema Structure

Each product uses `index.graphql` files to define composition:

```graphql
schema
  @sdl(
    files: [
      "entities/entities.graphql"
      "markets/markets.graphql"
    ]
    visibility: { expose: true, types: "Query|Mutation", fields: "[^_].*" }
  ) {
  query: Query
}
```

The `visibility` directive controls which types/fields are exposed in the public API (fields starting with `_` are internal).

---

## Apptio (Cloudability)

**Directory:** `apptio/`
**Files:** 37
**Lines:** 6,470
**Product Type:** Cloud Cost Management & FinOps

### Description
IBM Cloudability (formerly Apptio) provides cloud cost optimization, budget management, and FinOps capabilities.

### Structure
```
apptio/
└── cloudability/
    ├── index.graphql              # Main schema composition
    ├── budgets/                   # Budget tracking
    │   ├── index.graphql
    │   └── op-budgets.graphql
    ├── clusters/                  # Kubernetes cluster costs
    │   ├── index.graphql
    │   └── op-clusters.graphql
    ├── recommendations/           # Cost optimization recommendations
    │   ├── index.graphql
    │   └── op-recommendations.graphql
    ├── reporting/                 # Cost reporting & analytics
    │   ├── atom.graphql
    │   ├── backend_api.graphql
    │   ├── llm_*.graphql          # LLM-friendly schema definitions
    │   ├── build/                 # Generated schemas
    │   └── operations/            # Sample queries by use case
    └── operations/                # Debug & utility operations
```

### Key Capabilities
- **Budget Management:** Track spending against budgets
- **Cluster Analysis:** Container/Kubernetes cost attribution
- **EC2 Recommendations:** Rightsizing and reservation recommendations
- **Cost Reporting:** Flexible reporting with LLM-friendly interfaces
- **Tag Analysis:** Cost allocation by tags

---

## Concert

**Directory:** `concert/`
**Files:** 15
**Lines:** 18,017
**Product Type:** Security & Vulnerability Management

### Description
IBM Concert provides unified security posture management, vulnerability tracking, and application security analysis.

### Structure
```
concert/
├── openapi/
│   ├── schema/
│   │   ├── index.graphql          # Main schema composition
│   │   ├── common.graphql         # Shared types
│   │   ├── applications/          # Application inventory
│   │   ├── vulnerabilities/       # CVE tracking
│   │   └── access_management/     # Auth & permissions
│   └── build/
│       ├── genai-schema/          # GenAI-optimized schemas
│       └── openapi-components/    # Generated from OpenAPI
└── postgres/
    ├── schema/                    # Direct DB schema
    └── build/                     # Generated schemas
```

### Key Capabilities
- **Vulnerability Management:** CVE tracking, severity scoring, remediation
- **Application Security:** SBOM analysis, package dependencies
- **Access Management:** SSO, API keys, permissions
- **GenAI Integration:** LLM-friendly schemas for AI assistants

---

## Instana

**Directory:** `instana/`
**Files:** 19
**Lines:** 44,643
**Product Type:** Application Performance Monitoring (APM)

### Description
Instana provides real-time application performance monitoring, distributed tracing, and infrastructure observability.

### Structure
```
instana/
├── openapi/
│   ├── schema/
│   │   ├── index.graphql          # Main schema composition
│   │   ├── common.graphql         # Shared types & auth
│   │   ├── applications/          # Application monitoring
│   │   │   ├── atom.graphql       # Core types
│   │   │   ├── applications.graphql
│   │   │   ├── services.graphql
│   │   │   ├── endpoints.graphql
│   │   │   └── filter.graphql
│   │   └── infrastructure/        # Infrastructure monitoring
│   │       ├── atom.graphql
│   │       ├── topology.graphql
│   │       └── analyze.graphql
│   ├── build/                     # Generated schemas
│   └── demo_schema/               # Demo/example schemas
└── events/
    ├── schema/                    # Event handling
    │   ├── events.graphql
    │   └── atom.graphql
    └── test/                      # Test deployments
```

### Key Capabilities
- **Application Monitoring:** Services, endpoints, calls, errors
- **Infrastructure Monitoring:** Hosts, containers, topology
- **Event Management:** Alerts, incidents, issues
- **Topology Analysis:** Service maps, dependencies

---

## NS1

**Directory:** `ns1/`
**Files:** 6
**Lines:** 571
**Product Type:** DNS & Traffic Management

### Description
IBM NS1 provides intelligent DNS and traffic management services.

### Structure
```
ns1/
├── index.graphql                  # Main schema composition
└── pulsar/                        # Pulsar traffic steering
    ├── index.graphql
    ├── apps.graphql               # Application definitions
    ├── jobs.graphql               # Job management
    ├── operations.graphql         # API operations
    └── performance.graphql        # Performance metrics
```

### Key Capabilities
- **Pulsar Traffic Steering:** Intelligent traffic routing
- **Performance Monitoring:** Latency, availability metrics
- **Job Management:** DNS configuration jobs

---

## Turbo (Turbonomic)

**Directory:** `turbo/`
**Files:** 7
**Lines:** 1,526
**Product Type:** Application Resource Management (ARM)

### Description
Turbonomic provides AI-driven resource optimization for hybrid cloud environments.

### Structure
```
turbo/
├── index.graphql                  # Main schema composition
├── common.graphql                 # Shared types & auth
├── types.graphql                  # Core type definitions
├── extendTypes.graphql            # Type extensions
├── entities/
│   └── entities.graphql           # Entity management (VMs, hosts, etc.)
├── markets/
│   └── markets.graphql            # Economic market simulation
└── search/
    └── search.graphql             # Entity search
```

### Key Capabilities
- **Entity Management:** VMs, containers, hosts, storage
- **Action Recommendations:** Resize, migrate, scale
- **Market Analysis:** Economic simulation for optimization
- **Search:** Flexible entity queries

---

## Watson Orchestrate

**Directory:** `watsonx-orchestrate/`
**Files:** 9
**Lines:** 1,026
**Product Type:** AI-powered Workflow Automation

### Description
IBM Watson Orchestrate (WXO) is an AI-powered automation platform for business workflows.

### Structure
```
watsonx-orchestrate/
├── index.graphql                  # Main schema composition
├── chats/
│   └── index.graphql              # Chat/conversation management
├── default-agent-settings/
│   ├── index.graphql
│   └── op-default-agent-settings.graphql
├── skillset-by-id/
│   ├── index.graphql
│   └── op-skillset-by-id.graphql
├── skillsets/
│   ├── index.graphql
│   └── op-skillsets.graphql
├── threads/
│   └── index.graphql              # Conversation threads
└── build/                         # Generated schemas
```

### Key Capabilities
- **Chat Execution:** Run conversations with AI agents
- **Skill Management:** Define and manage automation skills
- **Thread Management:** Maintain conversation context
- **Agent Settings:** Configure AI agent behavior

---

## Usage Notes

### StepZen Deployment

These schemas are designed for StepZen GraphQL federation. To deploy:

```bash
# Install StepZen CLI
npm install -g stepzen

# Deploy a product schema
cd instana/openapi/schema
stepzen deploy
```

### Schema Composition

The `index.graphql` files define how schemas are composed:

1. **@sdl directive** - Lists files to include
2. **visibility** - Controls public API exposure
3. **@tool directive** - Defines LLM tool metadata

### Authentication

Most schemas include authentication sequences using `@sequence`:

```graphql
type Query {
  entity(entity_uuid: ID): Entity
    @sequence(
      steps: [
        { query: "_accessToken" }
        { query: "_entity", arguments: [...] }
      ]
    )
}
```

The `_accessToken` query fetches credentials before the main query.

### Internal vs Public Fields

- Fields starting with `_` (underscore) are **internal** implementation details
- Public fields are exposed via the `visibility` directive
- Example: `_entity` is internal, `entity` is public

---

## Differences from Previous Source

This replaces the schemas previously extracted from `graphql-schema-mapping`:

| Product | Previous (graphql-schema-mapping) | Current (solis-graphql-schemas) |
|---------|-----------------------------------|--------------------------------|
| Apptio | 3 files, 224 lines | **37 files, 6,470 lines** |
| Concert | 39 files, 1,390 lines | **15 files, 18,017 lines** |
| Instana | 17 files, 4,955 lines | **19 files, 44,643 lines** |
| Turbo | 61 files, 7,775 lines | 7 files, 1,526 lines |
| WXO | 13 files, 430 lines | **9 files, 1,026 lines** |
| NS1 | (not present) | **6 files, 571 lines** |

Key improvements:
- **Apptio:** 29x more coverage with full reporting capabilities
- **Concert:** 13x more coverage with genai-schema
- **Instana:** 9x more coverage with full APM capabilities
- **NS1:** New product added
- **Production-ready:** These are actual deployed schemas

Note: Turbonomic has fewer lines because the previous source included recovered deleted files with extensive (but possibly outdated) schema definitions.

---

## Related Resources

- **Source Repository:** `github.ibm.com/solis-dal/solis-graphql-schemas`
- **StepZen Documentation:** https://stepzen.com/docs
- **Federation:** `dev/federation/` directory contains federated schemas combining multiple products
