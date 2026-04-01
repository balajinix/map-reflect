# Tag/Tagkey System Analysis

## Overview

The Cloudability API uses a tag-based system where resources can be tagged with key-value pairs. This system consists of:

- **`tagkeyN`** - The tag key/name (e.g., "team", "environment", "project")
- **`tagN`** - The tag value (e.g., "engineering", "production", "frontend")

Where N ranges from 1 to 50 (tagkey1-tagkey50 and tag1-tag50).

## Key Discovery: Tagkey/Tag Relationship

**Critical Understanding**: The `tagkeyN` and `tagN` fields are related and must be used together correctly.

### The Pattern

- When `tagkey21 = "team"`, then `tag21` contains the actual team value
- When `tagkey21 = "(not set)"`, then `tag21` is also "(not set)" and should be excluded from analysis

### Example from Data Analysis

```
tagkey21 = "team"     → tag21 = "flagship"     (Cost: $2,129,421.78)
tagkey21 = "team"     → tag21 = "cloudability" (Cost: $361,649.54)
tagkey21 = "team"     → tag21 = "suitex"       (Cost: $89,267.22)
tagkey21 = "(not set)" → tag21 = "(not set)"   (Cost: $2,756,096.02)
```

## Discovered Team-Related Fields

### Primary Team Tags

#### 1. `tagkey21` / `tag21` - "Team" (Most Active)

- **Purpose**: Primary team ownership tagging
- **Status**: **ACTIVELY USED** - Most comprehensive team data
- **Coverage**: ~$3.0M with team tags, ~$2.8M without
- **Common Values**: flagship, cloudability, suitex, pre, itp, cloud, techops, digitalfuel, pipeline, ci, apptiobi, pss, data-science, scotch, squad11, data, infosec, fjord, vedas, vanguard, snape, optimization, dos, ccm, datalake, nimbus, infosvc, tas, con, eng-and-product, ml-data, operations, dss, billmx, suitx, reco, compute, product, rightsizing, datalink, data science, wp, hackaton, datascience, bmx, foundation-ops, foundations-ops, cco, app-eng-r11

#### 2. `category24` - "Apptio Team" (Individual Names)

- **Purpose**: Individual ownership (not team names)
- **Status**: **ACTIVELY USED** - Shows individual names
- **Coverage**: 15 different individuals
- **Common Values**: matthew halls, ben zhou, andrey mihailenko, wayne gallegos, eric theriault, sasha jeltuhin, chris linder, sujit baliga, unknown, reserved instances, hackathon, sandbox accounts, pre-sales, eugene khvostov, govcloud us

#### 3. `tagkey27` / `tag27` - "account-level-team-dimension" (Azure Specific)

- **Purpose**: Azure subscription-level team tags
- **Status**: **LIMITED USE** - Only for Azure resources
- **Coverage**: ~$53K in Azure resources
- **Key Values**:
  - `tagkey27 = "cldy:azure:subscriptionleveltag:team"` → `tag27 = "cloudability", "cloudability containers", "cloudability data science", "snape", "azu"`
  - `tagkey27 = "dbr-type"` → `tag27 = "cldy", "cat"`

### Inactive Team Tags

#### 4. `tagkey2` / `tag2` - "Team (value/key)"

- **Status**: **NOT USED** - All values are "(not set)"

#### 5. `category7` - "Team Name (don't delete)"

- **Status**: **NOT USED** - All values are "na"

## Discovered Application-Related Fields

### Primary Application Tags

#### 1. `tagkey24` / `tag24` - "Application" (Most Active)

- **Purpose**: Application identification and ownership
- **Status**: **ACTIVELY USED** - Most comprehensive application data
- **Coverage**: ~$787K with application tags, ~$4.6M without
- **Key Patterns**:
  - `tagkey24 = "app"` → `tag24 = "biit"` (Cost: $787K) - Main application
  - `tagkey24 = "application"` → `tag24 = "rs-recs"` (Cost: $84K) - Recommendations service
  - `tagkey24 = "application"` → `tag24 = "cake-recs"` (Cost: $41K) - Cake recommendations
  - `tagkey24 = "application"` → `tag24 = "insights-hub"` (Cost: $17K) - Insights service
  - `tagkey24 = "app"` → `tag24 = "default"` (Cost: $12K) - Default application
  - `tagkey24 = "application"` → `tag24 = "datadrop"` (Cost: $8K) - Data service
  - `tagkey24 = "app"` → `tag24 = "moonlite-runner-stg"` (Cost: $6K) - Staging runner
  - `tagkey24 = "app"` → `tag24 = "kiam"` (Cost: $5K) - IAM service
  - `tagkey24 = "app"` → `tag24 = "platform-observability"` (Cost: $5K) - Observability
  - `tagkey24 = "application"` → `tag24 = "pipeline-worker"` (Cost: $5K) - Pipeline service

#### 2. `group_name3` - "Application" (Environment-like)

- **Purpose**: Application environment classification
- **Status**: **ACTIVELY USED** - Shows environment-like values
- **Coverage**: Multiple environment types
- **Common Values**: production, engineering, Finance, staging, (not set)
- **Note**: This appears to be more environment-focused than application-focused

#### 3. `category16` - "Apptio Project" (Customer Projects)

- **Purpose**: Customer project identification
- **Status**: **ACTIVELY USED** - Shows customer-specific projects
- **Coverage**: Multiple customer projects
- **Common Values**: not applicable, (not set), wellsfargo-tbm, nationalgrid, unilever-r12, socgen-reporting, hsbc-data, hsbc-r12, db-config, macquarie, bmo, macquarie-r12, db, boehringer, eon, intesasanpaolo, westpac, corning, filfundservices-r12, aexp-r12, metlife-r12, bktbm, commercebank, cs, natwestgroup, standardchartered, aldi-hofer

#### 4. `tagkey29` / `tag29` - "Project" (Limited Use)

- **Purpose**: Project identification
- **Status**: **LIMITED USE** - Only specific projects tagged
- **Coverage**: ~$89K with project tags
- **Key Values**:
  - `tagkey29 = "cldy:aws:accountleveltag:project"` → `tag29 = "networking"` (Cost: $89K)
  - `tagkey29 = "cldy:aws:accountleveltag:project"` → `tag29 = "ssa"` (Cost: $16K)

### Inactive Application Tags

#### 5. `tagkey32` / `tag32` - "Project"

- **Status**: **MINIMAL USE** - Only one value: `tagkey32 = "product"` → `tag32 = "cloudability"`

#### 6. `tagkey33` / `tag33` - "Project"

- **Status**: **MINIMAL USE** - Only matches tagkey29 values

#### 7. `tagkey49` / `tag49` - "Project"

- **Status**: **NOT USED** - All values are "(not set)"

## Discovered Environment-Related Fields

### Primary Environment Tags

#### 1. `tagkey20` / `tag20` - "environment" (Most Active)

- **Purpose**: Environment identification (prod, staging, dev, etc.)
- **Status**: **ACTIVELY USED** - Most comprehensive environment data
- **Coverage**: ~$5.5M with environment tags, ~$38K without
- **Key Patterns**:
  - **Production Environments**:
    - `tagkey20 = "cldy:aws:accountleveltag:environment"` → `tag20 = "production"` (Cost: $5.3M)
    - `tagkey20 = "environment"` → `tag20 = "production-us"` (Cost: $47K)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "production-us"` (Cost: $15K)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "production-au"` (Cost: $11K)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "production-eu"` (Cost: $9K)
  - **Engineering/Development**:
    - `tagkey20 = "cldy:aws:accountleveltag:environment"` → `tag20 = "engineering"` (Cost: $206K)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "engineering"` (Cost: $9K)
    - `tagkey20 = "environment"` → `tag20 = "engineering"` (Cost: $3K)
  - **Staging/Testing**:
    - `tagkey20 = "cldy:aws:accountleveltag:environment"` → `tag20 = "staging"` (Cost: $116K)
    - `tagkey20 = "environment"` → `tag20 = "staging"` (Cost: $308)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "staging"` (Cost: $15)
  - **Development**:
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "dev"` (Cost: $409)
    - `tagkey20 = "environment"` → `tag20 = "dev"` (Cost: $0.08)
  - **Testing**:
    - `tagkey20 = "environment"` → `tag20 = "pm-test"` (Cost: $3K)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "pm-test"` (Cost: $143)
    - `tagkey20 = "cldy:azure:resourcetag:environment"` → `tag20 = "dev-gov"` (Cost: $109)

#### 2. `group_name6` - "Environment"

- **Purpose**: Environment classification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 3. `category3` - "Environment List(BM)"

- **Purpose**: Environment list (Business Model)
- **Status**: **UNKNOWN** - Need to analyze separately

### Production-Specific Tags

#### 4. `tagkey19` / `tag19` - "Production"

- **Purpose**: Production environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 5. `tagkey37` / `tag37` - "prod"

- **Purpose**: Production environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 6. `tagkey41` / `tag41` - "Prod"

- **Purpose**: Production environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

### Environment Variants

#### 7. `tagkey40` / `tag40` - "env"

- **Purpose**: Environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 8. `tagkey43` / `tag43` - "env"

- **Purpose**: Environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 9. `tagkey45` / `tag45` - "Env"

- **Purpose**: Environment identification
- **Status**: **UNKNOWN** - Need to analyze separately

#### 10. `tagkey47` / `tag47` - "Environment-doNoReuseTagDimensionNames"

- **Purpose**: Environment identification (special naming)
- **Status**: **UNKNOWN** - Need to analyze separately

## Cross-Field Analysis Insights

### Individual vs Team Mapping

The data shows an interesting relationship between individual ownership (`category24`) and team ownership (`tag21`):

**Examples:**

- `category24 = "matthew halls"` → `tag21 = "flagship"` (Cost: $2.6M)
- `category24 = "ben zhou"` → `tag21 = "pipeline"` (Cost: $24.6K)
- `category24 = "sasha jeltuhin"` → `tag21 = "cloud"` (Cost: $33.4K)
- `category24 = "eric theriault"` → `tag21 = "flagship"` (Cost: $11.8K)

### Application vs Environment Mapping

The data shows relationships between applications and environments:

**Examples:**

- `tagkey24 = "app"` → `tag24 = "moonlite-runner-stg"` (Staging application)
- `tagkey24 = "app"` → `tag24 = "biit"` (Main application, various environments)
- `group_name3 = "production"` vs `group_name3 = "engineering"` (Environment classification)

### Multi-Cloud Environment Tagging

- **AWS**: Uses `tagkey20 = "cldy:aws:accountleveltag:environment"` with `tag20` values
- **Azure**: Uses `tagkey20 = "cldy:azure:resourcetag:environment"` with `tag20` values
- **Generic**: Uses `tagkey20 = "environment"` with `tag20` values

### Data Quality Patterns

1. **Inconsistent naming**: Same concept with variations (e.g., "data-science", "data science", "datascience")
2. **Individual vs Team**: Some resources tagged by individual, some by team, some by both
3. **Cross-cloud differences**: Azure uses different tagging system than AWS/GCP
4. **Application vs Project**: Clear distinction between internal applications and customer projects
5. **Environment granularity**: Production has regional variants (production-us, production-au, production-eu)

## Correct Usage Patterns

### ❌ Wrong Way

```graphql
# This will include resources without team tags
query CostByTeam {
  llmFriendlyCostReport(
    dimensions: [tag21]
    metrics: [unblended_cost]
    # Missing filter - will include "(not set)" values
  )
}
```

### ✅ Correct Way - Primary Team Tags

```graphql
# This filters for resources that actually have team tags
query CostByTeam {
  llmFriendlyCostReport(
    dimensions: [tag21]
    metrics: [unblended_cost]
    match: { tagkey21: { eq: "team" } } # Only include resources with team tags
  )
}
```

### ✅ Correct Way - Individual Ownership

```graphql
# Cost by individual owner
query CostByIndividual {
  llmFriendlyCostReport(
    dimensions: [category24]
    metrics: [unblended_cost]
    match: { category24: { ne: "unknown" } } # Exclude unknown owners
  )
}
```

### ✅ Correct Way - Azure Team Tags

```graphql
# Cost by Azure team tags
query CostByAzureTeam {
  llmFriendlyCostReport(
    dimensions: [tag27]
    metrics: [unblended_cost]
    match: { tagkey27: { eq: "cldy:azure:subscriptionleveltag:team" } }
  )
}
```

### ✅ Correct Way - Application Tags

```graphql
# Cost by application
query CostByApplication {
  llmFriendlyCostReport(
    dimensions: [tag24]
    metrics: [unblended_cost]
    match: { tagkey24: { in: ["app", "application"] } } # Filter for app tags
  )
}
```

### ✅ Correct Way - Environment Tags

```graphql
# Cost by environment
query CostByEnvironment {
  llmFriendlyCostReport(
    dimensions: [tag20]
    metrics: [unblended_cost]
    match: { tagkey20: { contains: "environment" } } # Filter for environment tags
  )
}
```

### ✅ Correct Way - Customer Projects

```graphql
# Cost by customer project
query CostByCustomerProject {
  llmFriendlyCostReport(
    dimensions: [category16]
    metrics: [unblended_cost]
    match: { category16: { ne: "not applicable" } } # Exclude internal projects
  )
}
```

## Helper Query Templates

### Cost by Team (Primary)

```graphql
query CostByTeam {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [tag21]
    metrics: [unblended_cost]
    match: { tagkey21: { eq: "team" } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      tag21
      unblended_cost
    }
  }
}
```

### Cost by Individual Owner

```graphql
query CostByIndividual {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [category24]
    metrics: [unblended_cost]
    match: { category24: { ne: "unknown" } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      category24
      unblended_cost
    }
  }
}
```

### Cost by Application

```graphql
query CostByApplication {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [tag24]
    metrics: [unblended_cost]
    match: { tagkey24: { in: ["app", "application"] } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      tag24
      unblended_cost
    }
  }
}
```

### Cost by Environment

```graphql
query CostByEnvironment {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [tag20]
    metrics: [unblended_cost]
    match: { tagkey20: { contains: "environment" } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      tag20
      unblended_cost
    }
  }
}
```

### Cost by Customer Project

```graphql
query CostByCustomerProject {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [category16]
    metrics: [unblended_cost]
    match: { category16: { ne: "not applicable" } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      category16
      unblended_cost
    }
  }
}
```

### Cross-Analysis: Individual + Team

```graphql
query CostByIndividualAndTeam {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [category24, tag21]
    metrics: [unblended_cost]
    match: { category24: { ne: "unknown" }, tagkey21: { eq: "team" } }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      category24
      tag21
      unblended_cost
    }
  }
}
```

### Cross-Analysis: Application + Environment

```graphql
query CostByApplicationAndEnvironment {
  llmFriendlyCostReport(
    dateRange: { start: "2024-01-01", end: "2024-01-31" }
    dimensions: [tag24, tag20]
    metrics: [unblended_cost]
    match: {
      tagkey24: { in: ["app", "application"] }
      tagkey20: { contains: "environment" }
    }
    sort: [{ field: "unblended_cost", direction: DESC }]
  ) {
    total_results
    results {
      tag24
      tag20
      unblended_cost
    }
  }
}
```

## Best Practices

1. **Always filter by tagkey**: When using a tag dimension, always filter by the corresponding tagkey to exclude "(not set)" values
2. **Use primary team tags**: `tagkey21 = "team"` is the most comprehensive team tagging system
3. **Consider individual ownership**: `category24` provides individual-level ownership data
4. **Handle multi-cloud differences**: Azure uses different tagging (`tagkey27`) than AWS/GCP (`tagkey21`)
5. **Validate data quality**: Large portions of resources may be untagged or inconsistently tagged
6. **Handle naming inconsistencies**: Same concept may have multiple spellings
7. **Distinguish applications from projects**: `tagkey24` for internal apps, `category16` for customer projects
8. **Use environment tags for cost allocation**: `tagkey20` provides comprehensive environment data
9. **Consider regional environments**: Production has regional variants (us, au, eu)
10. **Filter by environment type**: Use specific environment values (production, staging, engineering, dev)

## Data Quality Insights

### Tag Coverage

- **Most expensive resources** ($2.7M) have no team tags
- **Team-tagged resources** represent ~$3.0M in costs
- **Individual ownership** is more consistently applied than team ownership
- **Azure resources** use different tagging system than AWS/GCP
- **Application tags** cover ~$787K, with ~$4.6M untagged
- **Environment tags** cover ~$5.5M, with ~$38K untagged
- **Customer projects** are well-tagged with specific customer names

### Common Patterns

1. **Inconsistent naming**: Same concept with different spellings
2. **Missing tags**: Large portion of resources untagged
3. **Individual vs Team**: Some resources tagged by individual, some by team, some by both
4. **Cross-cloud differences**: Azure uses different tagging system than AWS/GCP
5. **Application vs Project**: Clear distinction between internal applications and customer projects
6. **Environment granularity**: Production has regional variants
7. **Multi-cloud environment tagging**: Different patterns for AWS vs Azure

## Future Analysis Needed

1. **Analyze remaining tagkeys**: Check tagkey1, tagkey3-tagkey19, tagkey22-tagkey23, tagkey25-tagkey26, tagkey28, tagkey30-tagkey50
2. **Cross-tagkey analysis**: See which tagkeys are used together
3. **Tagkey value discovery**: Find all unique tagkey values across the dataset
4. **Cost distribution**: Understand cost distribution across different tagkey combinations
5. **Environment tag analysis**: Analyze remaining environment-related tagkeys (tagkey19, tagkey37, tagkey40, tagkey41, tagkey43, tagkey45, tagkey47)
6. **Group name analysis**: Analyze group_name fields for additional categorization
7. **Category analysis**: Analyze remaining category fields for business categorization

## Analysis Queries Created

- `tagkey_frequency.graphql` - Shows frequency of tagkey values
- `tag_tagkey_relationship.graphql` - Shows relationship between tagkey and tag values
- `tagkey_analysis.graphql` - Shows multiple tagkeys together
- `tagkey21_analysis.graphql` - Specific analysis of team tags
- `category24_analysis.graphql` - Analysis of individual ownership
- `tag2_analysis.graphql` - Analysis of unused team tags
- `tag27_analysis.graphql` - Analysis of Azure team tags
- `team_analysis_comprehensive.graphql` - Cross-analysis of all team-related fields
- `tag24_application_analysis.graphql` - Analysis of application tags
- `tag20_environment_analysis.graphql` - Analysis of environment tags
- `application_analysis.graphql` - Comprehensive application field analysis

## Last Updated

- **Date**: June 2025
- **Data Period**: January 2024
- **Analysis Status**: Comprehensive analysis complete for team-related fields
