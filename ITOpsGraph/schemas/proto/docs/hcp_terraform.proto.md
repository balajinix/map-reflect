# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                       | Description |
|--------------------------------------------------------------|-------------|
| hashicorp/cloud/internal/infragraph/domain/v1/base.proto     |             |
| hashicorp/cloud/internal/infragraph/domain/v1/edge.proto     |             |
| hashicorp/cloud/internal/infragraph/options/v1/options.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |




### HcpTerraformOrganization Diagram

```mermaid
classDiagram
direction LR

%% HcpTerraformOrganization is a type containing HCP TF Organization specific fields.

class HcpTerraformOrganization {
  + BaseOrganization organization
  + Optional~bool~ is_unified
  + Optional~string~ agent_pool_id
}
HcpTerraformOrganization --> `BaseOrganization`

```
### HcpTerraformProject Diagram

```mermaid
classDiagram
direction LR

%% HcpTerraformProject is a type containing HCP TF Project specific fields.

class HcpTerraformProject {
  + BaseAccount account
  + Optional~bool~ is_legacy_organization_project
  + Optional~bool~ is_unified
}
HcpTerraformProject --> `BaseAccount`

```
### HcpTerraformWorkspace Diagram

```mermaid
classDiagram
direction LR

%% Workspace is a type containing Terraform Workspace specific fields.

class HcpTerraformWorkspace {
  + Base base
  + Optional~string~ current_run_id
  + Optional~int32~ resource_count
  + Optional~string~ current_state_version_id
  + Optional~string~ environment
  + Optional~string~ execution_mode
  + List~string~ tag_names
  + Optional~string~ agent_pool_id
}
HcpTerraformWorkspace --> `Base`

```
### HcpTerraformStateVersion Diagram

```mermaid
classDiagram
direction LR

%% StateVersion is a type containing HCP State Version specific fields.

class HcpTerraformStateVersion {
  + Base base
  + Optional~string~ status
  + Optional~string~ hosted_state_download_url
  + List~Edge~ edge
  + Optional~string~ current_run_id
}
HcpTerraformStateVersion --> `Base`
HcpTerraformStateVersion --> `Edge`

```
### HcpTerraformAgentPool Diagram

```mermaid
classDiagram
direction LR

%% 

class HcpTerraformAgentPool {
  + Base base
  + Optional~int32~ agent_count
  + Optional~bool~ organization_scoped
}
HcpTerraformAgentPool --> `Base`

```
### HcpTerraformRun Diagram

```mermaid
classDiagram
direction LR

%% 

class HcpTerraformRun {
  + Base base
  + Optional~bool~ is_destroyed
  + Optional~string~ message
  + Optional~string~ status
  + Optional~string~ workspace_id
}
HcpTerraformRun --> `Base`

```

## Message: HcpTerraformOrganization
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformOrganization</div>

<div class="comment"><span>HcpTerraformOrganization is a type containing HCP TF Organization specific fields.</span><br/></div>

| Field         | Ordinal | Type             | Label    | Description |
|---------------|---------|------------------|----------|-------------|
| organization  | 1       | BaseOrganization |          |             |
| is_unified    | 2       | bool             | Optional |             |
| agent_pool_id | 3       | string           | Optional |             |




## Message: HcpTerraformProject
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformProject</div>

<div class="comment"><span>HcpTerraformProject is a type containing HCP TF Project specific fields.</span><br/></div>

| Field                          | Ordinal | Type        | Label    | Description |
|--------------------------------|---------|-------------|----------|-------------|
| account                        | 1       | BaseAccount |          |             |
| is_legacy_organization_project | 2       | bool        | Optional |             |
| is_unified                     | 3       | bool        | Optional |             |




## Message: HcpTerraformWorkspace
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformWorkspace</div>

<div class="comment"><span>Workspace is a type containing Terraform Workspace specific fields.</span><br/></div>

| Field                    | Ordinal | Type   | Label    | Description                                    |
|--------------------------|---------|--------|----------|------------------------------------------------|
| base                     | 1       | Base   |          |                                                |
| current_run_id           | 2       | string | Optional |                                                |
| resource_count           | 3       | int32  | Optional |                                                |
| current_state_version_id | 4       | string | Optional | the current state version id of the workspace  |
| environment              | 5       | string | Optional |                                                |
| execution_mode           | 6       | string | Optional |                                                |
| tag_names                | 7       | string | Repeated |                                                |
| agent_pool_id            | 8       | string | Optional |                                                |




## Message: HcpTerraformStateVersion
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformStateVersion</div>

<div class="comment"><span>StateVersion is a type containing HCP State Version specific fields.</span><br/></div>

| Field                     | Ordinal | Type   | Label    | Description |
|---------------------------|---------|--------|----------|-------------|
| base                      | 1       | Base   |          |             |
| status                    | 2       | string | Optional |             |
| hosted_state_download_url | 3       | string | Optional |             |
| edge                      | 4       | Edge   | Repeated |             |
| current_run_id            | 5       | string | Optional |             |




## Message: HcpTerraformAgentPool
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformAgentPool</div>

<div class="comment"><span></span><br/></div>

| Field               | Ordinal | Type  | Label    | Description |
|---------------------|---------|-------|----------|-------------|
| base                | 1       | Base  |          |             |
| agent_count         | 2       | int32 | Optional |             |
| organization_scoped | 3       | bool  | Optional |             |




## Message: HcpTerraformRun
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpTerraformRun</div>

<div class="comment"><span></span><br/></div>

| Field        | Ordinal | Type   | Label    | Description |
|--------------|---------|--------|----------|-------------|
| base         | 1       | Base   |          |             |
| is_destroyed | 2       | bool   | Optional |             |
| message      | 3       | string | Optional |             |
| status       | 4       | string | Optional |             |
| workspace_id | 5       | string | Optional |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
