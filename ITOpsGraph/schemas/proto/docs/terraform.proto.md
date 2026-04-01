# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                       | Description |
|--------------------------------------------------------------|-------------|
| hashicorp/cloud/internal/infragraph/domain/v1/base.proto     |             |
| hashicorp/cloud/internal/infragraph/domain/v1/edge.proto     |             |
| hashicorp/cloud/internal/infragraph/domain/v1/hcp.proto      |             |
| hashicorp/cloud/internal/infragraph/options/v1/options.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |




### HcpOrganization Diagram

```mermaid
classDiagram
direction LR

%% HcpOrganization is a type containing HCP TF or TFE organization specific fields.

class HcpOrganization {
  + BaseOrganization organization
}
HcpOrganization --> `BaseOrganization`

```
### HcpProject Diagram

```mermaid
classDiagram
direction LR

%% HcpProject is a type containing HCP TF or TFE Project specific fields.

class HcpProject {
  + BaseAccount account
}
HcpProject --> `BaseAccount`

```
### Workspace Diagram

```mermaid
classDiagram
direction LR

%% Workspace is a type containing Terraform Workspace specific fields.

class Workspace {
  + Base base
  + Optional~string~ current_run_id
  + Optional~string~ current_state_version_id
}
Workspace --> `Base`

```
### StateVersion Diagram

```mermaid
classDiagram
direction LR

%% StateVersion is a type containing HCP State Version specific fields.

class StateVersion {
  + Base base
  + Optional~string~ status
  + Optional~string~ hosted_state_download_url
  + List~Edge~ edge
}
StateVersion --> `Base`
StateVersion --> `Edge`

```

## Message: HcpOrganization
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpOrganization</div>

<div class="comment"><span>HcpOrganization is a type containing HCP TF or TFE organization specific fields.</span><br/></div>

| Field        | Ordinal | Type             | Label | Description |
|--------------|---------|------------------|-------|-------------|
| organization | 1       | BaseOrganization |       |             |




## Message: HcpProject
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpProject</div>

<div class="comment"><span>HcpProject is a type containing HCP TF or TFE Project specific fields.</span><br/></div>

| Field   | Ordinal | Type        | Label | Description |
|---------|---------|-------------|-------|-------------|
| account | 1       | BaseAccount |       |             |




## Message: Workspace
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.Workspace</div>

<div class="comment"><span>Workspace is a type containing Terraform Workspace specific fields.</span><br/></div>

| Field                    | Ordinal | Type   | Label    | Description                                    |
|--------------------------|---------|--------|----------|------------------------------------------------|
| base                     | 1       | Base   |          | project_id is mapped to Account ID in base     |
| current_run_id           | 2       | string | Optional |                                                |
| current_state_version_id | 3       | string | Optional | the current state version id of the workspace  |




## Message: StateVersion
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.StateVersion</div>

<div class="comment"><span>StateVersion is a type containing HCP State Version specific fields.</span><br/></div>

| Field                     | Ordinal | Type   | Label    | Description |
|---------------------------|---------|--------|----------|-------------|
| base                      | 1       | Base   |          |             |
| status                    | 2       | string | Optional |             |
| hosted_state_download_url | 3       | string | Optional |             |
| edge                      | 4       | Edge   | Repeated |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
