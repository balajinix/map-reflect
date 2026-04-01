# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                       | Description |
|--------------------------------------------------------------|-------------|
| hashicorp/cloud/internal/infragraph/domain/v1/base.proto     |             |
| hashicorp/cloud/internal/infragraph/options/v1/options.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |




### HcpResource Diagram

```mermaid
classDiagram
direction LR

%% HcpResource is a type containing HCP specific fields.

class HcpResource {
  + string organization_id
  + string project_id
}

```
### HcpOrganization Diagram

```mermaid
classDiagram
direction LR

%% HcpOrganization is a type containing HCP Organization specific fields.

class HcpOrganization {
  + BaseOrganization organization
}
HcpOrganization --> `BaseOrganization`

```
### HcpProject Diagram

```mermaid
classDiagram
direction LR

%% HcpProject is a type containing HCP Project specific fields.

class HcpProject {
  + BaseAccount account
}
HcpProject --> `BaseAccount`

```

## Message: HcpResource
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpResource</div>

<div class="comment"><span>HcpResource is a type containing HCP specific fields.</span><br/></div>

| Field           | Ordinal | Type   | Label | Description                             |
|-----------------|---------|--------|-------|-----------------------------------------|
| organization_id | 1       | string |       | organization_id that owns the resource  |
| project_id      | 2       | string |       | project_id the resource belongs to      |




## Message: HcpOrganization
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpOrganization</div>

<div class="comment"><span>HcpOrganization is a type containing HCP Organization specific fields.</span><br/></div>

| Field        | Ordinal | Type             | Label | Description |
|--------------|---------|------------------|-------|-------------|
| organization | 1       | BaseOrganization |       |             |




## Message: HcpProject
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.HcpProject</div>

<div class="comment"><span>HcpProject is a type containing HCP Project specific fields.</span><br/></div>

| Field   | Ordinal | Type        | Label | Description |
|---------|---------|-------------|-------|-------------|
| account | 1       | BaseAccount |       |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
