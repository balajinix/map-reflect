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




### AzureResource Diagram

```mermaid
classDiagram
direction LR

%% AzureResource is a type containing Azure specific fields for all resources.

class AzureResource {
  + string etag
  + string provisioning_state
}

```
### AzureVirtualMachine Diagram

```mermaid
classDiagram
direction LR

%% AzureVirtualMachine domain model message.

class AzureVirtualMachine {
  + BaseVirtualMachine virtual_machine
  + AzureResource azure_resource
  + Optional~string~ priority
  + Optional~string~ extensions_time_budget
}
AzureVirtualMachine --> `BaseVirtualMachine`
AzureVirtualMachine --> `AzureResource`

```

## Message: AzureResource
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.AzureResource</div>

<div class="comment"><span>AzureResource is a type containing Azure specific fields for all resources.</span><br/></div>

| Field              | Ordinal | Type   | Label | Description           |
|--------------------|---------|--------|-------|-----------------------|
| etag               | 1       | string |       | type specific fields  |
| provisioning_state | 2       | string |       |                       |




## Message: AzureVirtualMachine
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.AzureVirtualMachine</div>

<div class="comment"><span>AzureVirtualMachine domain model message.</span><br/></div>

| Field                  | Ordinal | Type               | Label    | Description           |
|------------------------|---------|--------------------|----------|-----------------------|
| virtual_machine        | 1       | BaseVirtualMachine |          | inherited types       |
| azure_resource         | 2       | AzureResource      |          |                       |
| priority               | 3       | string             | Optional | type specific fields  |
| extensions_time_budget | 4       | string             | Optional |                       |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
