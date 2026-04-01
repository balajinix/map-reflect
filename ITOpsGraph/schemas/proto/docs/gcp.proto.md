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




### GcpComputeInstance Diagram

```mermaid
classDiagram
direction LR

%% GcpComputeInstance domain model message.

class GcpComputeInstance {
  + BaseVirtualMachine virtual_machine
}
GcpComputeInstance --> `BaseVirtualMachine`

```

## Message: GcpComputeInstance
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.GcpComputeInstance</div>

<div class="comment"><span>GcpComputeInstance domain model message.</span><br/></div>

| Field           | Ordinal | Type               | Label | Description |
|-----------------|---------|--------------------|-------|-------------|
| virtual_machine | 1       | BaseVirtualMachine |       |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
