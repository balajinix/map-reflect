# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                    | Description |
|-----------------------------------------------------------|-------------|
| google/protobuf/timestamp.proto                           |             |
| hashicorp/cloud/internal/infragraph/domain/v1/model.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |




### Resource Diagram

```mermaid
classDiagram
direction LR

%% 

class Resource {
  + google.protobuf.Timestamp witnessed_time
  + Model model
}
Resource --> `google.protobuf.Timestamp`
Resource --> `Model`

```

## Message: Resource
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.Resource</div>

<div class="comment"><span></span><br/></div>

| Field          | Ordinal | Type                      | Label | Description                                                                                                                                                           |
|----------------|---------|---------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| witnessed_time | 1       | google.protobuf.Timestamp |       | witnessed_time is the time when this resource was last witnessed by the connector. The same clock should be used universally within the scope of a single connector.  |
| model          | 2       | Model                     |       | model is the domain model that represents this resource.                                                                                                              |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
