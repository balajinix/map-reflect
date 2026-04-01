# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                       | Description |
|--------------------------------------------------------------|-------------|
| google/protobuf/timestamp.proto                              |             |
| hashicorp/cloud/internal/infragraph/domain/v1/base.proto     |             |
| hashicorp/cloud/internal/infragraph/domain/v1/hcp.proto      |             |
| hashicorp/cloud/internal/infragraph/options/v1/options.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |




### PackerRegistry Diagram

```mermaid
classDiagram
direction LR

%% PackerRegistry domain model message.

class PackerRegistry {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ feature_tier
  + Optional~bool~ activated
  + Optional~int32~ billing_deprovision_reason
  + Optional~string~ billing_deprovision_message
}
PackerRegistry --> `Base`
PackerRegistry --> `HcpResource`

```
### PackerBucket Diagram

```mermaid
classDiagram
direction LR

%% PackerBucket domain model message.

class PackerBucket {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ registry_id
  + List~string~ platforms
}
PackerBucket --> `Base`
PackerBucket --> `HcpResource`

```
### PackerInheritedRevocation Diagram

```mermaid
classDiagram
direction LR

%% PackerInheritedRevocation domain model message.

class PackerInheritedRevocation {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ descendant_id
  + Optional~string~ ancestor_id
}
PackerInheritedRevocation --> `Base`
PackerInheritedRevocation --> `HcpResource`

```
### PackerSbom Diagram

```mermaid
classDiagram
direction LR

%% PackerSBOM domain model message.

class PackerSbom {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ build_id
  + Optional~string~ format
}
PackerSbom --> `Base`
PackerSbom --> `HcpResource`

```
### PackerBuild Diagram

```mermaid
classDiagram
direction LR

%% PackerBuild domain model message.

class PackerBuild {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ version_id
  + Optional~string~ source_external_identifier
  + Optional~string~ component_type
  + Optional~string~ packer_run_uuid
  + Optional~string~ platform
  + Optional~string~ status
}
PackerBuild --> `Base`
PackerBuild --> `HcpResource`

```
### PackerAncestry Diagram

```mermaid
classDiagram
direction LR

%% PackerAncestry domain model message.

class PackerAncestry {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ parent_bucket_id
  + Optional~string~ child_bucket_id
  + Optional~string~ parent_version_id
  + Optional~string~ child_version_id
  + Optional~string~ channel_name
  + Optional~string~ channel_id
}
PackerAncestry --> `Base`
PackerAncestry --> `HcpResource`

```
### PackerVersion Diagram

```mermaid
classDiagram
direction LR

%% PackerVersion domain model message

class PackerVersion {
  + Base base
  + HcpResource hcp_resource
  + Optional~string~ bucket_id
  + string status
  + string author_id
  + Optional~string~ revocation_type
  + Optional~string~ revocation_message
  + Optional~string~ revocation_author
  + Optional~google.protobuf.Timestamp~ revoked_at
  + Optional~string~ template_type
}
PackerVersion --> `Base`
PackerVersion --> `HcpResource`
PackerVersion --> `google.protobuf.Timestamp`

```
### PackerChannel Diagram

```mermaid
classDiagram
direction LR

%% 

class PackerChannel {
  + Base base
  + HcpResource hcp_resource
  + string bucket_id
  + string version_id
  + string author_id
  + bool managed
  + bool restricted
}
PackerChannel --> `Base`
PackerChannel --> `HcpResource`

```
### PackerArtifact Diagram

```mermaid
classDiagram
direction LR

%% PackerArtifact domain model message

class PackerArtifact {
  + Base base
  + HcpResource hcp_resource
  + string build_id
  + string external_identifier
  + string region
}
PackerArtifact --> `Base`
PackerArtifact --> `HcpResource`

```

## Message: PackerRegistry
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerRegistry</div>

<div class="comment"><span>PackerRegistry domain model message.</span><br/></div>

| Field                       | Ordinal | Type        | Label    | Description |
|-----------------------------|---------|-------------|----------|-------------|
| base                        | 1       | Base        |          |             |
| hcp_resource                | 2       | HcpResource |          |             |
| feature_tier                | 3       | string      | Optional |             |
| activated                   | 4       | bool        | Optional |             |
| billing_deprovision_reason  | 5       | int32       | Optional |             |
| billing_deprovision_message | 6       | string      | Optional |             |




## Message: PackerBucket
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerBucket</div>

<div class="comment"><span>PackerBucket domain model message.</span><br/></div>

| Field        | Ordinal | Type        | Label    | Description                                                                                            |
|--------------|---------|-------------|----------|--------------------------------------------------------------------------------------------------------|
| base         | 1       | Base        |          |                                                                                                        |
| hcp_resource | 2       | HcpResource |          |                                                                                                        |
| registry_id  | 3       | string      | Optional | id of the parent registry                                                                              |
| platforms    | 4       | string      | Repeated | array of the platforms associated with the bucket TODO: string array currently unsupported by codegen  |




## Message: PackerInheritedRevocation
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerInheritedRevocation</div>

<div class="comment"><span>PackerInheritedRevocation domain model message.</span><br/></div>

| Field         | Ordinal | Type        | Label    | Description |
|---------------|---------|-------------|----------|-------------|
| base          | 1       | Base        |          |             |
| hcp_resource  | 2       | HcpResource |          |             |
| descendant_id | 3       | string      | Optional |             |
| ancestor_id   | 4       | string      | Optional |             |




## Message: PackerSbom
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerSbom</div>

<div class="comment"><span>PackerSBOM domain model message.</span><br/></div>

| Field        | Ordinal | Type        | Label    | Description             |
|--------------|---------|-------------|----------|-------------------------|
| base         | 1       | Base        |          |                         |
| hcp_resource | 2       | HcpResource |          |                         |
| build_id     | 3       | string      | Optional | id of the parent build  |
| format       | 4       | string      | Optional | sbom format type        |




## Message: PackerBuild
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerBuild</div>

<div class="comment"><span>PackerBuild domain model message.</span><br/></div>

| Field                      | Ordinal | Type        | Label    | Description               |
|----------------------------|---------|-------------|----------|---------------------------|
| base                       | 1       | Base        |          |                           |
| hcp_resource               | 2       | HcpResource |          |                           |
| version_id                 | 3       | string      | Optional | id of the parent version  |
| source_external_identifier | 4       | string      | Optional | id of the source          |
| component_type             | 5       | string      | Optional |                           |
| packer_run_uuid            | 6       | string      | Optional |                           |
| platform                   | 7       | string      | Optional |                           |
| status                     | 8       | string      | Optional |                           |




## Message: PackerAncestry
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerAncestry</div>

<div class="comment"><span>PackerAncestry domain model message.</span><br/></div>

| Field             | Ordinal | Type        | Label    | Description              |
|-------------------|---------|-------------|----------|--------------------------|
| base              | 1       | Base        |          |                          |
| hcp_resource      | 2       | HcpResource |          |                          |
| parent_bucket_id  | 3       | string      | Optional | id of the parent bucket  |
| child_bucket_id   | 4       | string      | Optional | id of the child bucket   |
| parent_version_id | 5       | string      | Optional | id of the child bucket   |
| child_version_id  | 6       | string      | Optional | id of the child bucket   |
| channel_name      | 7       | string      | Optional |                          |
| channel_id        | 8       | string      | Optional | id of the channel        |




## Message: PackerVersion
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerVersion</div>

<div class="comment"><span>PackerVersion domain model message</span><br/></div>

| Field              | Ordinal | Type                      | Label    | Description |
|--------------------|---------|---------------------------|----------|-------------|
| base               | 1       | Base                      |          |             |
| hcp_resource       | 2       | HcpResource               |          |             |
| bucket_id          | 3       | string                    | Optional |             |
| status             | 5       | string                    |          |             |
| author_id          | 6       | string                    |          |             |
| revocation_type    | 7       | string                    | Optional |             |
| revocation_message | 8       | string                    | Optional |             |
| revocation_author  | 9       | string                    | Optional |             |
| revoked_at         | 10      | google.protobuf.Timestamp | Optional |             |
| template_type      | 11      | string                    | Optional |             |




## Message: PackerChannel
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerChannel</div>

<div class="comment"><span></span><br/></div>

| Field        | Ordinal | Type        | Label | Description |
|--------------|---------|-------------|-------|-------------|
| base         | 1       | Base        |       |             |
| hcp_resource | 2       | HcpResource |       |             |
| bucket_id    | 3       | string      |       |             |
| version_id   | 4       | string      |       |             |
| author_id    | 5       | string      |       |             |
| managed      | 6       | bool        |       |             |
| restricted   | 7       | bool        |       |             |




## Message: PackerArtifact
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.PackerArtifact</div>

<div class="comment"><span>PackerArtifact domain model message</span><br/></div>

| Field               | Ordinal | Type        | Label | Description |
|---------------------|---------|-------------|-------|-------------|
| base                | 1       | Base        |       |             |
| hcp_resource        | 2       | HcpResource |       |             |
| build_id            | 3       | string      |       |             |
| external_identifier | 4       | string      |       |             |
| region              | 5       | string      |       |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
