# Package: hashicorp.cloud.internal.infragraph.domain.v1

<div class="comment"><span></span><br/></div>

## Imports

| Import                                                   | Description |
|----------------------------------------------------------|-------------|
| hashicorp/cloud/internal/infragraph/domain/v1/base.proto |             |



## Options

| Name       | Value                                                   | Description |
|------------|---------------------------------------------------------|-------------|
| go_package | github.com/hashicorp/cloud-infragraph/proto-internal/go |             |



## Enum: EdgeLabel
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.EdgeLabel</div>

<div class="comment"><span>EdgeLabel is a label that describes the relationship between nodes in the graph.</span><br/></div>

| Name                   | Ordinal | Description |
|------------------------|---------|-------------|
| EDGE_LABEL_UNSPECIFIED | 0       |             |
| EDGE_LABEL_CONTAINS    | 1       |             |
| EDGE_LABEL_RUNS        | 2       |             |
| EDGE_LABEL_ATTACHES_TO | 4       |             |
| EDGE_LABEL_APPLIES_TO  | 5       |             |
| EDGE_LABEL_PRODUCES    | 6       |             |
| EDGE_LABEL_CONNECTS_TO | 7       |             |
| EDGE_LABEL_STORES      | 8       |             |
| EDGE_LABEL_HAS_CURRENT | 9       |             |
| EDGE_LABEL_INSTANCE_OF | 10      |             |
| EDGE_LABEL_MANAGES     | 11      |             |
| EDGE_LABEL_RUNS_ON     | 12      |             |
| EDGE_LABEL_OWNS        | 13      |             |


## Enum: EdgeDirection
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.EdgeDirection</div>

<div class="comment"><span>EdgeDirection indicates the direction of the edge in the graph.</span><br/></div>

| Name                       | Ordinal | Description |
|----------------------------|---------|-------------|
| EDGE_DIRECTION_UNSPECIFIED | 0       |             |
| EDGE_DIRECTION_INCOMING    | 1       |             |
| EDGE_DIRECTION_OUTGOING    | 2       |             |



### EdgeLabel Diagram

```mermaid
classDiagram
direction LR
%% EdgeLabel is a label that describes the relationship between nodes in the graph.

class EdgeLabel{
  <<enumeration>>
  EDGE_LABEL_UNSPECIFIED
  EDGE_LABEL_CONTAINS
  EDGE_LABEL_RUNS
  EDGE_LABEL_ATTACHES_TO
  EDGE_LABEL_APPLIES_TO
  EDGE_LABEL_PRODUCES
  EDGE_LABEL_CONNECTS_TO
  EDGE_LABEL_STORES
  EDGE_LABEL_HAS_CURRENT
  EDGE_LABEL_INSTANCE_OF
  EDGE_LABEL_MANAGES
  EDGE_LABEL_RUNS_ON
  EDGE_LABEL_OWNS
}
```
### EdgeDirection Diagram

```mermaid
classDiagram
direction LR
%% EdgeDirection indicates the direction of the edge in the graph.

class EdgeDirection{
  <<enumeration>>
  EDGE_DIRECTION_UNSPECIFIED
  EDGE_DIRECTION_INCOMING
  EDGE_DIRECTION_OUTGOING
}
```
### NodeIdentifier Diagram

```mermaid
classDiagram
direction LR

%% NodeIdentifier encompasses values that uniquely identify a node in the graph.

class NodeIdentifier {
  + string id
  + Source source
  + Kind kind
}
NodeIdentifier --> `Source`
NodeIdentifier --> `Kind`

```
### Edge Diagram

```mermaid
classDiagram
direction LR

%% Edge is a representation of a directed edge out or into a node in the graph.

class Edge {
  + EdgeLabel label
  + EdgeDirection direction
  + NodeIdentifier node_id
}
Edge --> `EdgeLabel`
Edge --> `EdgeDirection`
Edge --> `NodeIdentifier`

```

## Message: NodeIdentifier
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.NodeIdentifier</div>

<div class="comment"><span>NodeIdentifier encompasses values that uniquely identify a node in the graph.</span><br/></div>

| Field  | Ordinal | Type   | Label | Description |
|--------|---------|--------|-------|-------------|
| id     | 1       | string |       |             |
| source | 2       | Source |       |             |
| kind   | 3       | Kind   |       |             |




## Message: Edge
<div style="font-size: 12px; margin-top: -10px;" class="fqn">FQN: hashicorp.cloud.internal.infragraph.domain.v1.Edge</div>

<div class="comment"><span>Edge is a representation of a directed edge out or into a node in the graph.</span><br/></div>

| Field     | Ordinal | Type           | Label | Description |
|-----------|---------|----------------|-------|-------------|
| label     | 1       | EdgeLabel      |       |             |
| direction | 2       | EdgeDirection  |       |             |
| node_id   | 3       | NodeIdentifier |       |             |






<!-- Created by: Proto Diagram Tool -->
<!-- https://github.com/GoogleCloudPlatform/proto-gen-md-diagrams -->
