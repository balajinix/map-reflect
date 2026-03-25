# ITOM-Hashicorp Field Annotation (v2)

Instructions:
- For each source field, choose the best target field from the candidates
- Write the target field key in the 'Your Choice' column
- Write `--` for fields with no reasonable match
- Each target field should be used AT MOST ONCE per type pair (bipartite constraint)

Total: 45 type pairs to annotate

---

## itom_hashi_0010: instana.ApplicationService → infragraph.BaseServerlessFunction
**Relationship:** projection
**Description:** Instana surfaces core infrastructure Service with observability attributes
**Source fields:** 8 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `entityType` | EntityType | description (0.57, EM) / id (0.54, EM) / account_id (0.54, EM) | description |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `label` | String | description (0.56, EM) / name (0.55, EM) / account_id (0.52, EM) | description |
| `technologies` | String | description (0.55, EM) / name (0.54, EM) | description |
| `types` | ServiceType | description (0.55, EM) / name (0.51, EM) / account_id (0.48, EM) | description |
| `_appid` | ID | account_id (0.60, EM) / id (0.57, EM) / region_id (0.54, EM) | account_id |
| `_to` | String | description (0.67, EM) / name (0.59, EM) / account_id (0.58, EM) | description |
| `_windowSize` | String | description (0.55, EM) / name (0.50, EM) | description |

---

## itom_hashi_0011: concert.BuildArtifact → infragraph.BaseVirtualMachineImage
**Relationship:** supply_chain
**Description:** Concert build artifacts packaged into container images
**Source fields:** 18 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | String | id (1.00, NA) / account_id (1.00, NA) / region_id (1.00, NA) | id |
| `name` | String | name (1.00, NA) / description (0.61, EM) / id (0.50, EM) | name |
| `image_tag` | String | description (0.54, EM) | description |
| `digest` | String | description (0.56, EM) / id (0.52, EM) / account_id (0.52, EM) | description |
| `cyclonedx_link` | String | description (0.54, EM) / id (0.52, EM) / name (0.50, EM) | description |
| `type` | ArtifactType | name (0.56, EM) / id (0.53, EM) / description (0.52, EM) | name |
| `version` | String | description (0.64, EM) / name (0.58, EM) / id (0.54, EM) | description |
| `purl` | String | description (0.65, EM) / name (0.59, EM) / id (0.56, EM) | description |
| `file_name` | String | name (1.00, NA) / description (0.55, EM) / account_id (0.52, EM) | name |
| `url` | String | description (0.57, EM) / id (0.53, EM) / name (0.53, EM) | description |
| `scope` | String | description (0.57, EM) / name (0.52, EM) / region_id (0.49, EM) | description |
| `tags` | TagsPrototype | description (0.52, EM) | description |
| `data_impact_risk` | Int | — | -- |
| `last_updated_on` | Int | updated_at (0.55, EM) | updated_at |
| `last_updated_by` | String | updated_at (0.54, EM) | updated_at |
| `uri` | String | description (0.54, EM) / id (0.49, EM) | description |
| `bom_ref` | String | description (0.57, EM) / name (0.56, EM) / account_id (0.54, EM) | description |
| `associations` | BuildArtifactAssociations | name (0.49, EM) | name |

---

## itom_hashi_0013: turbonomic.Action → infragraph.BaseVirtualMachine
**Relationship:** acts_on
**Description:** Turbonomic optimization actions target container workloads
**Source fields:** 43 | **Target fields:** 14

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |
| `actionType` | String | state (1.00, MA) / description (0.68, EM) / name (0.57, EM) | state |
| `actionMode` | String | state (1.00, MA) / description (0.62, EM) / virtual_machine_instance_type (0.60, MA) | state |
| `category` | String | description (0.71, EM) / id (0.58, EM) / name (0.56, EM) | description |
| `createTime` | String | state (1.00, MA) / created_at (0.60, EM) / description (0.58, EM) | state |
| `currentLocation` | EntityReference | vpc_id (0.80, MA) / subnet_id (0.70, MA) / region_id (0.48, EM) | vpc_id |
| `newLocation` | EntityReference | vpc_id (0.80, MA) / subnet_id (0.70, MA) / region_id (0.48, EM) | vpc_id |
| `target` | EntityReference | state (0.60, MA) / name (0.51, EM) / virtual_machine_instance_type (0.50, EM) | state |
| `risk` | Risk | state (0.80, MA) / vpc_id (0.30, MA) / subnet_id (0.30, MA) | state |
| `details` | String | state (1.00, MA) / description (0.74, EM) / name (0.60, EM) | state |
| `explanation` | String | description (0.76, EM) / name (0.65, EM) / id (0.64, EM) | description |
| `savings` | Savings | — | -- |
| `actionID` | ID | state (0.80, MA) / id (0.61, EM) / vpc_id (0.60, MA) | state |
| `actionImpactID` | ID | vpc_id (0.80, MA) / subnet_id (0.80, MA) / virtual_machine_image_id (0.80, MA) | vpc_id |
| `actionSchedule` | ActionSchedule | state (0.80, MA) | state |
| `actionState` | ActionState | state (0.90, MA) / description (0.49, EM) / vpc_id (0.10, MA) | state |
| `compoundActions` | CompoundActionsEntry | network_interface_ids (0.80, MA) / state (0.50, MA) | network_interface_ids |
| `updateTime` | DateTime | state (0.80, MA) / updated_at (0.59, EM) / created_at (0.52, EM) | state |
| `clearTime` | DateTime | state (0.80, MA) / created_at (0.54, EM) / updated_at (0.54, EM) | state |
| `currentEntity` | CurrentEntity | vpc_id (0.60, MA) / subnet_id (0.60, MA) / state (0.59, EM) | vpc_id |
| `currentValue` | Float | state (0.80, MA) / vpc_id (0.40, MA) | state |
| `displayName` | String | description (0.55, EM) | description |
| `entityCharacteristics` | JSON | state (0.80, MA) / virtual_machine_instance_type (0.30, MA) | state |
| `executionCharacteristics` | ExecutionCharacteristics | state (0.80, MA) / vpc_id (0.20, MA) | state |
| `importance` | Float | state (0.80, MA) / virtual_machine_instance_type (0.50, MA) / virtual_machine_image_id (0.40, MA) | state |
| `marketID` | ID | vpc_id (1.00, MA) / subnet_id (0.90, MA) / virtual_machine_image_id (0.80, MA) | vpc_id |
| `newEntity` | NewEntity | vpc_id (0.70, MA) / subnet_id (0.70, MA) / state (0.58, EM) | vpc_id |
| `newValue` | Float | state (0.80, MA) / virtual_machine_instance_type (0.60, MA) / description (0.49, EM) | state |
| `policy` | JSON | state (0.40, MA) / vpc_id (0.20, MA) / subnet_id (0.20, MA) | state |
| `prerequisites` | String | state (1.00, MA) / description (0.56, EM) / name (0.50, EM) | state |
| `prioritizedAutomationOpportunities` | String | virtual_machine_instance_type (1.00, MA) / state (0.80, MA) | virtual_machine_instance_type |
| `relatedActionsCountByType` | RelatedActionsCountByType | state (0.80, MA) / network_interface_ids (0.60, MA) / vpc_id (0.40, MA) | state |
| `relatedSettingsPolicies` | JSON | state (0.40, MA) / vpc_id (0.20, MA) | state |
| `reservedInstance` | JSON | virtual_machine_instance_type (0.80, MA) / description (0.51, EM) / id (0.48, EM) | virtual_machine_instance_type |
| `reservedInstancePlatform` | String | virtual_machine_instance_type (1.00, MA) / virtual_machine_image_id (0.90, MA) / state (0.70, MA) | virtual_machine_instance_type |
| `externalActionName` | String | name (0.52, EM) / id (0.49, EM) | name |
| `externalActionUrl` | String | — | -- |
| `resizeAttribute` | ResizeAttribute | virtual_machine_instance_type (0.80, MA) / state (0.40, MA) | virtual_machine_instance_type |
| `source` | ActionSource | name (0.56, EM) / virtual_machine_image_id (0.50, MA) / id (0.49, EM) | name |
| `stats` | StatsEntry | state (0.80, MA) / vpc_id (0.20, MA) / subnet_id (0.20, MA) | state |
| `template` | Template | description (0.52, EM) / virtual_machine_image_id (0.50, MA) / virtual_machine_instance_type (0.50, EM) | description |
| `valueUnits` | String | state (1.00, MA) / virtual_machine_instance_type (1.00, MA) / description (0.62, EM) | state |
| `virtualDisks` | VirtualDisksEntry | state (0.60, EM) / virtual_machine_image_id (0.57, EM) / virtual_machine_instance_type (0.54, EM) | state |

---

## itom_hashi_0014: turbonomic.Entity → infragraph.BaseServerlessFunction
**Relationship:** composed_of
**Description:** Turbonomic application is composed of core infrastructure services
**Source fields:** 17 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | description (0.69, EM) / name (0.59, EM) / id (0.56, EM) | description |
| `connectedEntities` | ConnectedEntitiesEntry | — | -- |
| `consumers` | ConsumersEntry | — | -- |
| `costPrice` | Float | — | -- |
| `discoveredBy` | DiscoveredBy | id (0.59, EM) / name (0.53, EM) / account_id (0.52, EM) | id |
| `displayName` | String | description (0.54, EM) | description |
| `environmentType` | EnvironmentType | name (0.51, EM) / description (0.49, EM) | name |
| `numRelatedVMs` | Int | — | -- |
| `providers` | ProvidersEntry | — | -- |
| `severity` | String | description (0.52, EM) | description |
| `severityBreakdown` | SeverityBreakdown | description (0.52, EM) | description |
| `staleness` | String | description (0.59, EM) / id (0.54, EM) / name (0.53, EM) | description |
| `state` | String | description (0.68, EM) / name (0.58, EM) / id (0.56, EM) | description |
| `tags` | JSON | description (0.56, EM) / name (0.50, EM) | description |
| `template` | Template | — | -- |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |
| `vendorIds` | JSON | id (0.58, EM) / account_id (0.56, EM) / region_id (0.50, EM) | id |

---

## itom_hashi_0015: instana.ApplicationPerspective → infragraph.BaseServerlessFunction
**Relationship:** composed_of
**Description:** Instana application is composed of core infrastructure services
**Source fields:** 6 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `boundaryScope` | BoundaryScope | — | -- |
| `entityType` | EntityType | description (0.57, EM) / id (0.54, EM) / account_id (0.54, EM) | description |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `label` | String | description (0.61, EM) / name (0.61, EM) / id (0.57, EM) | description |
| `to` | String | description (0.69, EM) / name (0.59, EM) / id (0.57, EM) | description |
| `windowSize` | String | description (0.58, EM) / name (0.52, EM) / id (0.49, EM) | description |

---

## itom_hashi_0016: concert.Container_builds → infragraph.BaseVirtualMachineImage
**Relationship:** produces
**Description:** Concert CI/CD pipelines produce container images
**Source fields:** 8 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `build_number` | String | description (0.58, EM) / account_id (0.57, EM) / name (0.55, EM) | description |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `image_id` | ID | id (0.92, NA) / account_id (0.57, EM) / region_id (0.54, EM) | id |
| `last_updated_by` | String | updated_at (0.55, EM) / account_id (0.49, EM) | updated_at |
| `last_updated_on` | ID | updated_at (0.52, EM) / account_id (0.49, EM) | updated_at |
| `name` | String | name (1.00, NA) / description (0.75, EM) / id (0.61, EM) | name |
| `sbom_id` | ID | id (0.92, NA) / account_id (0.60, EM) / zone_ids (0.54, EM) | id |
| `version` | String | description (0.73, EM) / name (0.64, EM) / id (0.61, EM) | description |

---

## itom_hashi_0018: hcp_terraform.HcpTerraformWorkspace → turbonomic.ContainerClusterEntity
**Relationship:** governs
**Description:** Terraform templates govern cluster provisioning
**Source fields:** 15 | **Target fields:** 3

Target fields: `className`, `displayName`, `uuid`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | string | uuid (0.65, EM) / className (0.56, EM) / displayName (0.53, EM) | uuid |
| `name` | string | className (0.59, EM) / displayName (0.57, EM) | className |
| `account_id` | string | uuid (0.65, EM) / className (0.54, EM) / displayName (0.51, EM) | uuid |
| `region_id` | string | uuid (0.57, EM) | uuid |
| `zone_ids` | string | uuid (0.59, EM) | uuid |
| `created_at` | google.protobuf.Timestamp | — | -- |
| `updated_at` | google.protobuf.Timestamp | — | -- |
| `description` | string | className (0.69, EM) / displayName (0.67, EM) | className |
| `current_run_id` | string | uuid (1.00, MA) / className (0.70, MA) / displayName (0.70, MA) | uuid |
| `resource_count` | int32 | className (0.80, MA) / displayName (0.70, MA) / uuid (0.60, MA) | className |
| `current_state_version_id` | string | uuid (1.00, MA) / className (0.70, MA) / displayName (0.70, MA) | uuid |
| `environment` | string | className (1.00, MA) / displayName (1.00, MA) / uuid (0.60, MA) | className |
| `execution_mode` | string | className (1.00, MA) / displayName (0.90, MA) / uuid (0.60, MA) | className |
| `tag_names` | string | displayName (1.00, MA) / className (0.90, MA) / uuid (0.60, MA) | displayName |
| `agent_pool_id` | string | uuid (1.00, MA) / className (0.90, MA) / displayName (0.90, MA) | uuid |

---

## itom_hashi_0057: turbonomic.VirtualDisksEntry → infragraph.BaseVolume
**Relationship:** schema_mapping
**Description:** Turbonomic virtual disk maps to infragraph base volume
**Source fields:** 21 | **Target fields:** 13

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `attached_virtual_machine_ids`, `size`, `iops`, `throughput`, `is_encrypted`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `attachedVMStorageCompatibility` | String | attached_virtual_machine_ids (1.00, MA) / description (0.54, EM) / name (0.48, EM) | attached_virtual_machine_ids |
| `attachedVirtualMachine` | AttachedVirtualMachine | attached_virtual_machine_ids (0.90, MA) / name (0.50, EM) / description (0.49, EM) | attached_virtual_machine_ids |
| `attachmentState` | String | attached_virtual_machine_ids (1.00, MA) / description (0.58, EM) / size (0.10, MA) | attached_virtual_machine_ids |
| `businessAccount` | BusinessAccount | account_id (0.50, EM) | account_id |
| `creationTimeStamp` | Float | created_at (0.59, EM) / updated_at (0.52, EM) | created_at |
| `dataCenter` | DataCenter | — | -- |
| `diskRole` | String | attached_virtual_machine_ids (0.70, MA) / description (0.57, EM) / region_id (0.51, EM) | attached_virtual_machine_ids |
| `displayName` | String | attached_virtual_machine_ids (0.70, MA) / description (0.67, EM) / name (0.57, EM) | attached_virtual_machine_ids |
| `encryption` | String | is_encrypted (1.00, MA) / description (0.63, EM) / id (0.52, EM) | is_encrypted |
| `environmentType` | String | description (0.69, EM) / name (0.61, EM) / account_id (0.59, EM) | description |
| `hourlyBilledOps` | Float | size (0.37, MA) / iops (0.37, MA) / attached_virtual_machine_ids (0.17, MA) | size |
| `lastAttachedVm` | String | attached_virtual_machine_ids (1.00, MA) / description (0.54, EM) / size (0.20, MA) | attached_virtual_machine_ids |
| `lastModified` | Float | — | -- |
| `numDaysUnattached` | Int | attached_virtual_machine_ids (0.80, MA) / iops (0.49, EM) / size (0.48, EM) | attached_virtual_machine_ids |
| `provider` | Provider | description (0.51, EM) / account_id (0.49, EM) | description |
| `redundancyType` | String | attached_virtual_machine_ids (0.70, MA) / description (0.67, EM) / id (0.61, EM) | attached_virtual_machine_ids |
| `resourceId` | ID | id (0.59, EM) / account_id (0.57, EM) / region_id (0.55, EM) | id |
| `snapshotId` | ID | id (0.57, EM) / account_id (0.55, EM) / region_id (0.50, EM) | id |
| `stats` | StatsEntry | iops (0.90, MA) / throughput (0.80, MA) / size (0.70, MA) | iops |
| `tier` | String | description (0.67, EM) / name (0.58, EM) / id (0.55, EM) | description |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |

---

## itom_hashi_0058: turbonomic.CloudType → infragraph.Source
**Relationship:** schema_mapping
**Description:** Turbonomic cloud type maps to infragraph source
**Source fields:** 5 | **Target fields:** 7

Target fields: `SOURCE_UNSPECIFIED`, `SOURCE_AWS`, `SOURCE_AZURE`, `SOURCE_GCP`, `SOURCE_HCP_PACKER`, `SOURCE_HCP_TERRAFORM`, `SOURCE_HCP`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `AWS` | ENUM_VALUE | SOURCE_AWS (1.00, MA) / SOURCE_UNSPECIFIED (0.72, EM) / SOURCE_HCP_TERRAFORM (0.69, EM) | SOURCE_AWS |
| `AZURE` | ENUM_VALUE | SOURCE_AZURE (1.00, MA) / SOURCE_UNSPECIFIED (0.69, EM) / SOURCE_AWS (0.67, EM) | SOURCE_AZURE |
| `GCP` | ENUM_VALUE | SOURCE_GCP (1.00, MA) / SOURCE_UNSPECIFIED (0.70, EM) / SOURCE_HCP (0.70, EM) | SOURCE_GCP |
| `HYBRID` | ENUM_VALUE | SOURCE_HCP (1.00, MA) / SOURCE_HCP_PACKER (1.00, MA) / SOURCE_HCP_TERRAFORM (1.00, MA) | SOURCE_HCP |
| `UNKNOWN` | ENUM_VALUE | SOURCE_UNSPECIFIED (1.00, MA) / SOURCE_HCP (0.75, EM) / SOURCE_AWS (0.70, MA) | SOURCE_UNSPECIFIED |

---

## itom_hashi_0059: turbonomic.CloudAspect → aws.AwsAccount
**Relationship:** schema_mapping
**Description:** Turbonomic cloud aspect maps to AWS account
**Source fields:** 4 | **Target fields:** 11

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`, `arn`, `account`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `businessAccount` | BusinessAccount | account (0.90, MA) / account_id (0.50, EM) / organization_id (0.50, EM) | account |
| `resourceGroup` | ResourceGroup | account (0.80, MA) / organization_id (0.48, EM) / description (0.48, EM) | account |
| `resourceId` | ID | organization_id (0.65, EM) / id (0.59, EM) / account_id (0.57, EM) | organization_id |
| `type` | String | description (0.76, EM) / arn (0.75, EM) / name (0.62, EM) | description |

---

## itom_hashi_0060: turbonomic.CloudAspect → aws.AwsEcsCluster
**Relationship:** schema_mapping
**Description:** Turbonomic cloud aspect maps to AWS ECS cluster
**Source fields:** 4 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `businessAccount` | BusinessAccount | account_id (0.50, EM) | account_id |
| `resourceGroup` | ResourceGroup | description (0.48, EM) | description |
| `resourceId` | ID | id (0.59, EM) / account_id (0.57, EM) / region_id (0.55, EM) | id |
| `type` | String | description (0.76, EM) / arn (0.75, EM) / name (0.62, EM) | description |

---

## itom_hashi_0061: instana.Entity → infragraph.Base
**Relationship:** semantic_equivalent
**Description:** Both represent a base infrastructure entity — Instana monitors it, Infragraph models it
**Source fields:** 17 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `tags`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |
| `displayName` | String | description (0.67, EM) / tags (0.58, EM) / name (0.57, EM) | description |
| `className` | String | name (0.64, EM) / description (0.60, EM) / id (0.55, EM) | name |
| `entityType` | String | name (1.00, MA) / id (1.00, MA) / description (0.90, MA) | name |
| `state` | String | description (0.68, EM) / tags (0.62, EM) / name (0.58, EM) | description |
| `severity` | String | description (0.66, EM) / tags (0.56, EM) / id (0.53, EM) | description |
| `environmentType` | String | description (0.69, EM) / tags (0.63, EM) / name (0.61, EM) | description |
| `discoveredBy` | DiscoveredBy | id (0.54, EM) / name (0.49, EM) | id |
| `externalIds` | ExternalIds | id (0.59, EM) / account_id (0.54, EM) / zone_ids (0.54, EM) | id |
| `cves` | CVE | — | -- |
| `aspects` | EntityAspects | description (0.53, EM) / id (0.50, EM) / name (0.49, EM) | description |
| `actions` | Action | — | -- |
| `stats` | Statistic | description (0.50, EM) | description |
| `tags` | Tag | tags (0.95, NA) / description (0.57, EM) / name (0.49, EM) | tags |
| `topology` | Topology | — | -- |
| `id` | ID | id (1.00, MA) / account_id (1.00, MA) / region_id (0.92, NA) | id |
| `label` | String | name (1.00, MA) / description (1.00, MA) / id (0.90, MA) | name |

---

## itom_hashi_0062: instana.EntityType → infragraph.Kind
**Relationship:** semantic_equivalent
**Description:** Both enumerate types of infrastructure entities
**Source fields:** 8 | **Target fields:** 37

Target fields: `KIND_UNSPECIFIED`, `KIND_VIRTUAL_MACHINE`, `KIND_ACCOUNT`, `KIND_REGION`, `KIND_ZONE`, `KIND_VIRTUAL_NETWORK`, `KIND_SUBNET`, `KIND_NETWORK_INTERFACE`, `KIND_VIRTUAL_MACHINE_IMAGE`, `KIND_ECS_CLUSTER`, `KIND_ECS_SERVICE`, `KIND_PACKER_BUCKET`, `KIND_PACKER_REGISTRY`, `KIND_POLICY`, `KIND_USER`, `KIND_STORAGE_BUCKET`, `KIND_VOLUME`, `KIND_SECURITY_GROUP`, `KIND_SECURITY_RULE`, `KIND_PACKER_BUILD`, `KIND_PACKER_SBOM`, `KIND_PACKER_VERSION`, `KIND_PACKER_SOURCE`, `KIND_SERVERLESS_FUNCTION`, `KIND_PACKER_INHERITED_REVOCATION`, `KIND_PACKER_ANCESTRY`, `KIND_PACKER_CHANNEL`, `KIND_PACKER_ARTIFACT`, `KIND_ORGANIZATION`, `KIND_HCP_TERRAFORM_STATE_VERSION`
  ... and 7 more

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `APPLICATION` | ENUM_VALUE | KIND_ECS_CLUSTER (1.00, MA) / KIND_ECS_SERVICE (1.00, MA) / KIND_VIRTUAL_MACHINE (0.90, MA) | KIND_ECS_CLUSTER |
| `ENDPOINT` | ENUM_VALUE | KIND_NETWORK_INTERFACE (1.00, MA) / KIND_VIRTUAL_MACHINE (1.00, MA) / KIND_ECS_SERVICE (0.90, MA) | KIND_NETWORK_INTERFACE |
| `INFRASTRUCTURE` | ENUM_VALUE | KIND_VIRTUAL_MACHINE (1.00, MA) / KIND_VIRTUAL_NETWORK (1.00, MA) / KIND_SUBNET (1.00, MA) | KIND_VIRTUAL_MACHINE |
| `LOG` | ENUM_VALUE | KIND_UNSPECIFIED (0.75, EM) / KIND_ZONE (0.73, EM) / KIND_REGION (0.72, EM) | KIND_UNSPECIFIED |
| `MOBILE_APP` | ENUM_VALUE | KIND_ECS_SERVICE (1.00, MA) / KIND_ECS_CLUSTER (1.00, MA) / KIND_VIRTUAL_MACHINE (0.70, MA) | KIND_ECS_SERVICE |
| `SERVICE` | ENUM_VALUE | KIND_ECS_SERVICE (1.00, MA) / KIND_UNSPECIFIED (0.75, EM) / KIND_ORGANIZATION (0.72, EM) | KIND_ECS_SERVICE |
| `SYNTHETICS` | ENUM_VALUE | KIND_VIRTUAL_MACHINE (1.00, MA) / KIND_ECS_CLUSTER (1.00, MA) / KIND_ECS_SERVICE (1.00, MA) | KIND_VIRTUAL_MACHINE |
| `WEBSITE` | ENUM_VALUE | KIND_ECS_SERVICE (1.00, MA) / KIND_ECS_CLUSTER (1.00, MA) / KIND_VIRTUAL_MACHINE (0.90, MA) | KIND_ECS_SERVICE |

---

## itom_hashi_0063: instana.Event → aws.AwsEc2Instance
**Relationship:** monitors
**Description:** Instana events monitor AWS EC2 instances
**Source fields:** 11 | **Target fields:** 18

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`, `arn`, `tenancy`, `private_ip`, `monitoring`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `detail` | String | monitoring (0.80, MA) / description (0.67, EM) / private_ip (0.60, MA) | monitoring |
| `end` | DateTime | monitoring (0.80, MA) / created_at (0.53, EM) / updated_at (0.52, EM) | monitoring |
| `entityLabel` | String | description (0.57, EM) / id (0.53, EM) / name (0.53, EM) | description |
| `entityName` | String | name (0.56, EM) / description (0.56, EM) / id (0.54, EM) | name |
| `entityType` | EntityType | id (0.55, EM) / name (0.54, EM) / description (0.53, EM) | id |
| `eventId` | ID | id (0.56, EM) / account_id (0.52, EM) / network_interface_ids (0.51, EM) | id |
| `problem` | String | tenancy (0.70, MA) / monitoring (0.70, MA) / description (0.60, EM) | tenancy |
| `severity` | Int | monitoring (0.80, MA) | monitoring |
| `start` | DateTime | created_at (0.57, EM) / updated_at (0.54, EM) | created_at |
| `state` | EventState | state (0.95, NA) / monitoring (0.80, MA) / tenancy (0.20, MA) | state |
| `type` | EventType | monitoring (0.70, MA) / description (0.54, EM) / virtual_machine_instance_type (0.51, EM) | monitoring |

---

## itom_hashi_0064: instana.ApplicationService → aws.AwsEcsService
**Relationship:** monitors
**Description:** Instana monitors AWS ECS services
**Source fields:** 8 | **Target fields:** 11

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `cluster_id`, `subnet_ids`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `entityType` | EntityType | description (0.57, EM) / id (0.54, EM) / account_id (0.54, EM) | description |
| `id` | ID | id (1.00, NA) / cluster_id (0.92, NA) / account_id (0.92, NA) | id |
| `label` | String | description (0.56, EM) / name (0.55, EM) / arn (0.52, EM) | description |
| `technologies` | String | cluster_id (0.90, MA) / subnet_ids (0.80, MA) / description (0.55, EM) | cluster_id |
| `types` | ServiceType | cluster_id (0.60, MA) / description (0.55, EM) / name (0.51, EM) | cluster_id |
| `_appid` | ID | cluster_id (0.80, MA) / subnet_ids (0.60, EM) / account_id (0.60, EM) | cluster_id |
| `_to` | String | cluster_id (0.90, MA) / subnet_ids (0.80, MA) / description (0.67, EM) | cluster_id |
| `_windowSize` | String | cluster_id (0.90, MA) / subnet_ids (0.80, MA) / description (0.55, EM) | cluster_id |

---

## itom_hashi_0065: instana.ApplicationService → aws.AwsLambdaFunction
**Relationship:** monitors
**Description:** Instana monitors AWS Lambda functions
**Source fields:** 8 | **Target fields:** 16

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `role`, `memory_size`, `run_time`, `architectures`, `subnet_ids`, `security_group_ids`, `serverless_function`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `entityType` | EntityType | serverless_function (0.80, MA) / architectures (0.58, EM) / description (0.57, EM) | serverless_function |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `label` | String | description (1.00, MA) / role (0.70, MA) / serverless_function (0.60, MA) | description |
| `technologies` | String | architectures (1.00, MA) / run_time (0.90, MA) / description (0.70, MA) | architectures |
| `types` | ServiceType | architectures (0.80, MA) / run_time (0.60, MA) / description (0.55, EM) | architectures |
| `_appid` | ID | serverless_function (0.80, MA) / description (0.60, MA) / subnet_ids (0.60, EM) | serverless_function |
| `_to` | String | description (0.90, MA) / role (0.80, MA) / serverless_function (0.80, MA) | description |
| `_windowSize` | String | memory_size (0.80, MA) / run_time (0.80, MA) / description (0.70, MA) | memory_size |

---

## itom_hashi_0066: instana.ApplicationPerspective → aws.AwsEcsCluster
**Relationship:** monitors
**Description:** Instana application perspective monitors AWS ECS clusters
**Source fields:** 6 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `boundaryScope` | BoundaryScope | — | -- |
| `entityType` | EntityType | description (0.57, EM) / id (0.54, EM) / account_id (0.54, EM) | description |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `label` | String | description (0.61, EM) / name (0.61, EM) / id (0.57, EM) | description |
| `to` | String | description (0.69, EM) / arn (0.69, EM) / name (0.59, EM) | description |
| `windowSize` | String | description (0.58, EM) / arn (0.55, EM) / name (0.52, EM) | description |

---

## itom_hashi_0067: concert.Application → aws.AwsEcsService
**Relationship:** governs
**Description:** Concert application governance covers AWS ECS services
**Source fields:** 22 | **Target fields:** 11

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `cluster_id`, `subnet_ids`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | String | cluster_id (1.00, NA) / id (1.00, NA) / account_id (1.00, NA) | cluster_id |
| `name` | String | name (1.00, NA) / description (0.69, EM) / arn (0.62, EM) | name |
| `latest` | Boolean | — | -- |
| `status` | String | description (0.66, EM) / arn (0.57, EM) / name (0.52, EM) | description |
| `auth_user_role` | String | account_id (0.56, EM) / arn (0.55, EM) / description (0.54, EM) | account_id |
| `version` | String | cluster_id (0.70, MA) / description (0.69, EM) / name (0.64, EM) | cluster_id |
| `tags` | TagsPrototype | description (0.55, EM) / name (0.54, EM) / id (0.51, EM) | description |
| `description` | String | description (1.00, NA) / name (0.66, EM) / arn (0.62, EM) | description |
| `created_on` | Int | created_at (0.66, EM) / updated_at (0.56, EM) / cluster_id (0.50, MA) | created_at |
| `last_updated_on` | Int | updated_at (0.59, EM) / created_at (0.51, EM) | updated_at |
| `last_updated_by` | String | updated_at (0.55, EM) / account_id (0.49, EM) | updated_at |
| `last_modified_by` | String | cluster_id (0.80, MA) / updated_at (0.51, EM) / account_id (0.50, EM) | cluster_id |
| `criticality` | Int | cluster_id (0.20, MA) | cluster_id |
| `data_impact_risk` | Int | — | -- |
| `business_name` | String | name (1.00, NA) / description (0.60, EM) / arn (0.55, EM) | name |
| `revenue_per_hour` | Float | — | -- |
| `business_units` | String | cluster_id (0.60, MA) / description (0.53, EM) / subnet_ids (0.52, EM) | cluster_id |
| `business_id` | String | id (1.00, NA) / cluster_id (0.60, EM) / account_id (0.60, EM) | id |
| `sample` | Boolean | — | -- |
| `resilience_status` | String | cluster_id (0.90, MA) / subnet_ids (0.80, MA) / description (0.54, EM) | cluster_id |
| `properties` | ApplicationReleaseProperties | — | -- |
| `associations` | ApplicationAssociationsInfo | subnet_ids (0.80, MA) / cluster_id (0.70, MA) / id (0.48, EM) | subnet_ids |

---

## itom_hashi_0068: concert.Application → aws.AwsLambdaFunction
**Relationship:** governs
**Description:** Concert application governance covers AWS Lambda functions
**Source fields:** 22 | **Target fields:** 16

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `role`, `memory_size`, `run_time`, `architectures`, `subnet_ids`, `security_group_ids`, `serverless_function`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | String | id (1.00, NA) / account_id (1.00, NA) / region_id (1.00, NA) | id |
| `name` | String | description (1.00, MA) / name (1.00, NA) / role (0.70, MA) | description |
| `latest` | Boolean | description (0.80, MA) / serverless_function (0.60, MA) / run_time (0.50, MA) | description |
| `status` | String | description (1.00, MA) / role (0.90, MA) / run_time (0.80, MA) | description |
| `auth_user_role` | String | role (1.00, MA) / security_group_ids (0.62, EM) / description (0.60, MA) | role |
| `version` | String | description (1.00, MA) / run_time (0.90, MA) / architectures (0.68, EM) | description |
| `tags` | TagsPrototype | description (0.80, MA) / role (0.60, MA) / name (0.54, EM) | description |
| `description` | String | description (1.00, MA) / architectures (0.71, EM) / name (0.66, EM) | description |
| `created_on` | Int | run_time (0.80, MA) / created_at (0.66, EM) / updated_at (0.56, EM) | run_time |
| `last_updated_on` | Int | run_time (0.80, MA) / updated_at (0.59, EM) / created_at (0.51, EM) | run_time |
| `last_updated_by` | String | description (1.00, MA) / role (0.90, MA) / run_time (0.70, MA) | description |
| `last_modified_by` | String | role (1.00, MA) / run_time (0.55, EM) / updated_at (0.51, EM) | role |
| `criticality` | Int | description (0.80, MA) / role (0.60, MA) / serverless_function (0.50, MA) | description |
| `data_impact_risk` | Int | description (0.80, MA) / role (0.60, MA) / security_group_ids (0.50, MA) | description |
| `business_name` | String | description (1.00, MA) / name (1.00, NA) / role (0.57, EM) | description |
| `revenue_per_hour` | Float | run_time (0.80, MA) / memory_size (0.60, MA) / serverless_function (0.30, MA) | run_time |
| `business_units` | String | role (0.90, MA) / description (0.70, MA) / architectures (0.52, EM) | role |
| `business_id` | String | id (1.00, NA) / description (0.60, MA) / account_id (0.60, EM) | id |
| `sample` | Boolean | description (0.80, MA) / role (0.60, MA) / serverless_function (0.50, MA) | description |
| `resilience_status` | String | role (1.00, MA) / serverless_function (0.80, MA) / description (0.80, MA) | role |
| `properties` | ApplicationReleaseProperties | description (0.80, MA) / serverless_function (0.60, MA) / role (0.40, MA) | description |
| `associations` | ApplicationAssociationsInfo | subnet_ids (0.80, MA) / security_group_ids (0.70, MA) / role (0.60, MA) | subnet_ids |

---

## itom_hashi_0069: concert.Environment → aws.AwsVpc
**Relationship:** maps_to
**Description:** Concert deployment environment maps to AWS VPC
**Source fields:** 3 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `vpc`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | String | id (1.00, NA) / account_id (1.00, NA) / region_id (1.00, NA) | id |
| `name` | String | name (1.00, NA) / vpc (0.70, MA) / description (0.68, EM) | name |
| `config` | JSON | vpc (0.70, MA) | vpc |

---

## itom_hashi_0070: concert.Environment → aws.AwsRegion
**Relationship:** maps_to
**Description:** Concert deployment environment maps to AWS region
**Source fields:** 3 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `region`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `id` | String | id (1.00, NA) / account_id (1.00, NA) / region_id (1.00, NA) | id |
| `name` | String | name (1.00, NA) / region (0.70, MA) / description (0.68, EM) | name |
| `config` | JSON | region (0.70, MA) | region |

---

## itom_hashi_0071: concert.Resources_images → aws.AwsAmi
**Relationship:** maps_to
**Description:** Concert container images map to AWS AMIs
**Source fields:** 2 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `virtual_machine_image`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `image_id` | ID | id (0.92, NA) / virtual_machine_image (0.80, MA) / account_id (0.57, EM) | id |
| `resource_id` | ID | id (0.92, NA) / virtual_machine_image (0.90, MA) / account_id (0.60, EM) | id |

---

## itom_hashi_0072: concert.Resources_images → infragraph.BaseVirtualMachineImage
**Relationship:** maps_to
**Description:** Concert container images map to Infragraph VM images
**Source fields:** 2 | **Target fields:** 8

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `image_id` | ID | id (0.92, NA) / account_id (0.57, EM) / region_id (0.54, EM) | id |
| `resource_id` | ID | id (0.92, NA) / account_id (0.60, EM) / region_id (0.58, EM) | id |

---

## itom_hashi_0073: concert.Credentials → aws.AwsIamRole
**Relationship:** maps_to
**Description:** Concert credentials map to AWS IAM roles
**Source fields:** 10 | **Target fields:** 17

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `path`, `assume_role_policy_document`, `max_session_duration`, `last_used_date`, `last_used_region`, `permissions_boundary_arn`, `permissions_boundary_type`, `role`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `created_by` | String | last_used_date (0.80, MA) / description (0.80, MA) / path (0.70, MA) | last_used_date |
| `created_on` | ID | last_used_date (0.80, MA) / account_id (0.59, EM) / created_at (0.56, EM) | last_used_date |
| `credentials` | JSON | role (0.80, MA) / assume_role_policy_document (0.70, MA) / permissions_boundary_arn (0.60, MA) | role |
| `description` | String | description (1.00, MA) / path (0.70, MA) / assume_role_policy_document (0.70, MA) | description |
| `expiry` | ID | max_session_duration (0.80, MA) / path (0.60, MA) / last_used_date (0.60, MA) | max_session_duration |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `last_updated_by` | String | last_used_region (1.00, MA) / description (0.90, MA) / last_used_date (0.80, MA) | last_used_region |
| `last_updated_on` | ID | last_used_date (0.90, MA) / updated_at (0.52, EM) / path (0.50, MA) | last_used_date |
| `name` | String | description (1.00, MA) / name (1.00, NA) / path (0.90, MA) | description |
| `type` | String | assume_role_policy_document (1.00, MA) / permissions_boundary_arn (1.00, MA) / permissions_boundary_type (1.00, MA) | assume_role_policy_document |

---

## itom_hashi_0074: concert.Credentials → aws.AwsIamUser
**Relationship:** maps_to
**Description:** Concert credentials map to AWS IAM users
**Source fields:** 10 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `user`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `created_by` | String | user (0.80, MA) / created_at (0.58, EM) / account_id (0.57, EM) | user |
| `created_on` | ID | user (0.80, MA) / account_id (0.59, EM) / created_at (0.56, EM) | user |
| `credentials` | JSON | user (0.90, MA) | user |
| `description` | String | description (1.00, NA) / user (0.80, MA) / arn (0.65, EM) | description |
| `expiry` | ID | user (0.80, MA) / id (0.54, EM) / account_id (0.53, EM) | user |
| `id` | ID | id (1.00, NA) / account_id (0.92, NA) / region_id (0.92, NA) | id |
| `last_updated_by` | String | user (0.80, MA) / updated_at (0.55, EM) / account_id (0.49, EM) | user |
| `last_updated_on` | ID | user (0.80, MA) / updated_at (0.52, EM) / account_id (0.49, EM) | user |
| `name` | String | name (1.00, NA) / user (0.90, MA) / description (0.75, EM) | name |
| `type` | String | user (0.80, MA) / description (0.76, EM) / arn (0.75, EM) | user |

---

## itom_hashi_0075: concert.Vulnerability → packer.PackerSbom
**Relationship:** references
**Description:** Concert vulnerabilities reference Packer SBOM (software bill of materials)
**Source fields:** 37 | **Target fields:** 12

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`, `project_id`, `build_id`, `format`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `cve` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.60, EM) | build_id |
| `cvss` | Float | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `highest_finding_risk_score` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `custom_score` | Float | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `has_unassessed_findings` | Boolean | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `unassessed_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `has_exceptions_requested` | Boolean | build_id (0.20, MA) / format (0.20, MA) | build_id |
| `exceptions_requested_count` | String | build_id (0.90, MA) / format (0.90, MA) / description (0.56, EM) | build_id |
| `total_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `priority_1_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `priority_2_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `priority_3_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `deprioritized_findings` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `has_open_tickets` | Boolean | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `open_tickets` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `highest_finding_priority` | String | build_id (0.90, MA) / format (0.80, MA) | build_id |
| `severity` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.56, EM) | build_id |
| `package_name` | String | name (1.00, NA) / build_id (0.90, MA) / format (0.80, MA) | name |
| `package_version` | String | build_id (1.00, MA) / format (0.80, MA) / description (0.54, EM) | build_id |
| `package_type` | String | format (1.00, MA) / build_id (0.70, MA) / description (0.57, EM) | format |
| `description` | String | description (1.00, NA) / build_id (0.90, MA) / format (0.80, MA) | description |
| `vector` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.55, EM) | build_id |
| `component_id` | String | build_id (1.00, MA) / id (1.00, NA) / account_id (0.63, EM) | build_id |
| `component_type` | ComponentType | format (0.80, MA) / account_id (0.52, EM) / description (0.52, EM) | format |
| `vulnerability_findings_id` | String | build_id (1.00, MA) / id (0.60, EM) / project_id (0.58, EM) | build_id |
| `state` | StateType | build_id (0.80, MA) / format (0.40, MA) | build_id |
| `comment` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.60, EM) | build_id |
| `wx_details` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.64, EM) | build_id |
| `wx_recommendation` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.56, EM) | build_id |
| `aliases` | String | build_id (0.90, MA) / format (0.80, MA) / name (0.56, EM) | build_id |
| `related_cves` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.56, EM) | build_id |
| `osv_response_message` | String | build_id (0.90, MA) / format (0.80, MA) / description (0.58, EM) | build_id |
| `findings` | VulnerabilityAssociationComponent | build_id (0.60, MA) / format (0.50, MA) / name (0.49, EM) | build_id |
| `application_count` | Int | build_id (0.60, MA) / format (0.50, MA) | build_id |
| `access_point_count` | Int | build_id (0.40, MA) / format (0.20, MA) | build_id |
| `exceptions_requested` | Int | build_id (0.60, MA) / format (0.50, MA) / account_id (0.49, EM) | build_id |
| `associations` | VulnerabilityAssociations | build_id (0.60, MA) / id (0.50, EM) / account_id (0.48, EM) | build_id |

---

## itom_hashi_0076: concert.Container_builds → packer.PackerBuild
**Relationship:** maps_to
**Description:** Concert container builds map to Packer image builds
**Source fields:** 8 | **Target fields:** 16

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`, `project_id`, `version_id`, `source_external_identifier`, `component_type`, `packer_run_uuid`, `platform`, `status`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `build_number` | String | packer_run_uuid (1.00, MA) / version_id (1.00, MA) / source_external_identifier (0.70, MA) | packer_run_uuid |
| `id` | ID | id (1.00, NA) / version_id (0.92, NA) / account_id (0.92, NA) | id |
| `image_id` | ID | id (0.92, NA) / version_id (0.90, MA) / source_external_identifier (0.80, MA) | id |
| `last_updated_by` | String | status (1.00, MA) / version_id (1.00, MA) / source_external_identifier (0.80, MA) | status |
| `last_updated_on` | ID | version_id (1.00, MA) / status (0.80, MA) / source_external_identifier (0.60, MA) | version_id |
| `name` | String | version_id (1.00, MA) / name (1.00, NA) / source_external_identifier (0.90, MA) | version_id |
| `sbom_id` | ID | id (0.92, NA) / version_id (0.90, MA) / source_external_identifier (0.80, MA) | id |
| `version` | String | version_id (1.00, MA) / source_external_identifier (0.80, MA) / component_type (0.80, MA) | version_id |

---

## itom_hashi_0077: concert.Container_builds → packer.PackerArtifact
**Relationship:** produces
**Description:** Concert container builds produce Packer artifacts
**Source fields:** 8 | **Target fields:** 13

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`, `project_id`, `build_id`, `external_identifier`, `region`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `build_number` | String | build_id (1.00, MA) / external_identifier (0.80, MA) / region (0.60, MA) | build_id |
| `id` | ID | build_id (1.00, MA) / id (1.00, NA) / account_id (0.92, NA) | build_id |
| `image_id` | ID | id (0.92, NA) / build_id (0.90, MA) / external_identifier (0.80, MA) | id |
| `last_updated_by` | String | build_id (1.00, MA) / external_identifier (0.60, MA) / updated_at (0.55, EM) | build_id |
| `last_updated_on` | ID | build_id (1.00, MA) / updated_at (0.52, EM) / external_identifier (0.50, EM) | build_id |
| `name` | String | build_id (1.00, MA) / external_identifier (1.00, MA) / name (1.00, NA) | build_id |
| `sbom_id` | ID | build_id (1.00, MA) / id (0.92, NA) / external_identifier (0.80, MA) | build_id |
| `version` | String | build_id (1.00, MA) / external_identifier (0.90, MA) / description (0.73, EM) | build_id |

---

## itom_hashi_0078: concert.Image_deployments → hcp_terraform.HcpTerraformRun
**Relationship:** triggers
**Description:** Concert image deployments trigger Terraform runs
**Source fields:** 6 | **Target fields:** 12

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `is_destroyed`, `message`, `status`, `workspace_id`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `deployment_number` | String | status (1.00, MA) / message (0.70, MA) / description (0.57, EM) | status |
| `id` | ID | id (1.00, NA) / workspace_id (0.92, NA) / account_id (0.92, NA) | id |
| `image_id` | ID | id (0.92, NA) / workspace_id (0.90, MA) / status (0.60, MA) | id |
| `last_updated_by` | String | status (0.90, MA) / updated_at (0.55, EM) / is_destroyed (0.53, EM) | status |
| `last_updated_on` | ID | status (0.80, MA) / is_destroyed (0.52, EM) / updated_at (0.52, EM) | status |
| `sbom_id` | ID | id (0.92, NA) / workspace_id (0.90, MA) / status (0.80, MA) | id |

---

## itom_hashi_0079: concert.Topology_application_view → infragraph.Model
**Relationship:** maps_to
**Description:** Concert application topology view maps to Infragraph graph model
**Source fields:** 9 | **Target fields:** 39

Target fields: `aws_ec2_instance`, `azure_virtual_machine`, `gcp_compute_instance`, `packer_bucket`, `aws_subnet`, `aws_vpc`, `aws_ecs_cluster`, `aws_ecs_service`, `packer_registry`, `aws_availability_zone`, `aws_region`, `aws_iam_policy`, `aws_iam_user`, `aws_ami`, `packer_version`, `aws_security_group`, `aws_security_group_rule`, `packer_sbom`, `packer_build`, `packer_channel`, `packer_artifact`, `aws_s3_bucket`, `aws_ec2_volume`, `aws_lambda_function`, `packer_inherited_revocation`, `packer_ancestry`, `aws_account`, `aws_network_interface`, `aws_ec2_instance_type`, `hcp_organization`
  ... and 9 more

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `applications` | JSON | aws_ecs_cluster (0.80, MA) / aws_ecs_service (0.70, MA) / aws_ec2_instance (0.60, MA) | aws_ecs_cluster |
| `cves` | JSON | aws_ec2_instance (0.80, MA) / aws_ecs_cluster (0.70, MA) / aws_ecs_service (0.70, MA) | aws_ec2_instance |
| `dast_exposures` | JSON | aws_security_group (0.80, MA) / aws_security_group_rule (0.70, MA) / aws_ec2_instance (0.60, MA) | aws_security_group |
| `environments` | JSON | aws_vpc (0.80, MA) / aws_ec2_instance (0.70, MA) / azure_virtual_machine (0.70, MA) | aws_vpc |
| `images` | JSON | aws_ami (0.80, MA) / packer_build (0.60, MA) / packer_sbom (0.40, MA) | aws_ami |
| `private_access_points` | JSON | aws_security_group (0.80, MA) / aws_security_group_rule (0.70, MA) / aws_subnet (0.60, MA) | aws_security_group |
| `public_access_points` | JSON | aws_security_group (0.80, MA) / aws_security_group_rule (0.70, MA) / aws_subnet (0.60, MA) | aws_security_group |
| `repositories` | JSON | packer_bucket (0.80, MA) / packer_registry (0.70, MA) / aws_ecs_cluster (0.60, MA) | packer_bucket |
| `sast_exposures` | JSON | aws_security_group (0.80, MA) / aws_security_group_rule (0.70, MA) / aws_iam_policy (0.60, MA) | aws_security_group |

---

## itom_hashi_0080: concert.Profile → aws.AwsIamPolicy
**Relationship:** maps_to
**Description:** Concert compliance profiles map to AWS IAM policies
**Source fields:** 8 | **Target fields:** 11

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `attachment_count`, `policy`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `additional_payload` | JSON | policy (0.80, MA) / description (0.49, EM) / attachment_count (0.40, MA) | policy |
| `description` | String | description (1.00, NA) / policy (0.80, MA) / arn (0.65, EM) | description |
| `last_updated_by` | String | updated_at (0.55, EM) / account_id (0.49, EM) / arn (0.48, EM) | updated_at |
| `last_updated_on` | ID | policy (0.60, MA) / updated_at (0.52, EM) / account_id (0.49, EM) | policy |
| `profile_id` | ID | id (0.92, NA) / account_id (0.61, EM) / policy (0.60, MA) | id |
| `sample_data_upload` | Boolean | attachment_count (0.52, EM) | attachment_count |
| `title` | String | description (0.72, EM) / arn (0.64, EM) / name (0.59, EM) | description |
| `version` | String | description (0.73, EM) / arn (0.70, EM) / name (0.64, EM) | description |

---

## itom_hashi_0081: turbonomic.AttachedVirtualMachine → aws.AwsEc2Instance
**Relationship:** optimizes
**Description:** Turbonomic optimizes AWS EC2 virtual machines
**Source fields:** 3 | **Target fields:** 18

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`, `arn`, `tenancy`, `private_ip`, `monitoring`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | description (0.69, EM) / arn (0.65, EM) / virtual_machine_instance_type (0.61, EM) | description |
| `displayName` | String | description (0.67, EM) / private_ip (0.60, MA) / arn (0.60, EM) | description |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |

---

## itom_hashi_0082: turbonomic.AttachedVirtualMachine → azure.AzureVirtualMachine
**Relationship:** optimizes
**Description:** Turbonomic optimizes Azure virtual machines
**Source fields:** 3 | **Target fields:** 18

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`, `etag`, `provisioning_state`, `priority`, `extensions_time_budget`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | description (0.69, EM) / virtual_machine_instance_type (0.61, EM) / name (0.59, EM) | description |
| `displayName` | String | description (0.67, EM) / name (0.57, EM) / virtual_machine_instance_type (0.53, EM) | description |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |

---

## itom_hashi_0083: turbonomic.AttachedVirtualMachine → gcp.GcpComputeInstance
**Relationship:** optimizes
**Description:** Turbonomic optimizes GCP compute instances
**Source fields:** 3 | **Target fields:** 14

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | description (0.69, EM) / virtual_machine_instance_type (0.61, EM) / name (0.59, EM) | description |
| `displayName` | String | description (0.67, EM) / name (0.57, EM) / virtual_machine_instance_type (0.53, EM) | description |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |

---

## itom_hashi_0084: turbonomic.AttachedVirtualMachine → infragraph.BaseVirtualMachine
**Relationship:** optimizes
**Description:** Turbonomic optimizes virtual machines modeled in Infragraph
**Source fields:** 3 | **Target fields:** 14

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `state`, `vpc_id`, `subnet_id`, `network_interface_ids`, `virtual_machine_image_id`, `virtual_machine_instance_type`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | virtual_machine_instance_type (1.00, MA) / virtual_machine_image_id (0.90, MA) / description (0.69, EM) | virtual_machine_instance_type |
| `displayName` | String | state (0.70, MA) / description (0.67, EM) / name (0.57, EM) | state |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / state (0.60, MA) | id |

---

## itom_hashi_0085: turbonomic.ContainerClusterEntity → aws.AwsEcsCluster
**Relationship:** optimizes
**Description:** Turbonomic optimizes AWS ECS container clusters
**Source fields:** 3 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | description (0.69, EM) / arn (0.65, EM) / name (0.59, EM) | description |
| `displayName` | String | description (0.67, EM) / arn (0.60, EM) / name (0.57, EM) | description |
| `uuid` | ID | id (0.65, EM) / account_id (0.65, EM) / zone_ids (0.59, EM) | id |

---

## itom_hashi_0086: turbonomic.BusinessAccount → aws.AwsAccount
**Relationship:** maps_to
**Description:** Turbonomic business account maps to AWS account
**Source fields:** 9 | **Target fields:** 11

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`, `arn`, `account`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | account (0.80, MA) / description (0.69, EM) / arn (0.65, EM) | account |
| `displayName` | String | account (0.80, MA) / description (0.67, EM) / arn (0.60, EM) | account |
| `uuid` | ID | account (0.80, MA) / id (0.65, EM) / account_id (0.65, EM) | account |
| `discoveredBy` | DiscoveredBy | account (0.80, MA) / id (0.55, EM) / arn (0.51, EM) | account |
| `environmentType` | String | description (0.69, EM) / arn (0.64, EM) / name (0.61, EM) | description |
| `staleness` | String | account (0.80, MA) / arn (0.61, EM) / description (0.59, EM) | account |
| `state` | String | account (0.80, MA) / description (0.68, EM) / arn (0.65, EM) | account |
| `tags` | JSON | account (0.80, MA) / description (0.56, EM) / arn (0.50, EM) | account |
| `vendorIds` | JSON | account (0.80, MA) / region_id (0.52, EM) / organization_id (0.51, EM) | account |

---

## itom_hashi_0087: turbonomic.BusinessAccount → infragraph.BaseAccount
**Relationship:** maps_to
**Description:** Turbonomic business account maps to Infragraph base account
**Source fields:** 9 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `organization_id`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | organization_id (0.90, MA) / description (0.69, EM) / name (0.59, EM) | organization_id |
| `displayName` | String | organization_id (1.00, MA) / description (0.67, EM) / name (0.57, EM) | organization_id |
| `uuid` | ID | organization_id (0.90, MA) / id (0.65, EM) / account_id (0.65, EM) | organization_id |
| `discoveredBy` | DiscoveredBy | organization_id (0.60, MA) / id (0.55, EM) / name (0.49, EM) | organization_id |
| `environmentType` | String | organization_id (0.90, MA) / description (0.69, EM) / name (0.61, EM) | organization_id |
| `staleness` | String | organization_id (0.70, MA) / description (0.59, EM) / id (0.54, EM) | organization_id |
| `state` | String | organization_id (1.00, MA) / description (0.68, EM) / name (0.58, EM) | organization_id |
| `tags` | JSON | organization_id (0.70, MA) / description (0.56, EM) | organization_id |
| `vendorIds` | JSON | organization_id (0.70, MA) / region_id (0.52, EM) / account_id (0.50, EM) | organization_id |

---

## itom_hashi_0088: turbonomic.DataCenter → aws.AwsRegion
**Relationship:** maps_to
**Description:** Turbonomic data center maps to AWS region
**Source fields:** 3 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `region`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | region (0.80, MA) / description (0.69, EM) / arn (0.65, EM) | region |
| `displayName` | String | region (0.70, MA) / description (0.67, EM) / arn (0.60, EM) | region |
| `uuid` | ID | region (0.70, MA) / id (0.65, EM) / account_id (0.65, EM) | region |

---

## itom_hashi_0089: turbonomic.NamespaceEntity → aws.AwsVpc
**Relationship:** maps_to
**Description:** Turbonomic Kubernetes namespace maps to AWS VPC network isolation
**Source fields:** 3 | **Target fields:** 10

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `arn`, `vpc`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | vpc (0.80, MA) / description (0.69, EM) / arn (0.65, EM) | vpc |
| `displayName` | String | vpc (0.70, MA) / description (0.67, EM) / arn (0.60, EM) | vpc |
| `uuid` | ID | vpc (0.80, MA) / id (0.65, EM) / account_id (0.65, EM) | vpc |

---

## itom_hashi_0090: turbonomic.Action → hcp_terraform.HcpTerraformRun
**Relationship:** triggers
**Description:** Turbonomic optimization actions trigger Terraform runs
**Source fields:** 43 | **Target fields:** 12

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `is_destroyed`, `message`, `status`, `workspace_id`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `uuid` | ID | workspace_id (0.90, MA) / id (0.65, EM) / account_id (0.65, EM) | workspace_id |
| `actionType` | String | status (0.90, MA) / message (0.69, EM) / description (0.68, EM) | status |
| `actionMode` | String | status (1.00, MA) / message (0.65, EM) / description (0.62, EM) | status |
| `category` | String | description (0.71, EM) / status (0.66, EM) / message (0.66, EM) | description |
| `createTime` | String | status (1.00, MA) / message (0.90, MA) / created_at (0.60, EM) | status |
| `currentLocation` | EntityReference | region_id (0.48, EM) / workspace_id (0.40, MA) | region_id |
| `newLocation` | EntityReference | region_id (0.48, EM) | region_id |
| `target` | EntityReference | name (0.51, EM) | name |
| `risk` | Risk | message (0.80, MA) / status (0.70, MA) | message |
| `details` | String | message (1.00, MA) / status (0.80, MA) / description (0.74, EM) | message |
| `explanation` | String | description (0.76, EM) / message (0.72, EM) / status (0.68, EM) | description |
| `savings` | Savings | — | -- |
| `actionID` | ID | workspace_id (0.90, MA) / status (0.80, MA) / id (0.61, EM) | workspace_id |
| `actionImpactID` | ID | workspace_id (0.90, MA) / status (0.60, MA) / id (0.52, EM) | workspace_id |
| `actionSchedule` | ActionSchedule | status (0.60, MA) / is_destroyed (0.30, MA) | status |
| `actionState` | ActionState | status (0.90, MA) / message (0.50, EM) / description (0.49, EM) | status |
| `compoundActions` | CompoundActionsEntry | — | -- |
| `updateTime` | DateTime | status (0.80, MA) / updated_at (0.59, EM) / created_at (0.52, EM) | status |
| `clearTime` | DateTime | status (0.60, MA) / created_at (0.54, EM) / updated_at (0.54, EM) | status |
| `currentEntity` | CurrentEntity | id (0.48, EM) / workspace_id (0.40, MA) | id |
| `currentValue` | Float | status (0.60, MA) | status |
| `displayName` | String | message (1.00, MA) / status (0.80, MA) / description (0.55, EM) | message |
| `entityCharacteristics` | JSON | status (0.80, MA) / is_destroyed (0.70, MA) / message (0.60, MA) | status |
| `executionCharacteristics` | ExecutionCharacteristics | status (0.80, MA) / message (0.50, MA) / is_destroyed (0.50, MA) | status |
| `importance` | Float | status (0.60, MA) | status |
| `marketID` | ID | workspace_id (1.00, MA) / id (0.53, EM) / account_id (0.51, EM) | workspace_id |
| `newEntity` | NewEntity | name (0.49, EM) / workspace_id (0.40, MA) | name |
| `newValue` | Float | status (0.80, MA) / message (0.60, MA) / is_destroyed (0.50, MA) | status |
| `policy` | JSON | is_destroyed (0.40, MA) / message (0.40, MA) / status (0.40, MA) | is_destroyed |
| `prerequisites` | String | status (1.00, MA) / message (0.90, MA) / description (0.56, EM) | status |
| `prioritizedAutomationOpportunities` | String | status (0.90, MA) | status |
| `relatedActionsCountByType` | RelatedActionsCountByType | status (0.80, MA) | status |
| `relatedSettingsPolicies` | JSON | is_destroyed (0.20, MA) / message (0.20, MA) | is_destroyed |
| `reservedInstance` | JSON | status (0.70, MA) / workspace_id (0.60, MA) / message (0.52, EM) | status |
| `reservedInstancePlatform` | String | status (0.80, MA) / id (0.54, EM) / message (0.52, EM) | status |
| `externalActionName` | String | message (1.00, MA) / status (0.70, MA) / name (0.52, EM) | message |
| `externalActionUrl` | String | message (1.00, MA) / workspace_id (0.80, MA) / status (0.70, MA) | message |
| `resizeAttribute` | ResizeAttribute | status (0.70, MA) | status |
| `source` | ActionSource | status (0.60, MA) / name (0.56, EM) / message (0.50, MA) | status |
| `stats` | StatsEntry | status (0.60, MA) | status |
| `template` | Template | status (0.60, MA) / description (0.52, EM) / message (0.51, EM) | status |
| `valueUnits` | String | status (0.90, MA) / description (0.62, EM) / message (0.60, EM) | status |
| `virtualDisks` | VirtualDisksEntry | message (0.60, MA) | message |

---

## itom_hashi_0091: turbonomic.Template → hcp_terraform.HcpTerraformWorkspace
**Relationship:** maps_to
**Description:** Turbonomic resource templates map to Terraform workspaces
**Source fields:** 6 | **Target fields:** 15

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `current_run_id`, `resource_count`, `current_state_version_id`, `environment`, `execution_mode`, `tag_names`, `agent_pool_id`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `className` | String | environment (0.90, MA) / execution_mode (0.80, MA) / current_state_version_id (0.70, MA) | environment |
| `discovered` | Boolean | current_state_version_id (0.70, MA) / id (0.52, EM) / resource_count (0.50, MA) | current_state_version_id |
| `displayName` | String | environment (1.00, MA) / tag_names (0.90, MA) / current_state_version_id (0.80, MA) | environment |
| `enableMatch` | Boolean | execution_mode (0.80, MA) / current_state_version_id (0.50, MA) / environment (0.40, MA) | execution_mode |
| `price` | Float | resource_count (0.80, MA) / environment (0.40, MA) / agent_pool_id (0.40, MA) | resource_count |
| `uuid` | ID | current_state_version_id (1.00, MA) / current_run_id (0.90, MA) / agent_pool_id (0.90, MA) | current_state_version_id |

---

## itom_hashi_0092: turbonomic.Statistics → infragraph.Base
**Relationship:** monitors
**Description:** Turbonomic statistics monitor Infragraph base entities
**Source fields:** 4 | **Target fields:** 9

Target fields: `id`, `name`, `account_id`, `region_id`, `zone_ids`, `created_at`, `updated_at`, `description`, `tags`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `date` | DateTime | created_at (0.90, MA) / updated_at (0.80, MA) / id (0.50, MA) | created_at |
| `displayName` | String | name (1.00, MA) / description (1.00, MA) / id (0.90, MA) | name |
| `epoch` | String | created_at (0.80, MA) / updated_at (0.70, MA) / id (0.70, MA) | created_at |
| `statistics` | StatsEntry | name (0.80, MA) / description (0.70, MA) / tags (0.60, MA) | name |

---

## itom_hashi_0093: turbonomic.Target → infragraph.Source
**Relationship:** maps_to
**Description:** Turbonomic optimization target maps to Infragraph cloud source
**Source fields:** 10 | **Target fields:** 7

Target fields: `SOURCE_UNSPECIFIED`, `SOURCE_AWS`, `SOURCE_AZURE`, `SOURCE_GCP`, `SOURCE_HCP_PACKER`, `SOURCE_HCP_TERRAFORM`, `SOURCE_HCP`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `aspects` | JSON | SOURCE_UNSPECIFIED (0.40, MA) / SOURCE_AZURE (0.32, MA) | SOURCE_UNSPECIFIED |
| `className` | String | SOURCE_UNSPECIFIED (0.20, MA) / SOURCE_AWS (0.20, MA) / SOURCE_AZURE (0.20, MA) | SOURCE_UNSPECIFIED |
| `connectedEntities` | ConnectedEntitiesEntry | SOURCE_UNSPECIFIED (0.50, MA) / SOURCE_AWS (0.50, MA) | SOURCE_UNSPECIFIED |
| `discoveredBy` | DiscoveredBy | SOURCE_UNSPECIFIED (0.40, MA) / SOURCE_AWS (0.30, MA) / SOURCE_AZURE (0.30, MA) | SOURCE_UNSPECIFIED |
| `displayName` | String | SOURCE_UNSPECIFIED (0.50, MA) / SOURCE_AWS (0.50, MA) / SOURCE_AZURE (0.50, MA) | SOURCE_UNSPECIFIED |
| `environmentType` | String | SOURCE_UNSPECIFIED (0.80, MA) / SOURCE_AWS (0.70, MA) / SOURCE_AZURE (0.70, MA) | SOURCE_UNSPECIFIED |
| `state` | String | — | -- |
| `tags` | JSON | SOURCE_UNSPECIFIED (0.00, MA) / SOURCE_AWS (0.00, MA) | SOURCE_UNSPECIFIED |
| `uuid` | ID | SOURCE_UNSPECIFIED (0.50, MA) / SOURCE_AWS (0.50, MA) | SOURCE_UNSPECIFIED |
| `vendorIds` | JSON | SOURCE_AWS (0.40, MA) / SOURCE_AZURE (0.40, MA) / SOURCE_GCP (0.40, MA) | SOURCE_AWS |

---

## itom_hashi_0094: turbonomic.CurrentEntity → infragraph.NodeIdentifier
**Relationship:** maps_to
**Description:** Turbonomic current entity maps to Infragraph node identifier
**Source fields:** 9 | **Target fields:** 3

Target fields: `id`, `source`, `kind`

| Source Field | Type | Candidates (ranked) | Your Choice |
|-------------|------|---------------------|-------------|
| `aspects` | JSON | id (0.80, MA) / source (0.60, MA) / kind (0.60, MA) | id |
| `className` | String | kind (0.80, MA) / id (0.70, MA) / source (0.49, EM) | kind |
| `discoveredBy` | DiscoveredBy | source (0.80, MA) / kind (0.56, EM) / id (0.54, EM) | source |
| `displayName` | String | id (1.00, MA) / kind (0.60, MA) / source (0.40, MA) | id |
| `environmentType` | String | kind (0.80, MA) / id (0.70, MA) / source (0.60, MA) | kind |
| `state` | String | id (1.00, MA) / kind (0.70, MA) / source (0.60, MA) | id |
| `tags` | JSON | id (0.80, MA) / source (0.60, MA) / kind (0.60, MA) | id |
| `uuid` | ID | id (1.00, MA) / kind (0.48, EM) | id |
| `vendorIds` | JSON | id (0.80, MA) | id |
