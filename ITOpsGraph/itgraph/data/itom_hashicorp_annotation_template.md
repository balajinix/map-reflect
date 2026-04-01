# ITOM-Hashicorp Field Mapping Annotation

Instructions:
- For each source field, write the target field it maps to
- Write `--` if no match exists in the target
- Bipartite: each source maps to at most one target, each target to at most one source
- Fields pre-filled with exact name matches — verify and correct
- Target fields with `(Base)` suffix are from the embedded Base type

Cross-product type pairs: 45

## itom_hashi_0010: instana.ApplicationService → infragraph.BaseServerlessFunction
*projection: Instana surfaces core infrastructure Service with observability attributes*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| _appid | -- |
| _to | -- |
| _windowSize | -- |
| entityType | -- |
| id | base.id |
| label | -- |
| technologies | -- |
| types | -- |

## itom_hashi_0011: concert.BuildArtifact → infragraph.BaseVirtualMachineImage
*supply_chain: Concert build artifacts packaged into container images*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| associations | -- |
| bom_ref | -- |
| cyclonedx_link | -- |
| data_impact_risk | -- |
| digest | -- |
| file_name | -- |
| id | base.id |
| image_tag | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | base.name |
| purl | -- |
| scope | -- |
| tags | base.tags |
| type | -- |
| uri | -- |
| url | -- |
| version | -- |

## itom_hashi_0013: turbonomic.Action → infragraph.BaseVirtualMachine
*acts_on: Turbonomic optimization actions target container workloads*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, network_interface_ids, state, subnet_id, virtual_machine_image_id, virtual_machine_instance_type, vpc_id

| Source Field | Maps To |
|-------------|---------|
| actionID | -- |
| actionImpactID | -- |
| actionMode | -- |
| actionSchedule | -- |
| actionState | -- |
| actionType | -- |
| clearTime | -- |
| compoundActions | -- |
| createTime | -- |
| currentEntity | -- |
| currentLocation | -- |
| currentValue | -- |
| details | -- |
| displayName | -- |
| entityCharacteristics | -- |
| executionCharacteristics | -- |
| externalActionName | -- |
| externalActionUrl | -- |
| importance | -- |
| marketID | -- |
| newEntity | -- |
| newLocation | -- |
| newValue | -- |
| policy | -- |
| prerequisites | -- |
| prioritizedAutomationOpportunities | -- |
| relatedActionsCountByType | -- |
| relatedSettingsPolicies | -- |
| reservedInstance | -- |
| reservedInstancePlatform | -- |
| resizeAttribute | -- |
| risk | -- |
| source | -- |
| stats | -- |
| target | -- |
| template | -- |
| updateTime | -- |
| uuid | -- |
| valueUnits | -- |
| virtualDisks | -- |

## itom_hashi_0014: turbonomic.Entity → infragraph.BaseServerlessFunction
*composed_of: Turbonomic application is composed of core infrastructure services*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| connectedEntities | -- |
| consumers | -- |
| costPrice | -- |
| discoveredBy | -- |
| displayName | -- |
| environmentType | -- |
| numRelatedVMs | -- |
| providers | -- |
| severity | -- |
| severityBreakdown | -- |
| staleness | -- |
| state | -- |
| tags | base.tags |
| template | -- |
| uuid | -- |
| vendorIds | -- |

## itom_hashi_0015: instana.ApplicationPerspective → infragraph.BaseServerlessFunction
*composed_of: Instana application is composed of core infrastructure services*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| boundaryScope | -- |
| entityType | -- |
| id | base.id |
| label | -- |
| to | -- |
| windowSize | -- |

## itom_hashi_0016: concert.Container_builds → infragraph.BaseVirtualMachineImage
*produces: Concert CI/CD pipelines produce container images*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| build_number | -- |
| id | base.id |
| image_id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | base.name |
| sbom_id | -- |
| version | -- |

## itom_hashi_0018: hcp_terraform.HcpTerraformWorkspace → turbonomic.ContainerClusterEntity
*governs: Terraform templates govern cluster provisioning*

**Target fields:** className, displayName, uuid

| Source Field | Maps To |
|-------------|---------|
| agent_pool_id | -- |
| base | -- |
| current_run_id | -- |
| current_state_version_id | -- |
| environment | -- |
| execution_mode | -- |
| resource_count | -- |
| tag_names | -- |

## itom_hashi_0057: turbonomic.VirtualDisksEntry → infragraph.BaseVolume
*schema_mapping: Turbonomic virtual disk maps to infragraph base volume*

**Target fields:** attached_virtual_machine_ids, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, iops, is_encrypted, size, throughput

| Source Field | Maps To |
|-------------|---------|
| attachedVMStorageCompatibility | -- |
| attachedVirtualMachine | -- |
| attachmentState | -- |
| businessAccount | -- |
| creationTimeStamp | -- |
| dataCenter | -- |
| diskRole | -- |
| displayName | -- |
| encryption | -- |
| environmentType | -- |
| hourlyBilledOps | -- |
| lastAttachedVm | -- |
| lastModified | -- |
| numDaysUnattached | -- |
| provider | -- |
| redundancyType | -- |
| resourceId | -- |
| snapshotId | -- |
| stats | -- |
| tier | -- |
| uuid | -- |

## itom_hashi_0058: turbonomic.CloudType → infragraph.Source
*schema_mapping: Turbonomic cloud type maps to infragraph source*

**Target fields:** SOURCE_AWS, SOURCE_AZURE, SOURCE_GCP, SOURCE_HCP, SOURCE_HCP_PACKER, SOURCE_HCP_TERRAFORM, SOURCE_UNSPECIFIED

| Source Field | Maps To |
|-------------|---------|
| AWS | -- |
| AZURE | -- |
| GCP | -- |
| HYBRID | -- |
| UNKNOWN | -- |

## itom_hashi_0059: turbonomic.CloudAspect → aws.AwsAccount
*schema_mapping: Turbonomic cloud aspect maps to AWS account*

**Target fields:** account, aws_resource

| Source Field | Maps To |
|-------------|---------|
| businessAccount | -- |
| resourceGroup | -- |
| resourceId | -- |
| type | -- |

## itom_hashi_0060: turbonomic.CloudAspect → aws.AwsEcsCluster
*schema_mapping: Turbonomic cloud aspect maps to AWS ECS cluster*

**Target fields:** aws_resource, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| businessAccount | -- |
| resourceGroup | -- |
| resourceId | -- |
| type | -- |

## itom_hashi_0061: instana.Entity → infragraph.Base
*semantic_equivalent: Both represent a base infrastructure entity — Instana monitors it, Infragraph models it*

**Target fields:** account_id, created_at, description, id, name, region_id, tags, updated_at, zone_ids

| Source Field | Maps To |
|-------------|---------|
| entityType | -- |
| id | id |
| label | -- |

## itom_hashi_0062: instana.EntityType → infragraph.Kind
*semantic_equivalent: Both enumerate types of infrastructure entities*

**Target fields:** KIND_ACCOUNT, KIND_ECS_CLUSTER, KIND_ECS_SERVICE, KIND_HCP_TERRAFORM_AGENT_POOL, KIND_HCP_TERRAFORM_ORGANIZATION, KIND_HCP_TERRAFORM_RUN, KIND_HCP_TERRAFORM_STATE_VERSION, KIND_HCP_TERRAFORM_WORKSPACE, KIND_NETWORK_INTERFACE, KIND_ORGANIZATION, KIND_PACKER_ANCESTRY, KIND_PACKER_ARTIFACT, KIND_PACKER_BUCKET, KIND_PACKER_BUILD, KIND_PACKER_CHANNEL, KIND_PACKER_INHERITED_REVOCATION, KIND_PACKER_REGISTRY, KIND_PACKER_SBOM, KIND_PACKER_SOURCE, KIND_PACKER_VERSION, KIND_POLICY, KIND_REGION, KIND_ROLE, KIND_ROLE_POLICY_ATTACHMENT, KIND_SECURITY_GROUP, KIND_SECURITY_RULE, KIND_SERVERLESS_FUNCTION, KIND_STORAGE_BUCKET, KIND_SUBNET, KIND_UNSPECIFIED, KIND_USER, KIND_VIRTUAL_MACHINE, KIND_VIRTUAL_MACHINE_IMAGE, KIND_VIRTUAL_MACHINE_TYPE, KIND_VIRTUAL_NETWORK, KIND_VOLUME, KIND_ZONE

| Source Field | Maps To |
|-------------|---------|
| APPLICATION | -- |
| ENDPOINT | -- |
| INFRASTRUCTURE | -- |
| LOG | -- |
| MOBILE_APP | -- |
| SERVICE | -- |
| SYNTHETICS | -- |
| WEBSITE | -- |

## itom_hashi_0063: instana.Event → aws.AwsEc2Instance
*monitors: Instana events monitor AWS EC2 instances*

**Target fields:** aws_resource, monitoring, private_ip, tenancy, virtual_machine

| Source Field | Maps To |
|-------------|---------|
| detail | -- |
| end | -- |
| entityLabel | -- |
| entityName | -- |
| entityType | -- |
| eventId | -- |
| problem | -- |
| severity | -- |
| start | -- |
| state | -- |
| type | -- |

## itom_hashi_0064: instana.ApplicationService → aws.AwsEcsService
*monitors: Instana monitors AWS ECS services*

**Target fields:** aws_resource, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, cluster_id, subnet_ids

| Source Field | Maps To |
|-------------|---------|
| _appid | -- |
| _to | -- |
| _windowSize | -- |
| entityType | -- |
| id | base.id |
| label | -- |
| technologies | -- |
| types | -- |

## itom_hashi_0065: instana.ApplicationService → aws.AwsLambdaFunction
*monitors: Instana monitors AWS Lambda functions*

**Target fields:** architectures, aws_resource, description, memory_size, role, run_time, security_group_ids, serverless_function, subnet_ids

| Source Field | Maps To |
|-------------|---------|
| _appid | -- |
| _to | -- |
| _windowSize | -- |
| entityType | -- |
| id | -- |
| label | -- |
| technologies | -- |
| types | -- |

## itom_hashi_0066: instana.ApplicationPerspective → aws.AwsEcsCluster
*monitors: Instana application perspective monitors AWS ECS clusters*

**Target fields:** aws_resource, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| boundaryScope | -- |
| entityType | -- |
| id | base.id |
| label | -- |
| to | -- |
| windowSize | -- |

## itom_hashi_0067: concert.Application → aws.AwsEcsService
*governs: Concert application governance covers AWS ECS services*

**Target fields:** aws_resource, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, cluster_id, subnet_ids

| Source Field | Maps To |
|-------------|---------|
| associations | -- |
| auth_user_role | -- |
| business_id | -- |
| business_name | -- |
| business_units | -- |
| created_on | -- |
| criticality | -- |
| data_impact_risk | -- |
| description | base.description |
| id | base.id |
| last_modified_by | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| latest | -- |
| name | base.name |
| properties | -- |
| resilience_status | -- |
| revenue_per_hour | -- |
| sample | -- |
| status | -- |
| tags | base.tags |
| version | -- |

## itom_hashi_0068: concert.Application → aws.AwsLambdaFunction
*governs: Concert application governance covers AWS Lambda functions*

**Target fields:** architectures, aws_resource, description, memory_size, role, run_time, security_group_ids, serverless_function, subnet_ids

| Source Field | Maps To |
|-------------|---------|
| associations | -- |
| auth_user_role | -- |
| business_id | -- |
| business_name | -- |
| business_units | -- |
| created_on | -- |
| criticality | -- |
| data_impact_risk | -- |
| description | description |
| id | -- |
| last_modified_by | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| latest | -- |
| name | -- |
| properties | -- |
| resilience_status | -- |
| revenue_per_hour | -- |
| sample | -- |
| status | -- |
| tags | -- |
| version | -- |

## itom_hashi_0069: concert.Environment → aws.AwsVpc
*maps_to: Concert deployment environment maps to AWS VPC*

**Target fields:** aws_resource, vpc

| Source Field | Maps To |
|-------------|---------|
| config | -- |
| id | -- |
| name | -- |

## itom_hashi_0070: concert.Environment → aws.AwsRegion
*maps_to: Concert deployment environment maps to AWS region*

**Target fields:** aws_resource, region

| Source Field | Maps To |
|-------------|---------|
| config | -- |
| id | -- |
| name | -- |

## itom_hashi_0071: concert.Resources_images → aws.AwsAmi
*maps_to: Concert container images map to AWS AMIs*

**Target fields:** aws_resource, virtual_machine_image

| Source Field | Maps To |
|-------------|---------|
| image_id | -- |
| resource_id | -- |

## itom_hashi_0072: concert.Resources_images → infragraph.BaseVirtualMachineImage
*maps_to: Concert container images map to Infragraph VM images*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| image_id | -- |
| resource_id | -- |

## itom_hashi_0073: concert.Credentials → aws.AwsIamRole
*maps_to: Concert credentials map to AWS IAM roles*

**Target fields:** assume_role_policy_document, aws_resource, description, last_used_date, last_used_region, max_session_duration, path, permissions_boundary_arn, permissions_boundary_type, role

| Source Field | Maps To |
|-------------|---------|
| created_by | -- |
| created_on | -- |
| credentials | -- |
| description | description |
| expiry | -- |
| id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | -- |
| type | -- |

## itom_hashi_0074: concert.Credentials → aws.AwsIamUser
*maps_to: Concert credentials map to AWS IAM users*

**Target fields:** aws_resource, user

| Source Field | Maps To |
|-------------|---------|
| created_by | -- |
| created_on | -- |
| credentials | -- |
| description | -- |
| expiry | -- |
| id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | -- |
| type | -- |

## itom_hashi_0075: concert.Vulnerability → packer.PackerSbom
*references: Concert vulnerabilities reference Packer SBOM (software bill of materials)*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, build_id, format, hcp_resource

| Source Field | Maps To |
|-------------|---------|
| access_point_count | -- |
| aliases | -- |
| application_count | -- |
| associations | -- |
| comment | -- |
| component_id | -- |
| component_type | -- |
| custom_score | -- |
| cve | -- |
| cvss | -- |
| deprioritized_findings | -- |
| description | base.description |
| exceptions_requested | -- |
| exceptions_requested_count | -- |
| findings | -- |
| has_exceptions_requested | -- |
| has_open_tickets | -- |
| has_unassessed_findings | -- |
| highest_finding_priority | -- |
| highest_finding_risk_score | -- |
| open_tickets | -- |
| osv_response_message | -- |
| package_name | -- |
| package_type | -- |
| package_version | -- |
| priority_1_findings | -- |
| priority_2_findings | -- |
| priority_3_findings | -- |
| related_cves | -- |
| severity | -- |
| state | -- |
| total_findings | -- |
| unassessed_findings | -- |
| vector | -- |
| vulnerability_findings_id | -- |
| wx_details | -- |
| wx_recommendation | -- |

## itom_hashi_0076: concert.Container_builds → packer.PackerBuild
*maps_to: Concert container builds map to Packer image builds*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, component_type, hcp_resource, packer_run_uuid, platform, source_external_identifier, status, version_id

| Source Field | Maps To |
|-------------|---------|
| build_number | -- |
| id | base.id |
| image_id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | base.name |
| sbom_id | -- |
| version | -- |

## itom_hashi_0077: concert.Container_builds → packer.PackerArtifact
*produces: Concert container builds produce Packer artifacts*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, build_id, external_identifier, hcp_resource, region

| Source Field | Maps To |
|-------------|---------|
| build_number | -- |
| id | base.id |
| image_id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| name | base.name |
| sbom_id | -- |
| version | -- |

## itom_hashi_0078: concert.Image_deployments → hcp_terraform.HcpTerraformRun
*triggers: Concert image deployments trigger Terraform runs*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, is_destroyed, message, status, workspace_id

| Source Field | Maps To |
|-------------|---------|
| deployment_number | -- |
| id | base.id |
| image_id | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| sbom_id | -- |

## itom_hashi_0079: concert.Topology_application_view → infragraph.Model
*maps_to: Concert application topology view maps to Infragraph graph model*

**Target fields:** aws_account, aws_ami, aws_availability_zone, aws_ec2_instance, aws_ec2_instance_type, aws_ec2_volume, aws_ecs_cluster, aws_ecs_service, aws_iam_policy, aws_iam_role, aws_iam_role_policy_attachment, aws_iam_user, aws_lambda_function, aws_network_interface, aws_region, aws_s3_bucket, aws_security_group, aws_security_group_rule, aws_subnet, aws_vpc, azure_virtual_machine, gcp_compute_instance, hcp_organization, hcp_project, hcp_terraform_agent_pool, hcp_terraform_organization, hcp_terraform_project, hcp_terraform_run, hcp_terraform_state_version, hcp_terraform_workspace, packer_ancestry, packer_artifact, packer_bucket, packer_build, packer_channel, packer_inherited_revocation, packer_registry, packer_sbom, packer_version

| Source Field | Maps To |
|-------------|---------|
| applications | -- |
| cves | -- |
| dast_exposures | -- |
| environments | -- |
| images | -- |
| private_access_points | -- |
| public_access_points | -- |
| repositories | -- |
| sast_exposures | -- |

## itom_hashi_0080: concert.Profile → aws.AwsIamPolicy
*maps_to: Concert compliance profiles map to AWS IAM policies*

**Target fields:** attachment_count, aws_resource, policy

| Source Field | Maps To |
|-------------|---------|
| additional_payload | -- |
| description | -- |
| last_updated_by | -- |
| last_updated_on | -- |
| profile_id | -- |
| sample_data_upload | -- |
| title | -- |
| version | -- |

## itom_hashi_0081: turbonomic.AttachedVirtualMachine → aws.AwsEc2Instance
*optimizes: Turbonomic optimizes AWS EC2 virtual machines*

**Target fields:** aws_resource, monitoring, private_ip, tenancy, virtual_machine

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0082: turbonomic.AttachedVirtualMachine → azure.AzureVirtualMachine
*optimizes: Turbonomic optimizes Azure virtual machines*

**Target fields:** azure_resource, extensions_time_budget, priority, virtual_machine

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0083: turbonomic.AttachedVirtualMachine → gcp.GcpComputeInstance
*optimizes: Turbonomic optimizes GCP compute instances*

**Target fields:** virtual_machine

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0084: turbonomic.AttachedVirtualMachine → infragraph.BaseVirtualMachine
*optimizes: Turbonomic optimizes virtual machines modeled in Infragraph*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, network_interface_ids, state, subnet_id, virtual_machine_image_id, virtual_machine_instance_type, vpc_id

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0085: turbonomic.ContainerClusterEntity → aws.AwsEcsCluster
*optimizes: Turbonomic optimizes AWS ECS container clusters*

**Target fields:** aws_resource, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0086: turbonomic.BusinessAccount → aws.AwsAccount
*maps_to: Turbonomic business account maps to AWS account*

**Target fields:** account, aws_resource

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| discoveredBy | -- |
| displayName | -- |
| environmentType | -- |
| staleness | -- |
| state | -- |
| tags | -- |
| uuid | -- |
| vendorIds | -- |

## itom_hashi_0087: turbonomic.BusinessAccount → infragraph.BaseAccount
*maps_to: Turbonomic business account maps to Infragraph base account*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, organization_id

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| discoveredBy | -- |
| displayName | -- |
| environmentType | -- |
| staleness | -- |
| state | -- |
| tags | base.tags |
| uuid | -- |
| vendorIds | -- |

## itom_hashi_0088: turbonomic.DataCenter → aws.AwsRegion
*maps_to: Turbonomic data center maps to AWS region*

**Target fields:** aws_resource, region

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0089: turbonomic.NamespaceEntity → aws.AwsVpc
*maps_to: Turbonomic Kubernetes namespace maps to AWS VPC network isolation*

**Target fields:** aws_resource, vpc

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| displayName | -- |
| uuid | -- |

## itom_hashi_0090: turbonomic.Action → hcp_terraform.HcpTerraformRun
*triggers: Turbonomic optimization actions trigger Terraform runs*

**Target fields:** base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, is_destroyed, message, status, workspace_id

| Source Field | Maps To |
|-------------|---------|
| actionID | -- |
| actionImpactID | -- |
| actionMode | -- |
| actionSchedule | -- |
| actionState | -- |
| actionType | -- |
| clearTime | -- |
| compoundActions | -- |
| createTime | -- |
| currentEntity | -- |
| currentLocation | -- |
| currentValue | -- |
| details | -- |
| displayName | -- |
| entityCharacteristics | -- |
| executionCharacteristics | -- |
| externalActionName | -- |
| externalActionUrl | -- |
| importance | -- |
| marketID | -- |
| newEntity | -- |
| newLocation | -- |
| newValue | -- |
| policy | -- |
| prerequisites | -- |
| prioritizedAutomationOpportunities | -- |
| relatedActionsCountByType | -- |
| relatedSettingsPolicies | -- |
| reservedInstance | -- |
| reservedInstancePlatform | -- |
| resizeAttribute | -- |
| risk | -- |
| source | -- |
| stats | -- |
| target | -- |
| template | -- |
| updateTime | -- |
| uuid | -- |
| valueUnits | -- |
| virtualDisks | -- |

## itom_hashi_0091: turbonomic.Template → hcp_terraform.HcpTerraformWorkspace
*maps_to: Turbonomic resource templates map to Terraform workspaces*

**Target fields:** agent_pool_id, base.account_id, base.created_at, base.description, base.id, base.name, base.region_id, base.tags, base.updated_at, base.zone_ids, current_run_id, current_state_version_id, environment, execution_mode, resource_count, tag_names

| Source Field | Maps To |
|-------------|---------|
| className | -- |
| discovered | -- |
| displayName | -- |
| enableMatch | -- |
| price | -- |
| uuid | -- |

## itom_hashi_0092: turbonomic.Statistics → infragraph.Base
*monitors: Turbonomic statistics monitor Infragraph base entities*

**Target fields:** account_id, created_at, description, id, name, region_id, tags, updated_at, zone_ids

| Source Field | Maps To |
|-------------|---------|
| date | -- |
| displayName | -- |
| epoch | -- |
| statistics | -- |

## itom_hashi_0093: turbonomic.Target → infragraph.Source
*maps_to: Turbonomic optimization target maps to Infragraph cloud source*

**Target fields:** SOURCE_AWS, SOURCE_AZURE, SOURCE_GCP, SOURCE_HCP, SOURCE_HCP_PACKER, SOURCE_HCP_TERRAFORM, SOURCE_UNSPECIFIED

| Source Field | Maps To |
|-------------|---------|
| aspects | -- |
| className | -- |
| connectedEntities | -- |
| discoveredBy | -- |
| displayName | -- |
| environmentType | -- |
| state | -- |
| tags | -- |
| uuid | -- |
| vendorIds | -- |

## itom_hashi_0094: turbonomic.CurrentEntity → infragraph.NodeIdentifier
*maps_to: Turbonomic current entity maps to Infragraph node identifier*

**Target fields:** id, kind, source

| Source Field | Maps To |
|-------------|---------|
| aspects | -- |
| className | -- |
| discoveredBy | -- |
| displayName | -- |
| environmentType | -- |
| state | -- |
| tags | -- |
| uuid | -- |
| vendorIds | -- |
