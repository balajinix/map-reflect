# Development graphql schema for Turbonomic


Very quick and basic graphql API for Markets Actions

```
stepzen request 'query {marketActions {actionID } }'
stepzen request 'query q($m: ID!) {marketActions(market_uuid: $m)  {actionID } }' --var m=Market
stepzen request 'query {marketEntities {uuid displayName } }'

stepzen request 'query q($id: ID!) {entity(entity_uuid: $id) {uuid displayName } }' --var id=75703551344467

stepzen request 'query {entity(entity_uuid: 75703551344467) {uuid displayName } }'
stepzen request 'query q($id: ID!) {entityActions(entity_uuid: $id) {uuid displayName } }' --var id=75703528018065

stepzen request '{entityStats(entity_uuid: "75892917798240" ) { displayName }}'
stepzen request '{entityActionStats(entity_uuid: "75892917798240" ) { date statistics { name } }}'


# search
 stepzen request '{entitySearch(types: ["BusinessApplication"]  q: "robotshop-aws" query_type: 'EXACT' environment_type: 'HYBRID' limit: 1000)  {displayName uuid environmentType numRelatedVMs severity state vendorIds actions { source importance  uuid displayName risk { severity description subCategory } }} }'
Ah

# search with stats
    stepzen request '{entitySearch(types: ["BusinessApplication"]  q: "robotshop-aws" query_type: 'EXACT' environment_type: 'HYBRID' limit: 1000)  {displayName uuid environmentType numRelatedVMs severity state vendorIds stats { displayName date statistics { uuid name }} actions { displayName }} }'
```


### Notes

#### detail_level
The type structure was built using no `detail_level` and no difference between `STANDARD` and `EXECUTION` is apparent.  However, it's quite possible we're dropping some elements.   Fix by using OpenAPI to build the GraphQL type structure [LATER].

#### POST

Partial impl. in place as `marketActionsFiltered`.  When importing from OpenAPI, this should build out all the arguments properly, but it may be better to curate it and limit the number of inputs.

There is a POST version of /markets/{market_Uuid}/actions ( /apidoc/#/Markets/getActionsByMarketUuid ) that allows finer grained filtering of the actions.  Likely useful as you can do stuff like:
```
  "relatedEntityTypes": "Selecting a VCD, and specifying '\"related_entities\": [\"VirtualMachine\"] will include the actions for any VM in the given VDC",
  "targetEntityTypes": "Specifying '\"target_entities\": [\"VirtualMachine\"] will include the actions where the target entity is a VM.",
```

The filters are specified in the POST body.   Likely will need to be a JSON, but it would be nice if we could specify as a GraphQL type...

In addition, this will require finessing the URL query parameters (always set a value so they get specified).


# TODO
Use standard authentication model
- Currently: use a client_id and secret to generate a token to be used but eventually use a JWT that can access the system that was forwarded to the API
- Use standard pagination and filter models
