## WatsonX Orchestrate GraphQL Schema

[Main Docs](https://www.ibm.com/docs/en/watsonx/watson-orchestrate/current)

### Using the API - Getting Tokens:

[Authenticating to the API](https://developer.ibm.com/apis/catalog/watsonorchestrate--custom-assistants/Authenticating+to+the+API)

- [AWS](https://developer.ibm.com/apis/catalog/watsonorchestrate--custom-assistants/api/API--watsonorchestrate--generating-a-json-web-token#generatingjwt)

-  IBM Cloud - goto your instance of the Watson Orchestrate and get your API Key.

## NOTE!  The URL may not be correct!  

~~https://api.us-south.assistant-builder.watson.cloud.ibm.com/instances/7452da4b-822b-46b0-95be-cf52850b1f1d~~

`https://api.us-south.watson-orchestrate.cloud.ibm.com/instances/7452da4b-822b-46b0-95be-cf52850b1f1d`

<img width="1644" alt="image" src="https://github.ibm.com/solis-dal/solis-graphql-schemas/assets/1083/c8538869-2811-43c4-83c9-80d624f88b29">

### To find your Tenant ID ,  hover over the Launch button and it appears in the URL link at bottom:

<img width="1143" alt="image" src="https://github.ibm.com/solis-dal/solis-graphql-schemas/assets/1083/cb07d0a5-f1f4-435f-9145-77f8f1825939">

For example your .env might look like this:
```
STEPZEN_WXO_APIKEY=YOUR_APIKEY
STEPZEN_WXO_HOSTNAME=api.us-south.watson-orchestrate.cloud.ibm.com
STEPZEN_WXO_TENANTID=7452da4b-822b-46b0-95be-xxxxxxxxx
```

## WatsonX Orchestrate - chats and threads

Chats and threads support both AWS and IBM Cloud (based upon the hostname) [ skillsets not upgraded ]

For AWS, the .env file will look like:
```
STEPZEN_WXO_APIKEY=YOUR_APIKEY
STEPZEN_WXO_HOSTNAME=api.dl.watson-orchestrate.ibm.com
STEPZEN_WXO_TENANTID=7452da4b-822b-46b0-95be-xxxxxxxxx
```
and the APIKey will be used for HTTP Bearer based authentication.


### Chat

#### Create first

```
stepzen request  'mutation m { createChatRun(  message: {role: "user" content:  { response_type: "text" text: "What is the color red?  ." channels: [ "weather"]  }   }) { message_id run_id task_id thread_id}}'    | tee create.json
```
Extract the run_id
```
RUN_ID=$(jq -r  .data.createChatRun.run_id create.json )
THREAD_ID=$(jq -r .data.createChatRun.thread_id create.json )
```

#### Reuse to add to the thread:

```
stepzen request  'mutation m($tid: ID)  { createChatRun( thread_id: $tid message: {role: "user" content:  { response_type: "text" text: "What is the color green?  ." channels: [ "weather"]  }   }) { message_id run_id task_id thread_id}}'  --var tid=$THREAD_ID
```

#### List of all runs

```
stepzen request '{chatRuns { edges { cursor node { id status  }}pageInfo { startCursor endCursor hasNextPage  } }}'
```

#### Get run or  associated events
```
stepzen request  'query q( $i: ID ) { chatRunEvents(run_id: $i) { event id  data { last_error delta message { content { response_type text}}} } }'  --var i=$RUN_ID

stepzen request  'query q( $i: ID ) { chatRun(run_id: $i) { id status assistant_id thread_id created_at started_at failed_at completed_at cancelled_at llm_params last_error guardrails result { type next_run_id data { run_id delta thread_id message_id message { id role content {response_type text } }} }  } }'  --var i=$RUN_ID
```

#### Use subscriptions to follow
Currently this is of limited utility due to the definitions for chatRunEvents which underlie the subscription.
```
stepzen request 'subscription ($id: ID!) { chatEvents(run_id: $id) { event id data  { message_id delta run_id thread_id message { id role created_by created_on created_by_username }}}}' --var id=$ID
```

### Threads

List the threads and delete them.   This will also delete associated runs.

```
stepzen request '{threads { edges { cursor node { id tenant_id  parent_message_id additional_properties { wxa_message { output{ generic { response_type text }}}} mentions { assistant_name } } } pageInfo { startCursor endCursor hasNextPage  } }}'

stepzen request '{threads  }'
stepzen request 'mutation {deleteThread(thread_id: "575ce05c-602b-4299-84b8-7cf843effa01") { not_ok }}

```
## Pagination

A local guide to enabling GraphQL pagination with IBM watsonx Orchestrate.  Orchestrate is fairly consistent with offset pagination with "offset", "limit" query parameters that control pagination.  The result will be of the form:
```
{ 
  "data": [.... ]
  "total": 1, "limit": 10,  "offset": 0
}
```
data is the paginated data.  total is the total number of items.

### Converting to GraphQL node/edge

The steps are:
- Create the connection and edge.  You can use the following formula.  Just be sure to replace node's type with the proper type.  In this case `AssistantRun` is returned as an array of JSON for runs.
```
type AssistantRunConnection {
  edges: [AssistantRunEdge]
  pageInfo: PageInfo!
}
type AssistantRunEdge {
  node: AssistantRun
  cursor: String
}
```
- Add the query or mutation.  For example:
```
  _chatRuns(
    first: Int = 10
    after: String = ""): AssistantRunConnection
```
first and after must be present.  Your queries will then need to walk the edges and nodes as: `{_chatRuns { edges { cursor node { id status  }}pageInfo { startCursor endCursor hasNextPage  } }}`.   You can pass in `after=cursor` to go the next page.

- You'll then need to add in some directives that will automatically create the edges and nodes for you and extract the deata to the node properly.  See https://stepzen.com/docs/connecting-backends/rest-services-graphql-pagination and https://stepzen.com/docs/connecting-backends/rest-services-graphql-pagination 

- A little trick is needed to pass through first and after: pass the first as limit and offset as after "v1/orchestrate/runs?limit=$first&offset=$after"
  - after will be zero for the first call and first will be the whatever was set as the limit
  - use wxo-guard for the `stepzen.queryextensionguard`.   Some orchestrate endpoints will 
    accept '&first=$first&after=$after` (runs) but others will not (threads)
- You'll need to promote the underlying data one level so it matches the `node` type.  In this case, it is `AssistantRun`.  You'd add `resultroot: "data[]"` telling the pagination that nodes are in this array.  Indirection: the array is actually carried by the list of edges, so the type of node must not be an list.
- The pagination directive (see https://stepzen.com/docs/connecting-backends/rest-services-graphql-pagination ) must be set.
```
      pagination: {
        type: OFFSET
        setters: [{ field: "total", path: "total" }]
      }
``` 
Note: the setters are required to tell pagination where to find total.  It just happens to be that it can be found as `{"total": n}` in this case.
