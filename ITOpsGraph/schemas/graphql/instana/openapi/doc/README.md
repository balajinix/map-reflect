# Instana GraphQL API using REST APIs

## Overview

This demonstrates use of a StepZen against an existing REST api to produce a GraphQL schema for Instana:

- Data model is represented as a graph
- GraphQL api use the recommended style:
  - GraphQL pagination
  - GraphQL filtering
- Additional capability in the API not (easily?) exposed by the REST api.

## Opinionated GraphQL style.

We believe that GraphQL apis should have a consistent style:

- to allow application developers to have the same data access style regardless of the backend, e.g. a database or REST API fronted by GraphQL look the same
- to match the training of LLMs, in some ways simplifying the schema for the LLMs to have a single style

See: https://github.ibm.com/stepzen-graphql/tenets/blob/main/schemas.md

## Caveat

This GraphQL schema has been created without any real knowledge of Instana, it's intended to show concepts, not details.

## High-level implementation

StepZen GraphQL schema using its `@rest` and `@materializer` directives to expose Instana data.

![architecture](https://media.github.ibm.com/user/425393/files/5b9f311e-4738-453c-93d1-3d40433f756f?token=AADH3MLDBG7A6NXCTIUXVFTH3XD4Q)

https://instana.github.io/openapi/

## Concepts

### Graph

The objects are represented as a graph, through the GraphQL type system, so that:

- Applications have
  - Services which have
    - Endpoints
    - Connections

Through GraphQL selection a client can pull out what it needs in a single GraphQL request.

**Under the covers StepZen executes the required calls against the Instana REST API, hiding any complexity from the application.**

_For example, get applications and its services, connections and endpoints may result in say 15 calls to Instana REST api,
but it's just a single request for the application._

This reduces the potential for hallucinations with an LLM by having relationships defined in a graphl, instead of the
LLM say joining 15 tool calls together, with a chance for a hallucination at each hop.

(filtering details have been omitted in these initial examples)

**Fetch applications**

```graphql
query {
  applications {
    id
    label
    services {
      id
      label
    }
  }
}
```

**Applications, services and endpoints**

```graphql
query {
  applications {
    id
    label
    services {
      id
      label
      endpoints {
        id
        label
      }
    }
  }
}
```

**Applications and service connections**

```graphql
query {
  applications {
    label
    services {
      label
      incoming {
        from
        latency
      }
      outgoing {
        to
        latency
      }
    }
  }
}
```

### Filtering

StepZen uses a filtering style that is common in GraphQL implementations and is inspired/based upon Mongodb filtering.

We have found that LLMs are able to generate the filter "input objects" to represent the required data based upon natural language,
for example "show me the customer with email alice@secure.com" might result in a LLM generated GraphQL request such as:

```graphql
query customers(filter:{email:{eq:"alice@secure.com"}}) {
    name
    email
    since
    address {
        city
        street
    }
}
```

Filtering is defined by GraphQL input objects and the schema developer is able to constrain filtering as needed,
for example only a range on dates.

[Filtering docs for `@dbquery`](https://www.ibm.com/docs/en/api-connect/graphql/saas?topic=directives-directive-dbquery#filtering__title__1)

StepZen takes the filter definition and where possible can push it down to the backend, including databases (SQL),
document stores (mongodb, elastic search), GraphQL backends and with some additional schema development REST backends.

Thus:

- the schema developer (for most backends) doesn't have to understand how filtering is implemented in the backend
- the application developer doesn't even know what the backend is, they have a consistent filtering style.
- an LLM has a consistent style for selecting values

For Instana we implemented filtering for:

- label
- boundry scope
- timeframe - implemented as a mandatory date time range

GraphQL operation:

```
query Applications($f: ApplicationFilter!, $n: Int, $a: String) {
  applications(first: $n, filter: $f, after: $a) {
    edges {
      node {
        id
        label
        boundaryScope
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

with a filter of:

```
{
  "f": {
    "label": {"eq": "RobotShop"},
    "timeFrame": {
      "ge": "2025-03-18T15:10:57Z",
      "le": "2025-03-18T20:10:57Z"

    }
  }
}
```

Filtering can be at multiple levels, here filtering on application (variable `$f`) and services (inline):

```
query Applications($f: ApplicationFilter!, $n: Int, $a: String) {
  applications(first: $n, filter: $f, after: $a) {
    edges {
      node {
        id
        label
        boundaryScope
        services(filter:{label:{eq:"users"}}) {
          id
          label
          technologies
          types
          endpoints {
            id
            label
            technologies
            type
            syntheticType
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### Pagination

StepZen allows fields to have a consistent pagination style regardless of backend, based upon the "replay connections" spec.

Spec: https://relay.dev/graphql/connections.htm

The root field `Query.applications` uses pagination and returns an `ApplicationConnection`.

There is a minor fix that StepZen needs to fully support pagination with Instana REST api,
as StepZen supports multiple modes of REST pagination, but Instana does not return the
total number of edges (records) that match the predicate. Thus `Query.applications` just
returns the first N items, driven by the field's `first` argument.

### Types

Use of standard consistent types is preferred, typically this becomes an issue with timestamps.

By using a standard `DateTime` GraphQL scalar (or `Date`) applications need not be concerned with
questions about is this a unix second epoch time, milliseconds, random string format etc. etc.

For this instana schema timeFrames are represented as pair of `DateTime` types.

## Try it out

The schema definitions are in:
https://github.ibm.com/solis-dal/solis-graphql-schemas/tree/main/products/dev/instana/openapi/schema/applications

That focuses on "applications", **work is ongoing** to produce similar "curated" schema for other Instana elements, such as "infrastruture".

The GraphQL endpoint is live (as an example) at:

https://watzen.us-east-a.ibm.stepzen.net/solis-dal/instana/graphql

going against `ibmdevsandbox-instanaibm.instana.io`

StepZen (API Connect for GraphQL)'s dashboard can be used here:

https://dashboard.ibm.stepzen.com/explorer?endpoint=https%3A%2F%2Fwatzen.us-east-a.ibm.stepzen.net%2Fsolis-dal%2Finstana%2Fgraphql

A JWT is needed for authorization, ping @dan.debrunner on slack.

`Authorization: Bearer <token>`

![header](https://github.ibm.com/solis-dal/solis-graphql-schemas/assets/425393/97477918-9142-4ec0-836c-5b2b72781bc8)
