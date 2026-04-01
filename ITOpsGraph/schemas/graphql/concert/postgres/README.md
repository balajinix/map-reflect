# Development graphql schemas for Concert
---

## Pre-reqs
---
1. npm
2. stepzen
3. docker/podman

## Build process

1. Deploy a development instance of concert and expose the postgres container on any port.
2. The credentials for appdb can be grabbed from within the postgres container's environment prefixed by `APPDB_`
3. Use the following stepzen introspection command:
```graphql
stepzen import postgresql --name 'appdb' --db-host=IP-ADDRESS:PORT --db-database=appdb --db-schema=ibm_roja_app --db-user=<CONCERT-APPDB-USERNAME> --db-password=<CONCERT-APPDB-PASSWORD>
```
