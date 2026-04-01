# Instana GraphQL Builder for Watson WxFlows (StepZen)

## Build project
```
npm i
```

## To create a base schema index.graphql, proceed as follows:

### Create 'omnigraph.graphql', the Instana GraphQL schema file from openapi.json

```
./build.sh
```
This will create a file called omnigraph.graphql.  You'll want to make lots of manual change to this to accurately represent the Instana openapi.json. 

This process has already been done and the file placed via `cp omnigraph.graphql ../openapi/index.graphql`

### Deploy to your stepzen account via wxflows (from the root directory).  You'll need your .env
```
cd ..
wxflows deploy
```
