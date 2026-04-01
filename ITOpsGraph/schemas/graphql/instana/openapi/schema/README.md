# Deploy to Stepzen and test from this directory

## Make sure you have a .env in this directory with
```
STEPZEN_INSTANA_APITOKEN=YOUR_TOKEN
STEPZEN_INSTANA_HOSTNAME=ibmdevsandbox-instanaibm.instana.io
```

## Start Stepzen for local development
```
stepzen service start

stepzen login --config ~/.stepzen/stepzen-config.local.yaml
```

## Make sure you have a .env in this directory with
```
STEPZEN_INSTANA_APITOKEN=YOUR_TOKEN
STEPZEN_INSTANA_HOSTNAME=ibmdevsandbox-instanaibm.instana.io
```

## Deploy to Stepzen
```
stepzen deploy
```

## Test the Analyze Entities Query Operation
```
stepzen request -f infrastructure/op-infrastructure.graphql --operation-name=Entities --var-file infrastructure/entities-variables.json > entities.json
```

## Test the Topology Query Operation (warning: this takes a long time with large file)
```
stepzen request -f infrastructure/op-infrastructure.graphql --operation-name=Topology  > topology.json
```
