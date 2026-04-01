# Apptio GraphQL Schema

## Test the queries

### Get Cluster information based on time frame
```
stepzen request -f clusters/op-clusters.graphql --var-file clusters/cluster-query-parms.json > clusters.json
```

### Get Rightsizing EC2 Recommendations
```
stepzen request -f recommendations/op-recommendations.graphql > recommendations.json
```

### Get Budgets
```
stepzen request -f budgets/op-budgets.graphql > budgets.json
```