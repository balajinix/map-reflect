# GraphQL schemas for Concert REST APIs built from OpenAPI specification

- build: assets used in the construction of the schema
- schema: The StepZen schema

## How to make the REST API calls in Concert

- Step 1 - get YOUR_TOKEN via `POST /core/api/v1/login`
```
curl --location 'https://$HOSTNAME:$PORT/core/api/v1/login' \
--header 'InstanceId: 0000-0000-0000-0000' \
--header 'Content-Type: application/json' \
--data '{ "username" : $USERNAME, "password": $PASSWORD}'
```

Example response:
```
{
    "is_authenticated": true,
    "token": YOUR_TOKEN,
    "instance_id": "0000-0000-0000-0000",
    "username": "ibm_roja_adm",
    "display_name": "ibm_roja_adm",
    "expiry": 1741985454,
    "roles": [
        "Admin"
    ],
    "quickstart": false
}
```

- Step 2 - Use the YOUR_TOKEN in subsequent calls
```
curl --location 'https://$HOSTNAME:$PORT/core/api/v1/users_groups' \
--header 'InstanceId: 0000-0000-0000-0000' \
--header 'Authorization: Bearer $YOUR_TOKEN' \
```

Example response:
```
{"users_groups":[{"id":"ibm_roja_adm","type":"user","role":"Admin"}]}
```