#!/bin/bash

./node_modules/.bin/redocly bundle ./openapi.json --remove-unused-components --dereferenced -o redocly-openapi.yaml
node create-graphql.js > omnigraph.graphql
