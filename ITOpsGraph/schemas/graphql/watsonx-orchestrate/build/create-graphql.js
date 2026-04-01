const openapi_1 = require("@omnigraph/openapi");
const gql = require("graphql");

const config = {
  source: "redocly-openapi.yaml",
};

let it = openapi_1
  .loadGraphQLSchemaFromOpenAPI("mine", config)
  .then((it) => {
    console.log(gql.printSchema(it));
  })
  .catch((error) => {
    console.error("Error:", error);
  });
