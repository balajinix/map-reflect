# Openapi graphql schema for Instana

### openapi/index.graphql is the graphql schema for the Instana REST API defined in build/openapi.json

This schema was build from the process in the build directory.

### JWT authorization.

When deploying the schema include a secret in your `.env` file,

e.g. I used a randomly generated hex-string, this is an example secret, not used in any deploy.

```
STEPZEN_JWT_SECRET=cc827feca9e30cd89ce3617cbf1e788c2674cad36f42b00abd06d56c9dac8eb0eacdcb75c1
```

Now you can generate JWT using the same secret, e.g.

```
jwtgen -v \
  --algorithm HS256 \
  --secret cc827feca9e30cd89ce3617cbf1e788c2674cad36f42b00abd06d56c9dac8eb0eacdcb75c1 \
  --exp 3600
```

jwtgen is from https://www.npmjs.com/package/jwtgen

Then set this header for requests

```
Authorization: Bearer generated-token-from-jwtgen
```

For example:

```
> jwtgen --algorithm HS256 --secret cc827feca9e30cd89ce3617cbf1e788c2674cad36f42b00abd06d56c9dac8eb0eacdcb75c1 --exp 3600
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3NDI0ODQ0MjYsImV4cCI6MTc0MjQ4ODAyNn0.DZP3y56ySLnK0mhkWogKTQ_ZolMRt_9SXEzyDbaqV0w
```

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3NDI0ODQ0MjYsImV4cCI6MTc0MjQ4ODAyNn0.DZP3y56ySLnK0mhkWogKTQ_ZolMRt_9SXEzyDbaqV0w
```
