#/bin/bash
echo "Fetch Concert for new token!"
cd ../schema
source .env

# Verify that the variables have been sourced
echo "Host: $STEPZEN_CONCERT_HOST"
echo "Username: $STEPZEN_CONCERT_USERNAME"
echo "Password: $STEPZEN_CONCERT_PASSWORD"

response=$(curl -k --location "https://$STEPZEN_CONCERT_HOST/core/api/v1/login" \
  --header "InstanceId: 0000-0000-0000-0000" \
  --header "Content-Type: application/json" \
  --data "{\"password\": \"$STEPZEN_CONCERT_PASSWORD\", \"username\": \"$STEPZEN_CONCERT_USERNAME\"}")

new_token=$(echo "$response" | jq -r '.token')

if [ -z "$new_token" ]; then
    echo "Error: Token extraction failed!"
    exit 1
fi

env_file=".env"
if [ ! -f "$env_file" ]; then
    echo ".env file not found!"
    exit 1
fi
sed -i '' "s/^STEPZEN_CONCERT_API_TOKEN=.*/STEPZEN_CONCERT_API_TOKEN=\"$new_token\"/" "$env_file"

echo "Token has been updated successfully in $env_file"

echo "Redeploy to Stepzen!"

stepzen deploy
