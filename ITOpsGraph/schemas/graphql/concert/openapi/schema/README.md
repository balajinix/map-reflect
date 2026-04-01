# StepZen schema generated via LLM and OpenAPI specifications

## Get vulnerabilities via app name
```
stepzen request 'query {
  applications(filter: "name:Quote of the Day" )
  {
   applications{
    id
    name
     vulnerabilities {
       cve
       severity
       highest_finding_risk_score
       has_open_tickets
       wx_details
       wx_recommendation
     }
    }
  }
}'
```

## Get vulnerabilities via app id
```
stepzen request 'query {
  application (application_id: "65059c40-d3a2-4c92-89a5-a207fcf5e7cd") {
    id
    name
     vulnerabilities {
      cve
      exceptions_requested
      cvss
      highest_finding_risk_score
      description
      severity
      total_findings
      has_open_tickets
      open_tickets
      wx_details
      wx_recommendation
     }
  }
}'
```

## Get all vulnerabilities for all apps
```
stepzen request 'query { 
  applications(page_size: 5 ){
   applications{
    id
    name
     vulnerabilities {
       cve
       severity
       highest_finding_risk_score
       has_open_tickets
       wx_details
       wx_recommendation
     }
  }}
}'
```