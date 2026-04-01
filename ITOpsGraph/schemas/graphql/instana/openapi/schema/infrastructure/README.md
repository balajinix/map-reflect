# To call the get infrastructure entities endpoint

## Example 1 Input
```
{
  "input": {
    "type": "host",
    "pagination": {
        "retrievalSize": 200
    },
    "metrics": [
        {
            "metric": "cpu.used",
            "aggregation": "MEAN",
            "granularity": 120000
        },
        {
            "metric": "memory.used",
            "aggregation": "MEAN",
            "granularity": 120000
        }
    ],
    "timeFrame": {
        "windowSize": 600000,
        "to": 1741339320000
    },
    "tagFilterExpression": {
        "type": "TAG_FILTER",
        "entity": "NOT_APPLICABLE",
        "name": "label",
        "operator": "NOT_EMPTY",
        "value": ""
    }
  }
}
```

## RESP 1
```
{
  "data": {
    "entities": {
      "totalHits": 246,
      "canLoadMore": true,
      "adjustedTimeframe": null,
      "items": [
        {
          "label": "BackendResponse1.fyre.ibm.com",
          "snapshotId": "jpn24m-0tejuPOMl7yaBh4Xa_xo",
          "time": 1742471940000,
          "metrics": {
            "cpu.used.MEAN.120000": [
              [
                1741338720000,
                "0.0185833333333334"
              ],
              [
                1741338840000,
                "0.0195833333333334"
              ],
              [
                1741338960000,
                "0.0193333333333334"
              ],
              [
                1741339080000,
                "0.0169166666666667"
              ],
              [
                1741339200000,
                "0.0189166666666667"
              ]
            ],
            "memory.used.MEAN.120000": [
              [
                1741338720000,
                "0.418571482439839"
              ],
              [
                1741338840000,
                "0.418517538194795"
              ],
              [
                1741338960000,
                "0.418543695753726"
              ],
              [
                1741339080000,
                "0.418557149781587"
              ],
              [
                1741339200000,
                "0.418554147794418"
              ]
            ]
          },
          "tags": {
            "id.host": "jpn24m-0tejuPOMl7yaBh4Xa_xo",
            "label.host": "BackendResponse1.fyre.ibm.com"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "HKTLSRBAST03",
          "snapshotId": "NUkiVZ0WgRzHUKrR7U6_L2ZZdHs",
          "time": 1741384800000,
          "metrics": {
            "cpu.used.MEAN.120000": [
              [
                1741338720000,
                "0.0249166666666667"
              ],
              [
                1741338840000,
                "0.0266666666666667"
              ],
              [
                1741338960000,
                0.02725
              ],
              [
                1741339080000,
                0.03125
              ],
              [
                1741339200000,
                0.027
              ]
            ],
            "memory.used.MEAN.120000": [
              [
                1741338720000,
                "0.792982481329295"
              ],
              [
                1741338840000,
                "0.792951059620911"
              ],
              [
                1741338960000,
                "0.792975017605273"
              ],
              [
                1741339080000,
                "0.792967730503727"
              ],
              [
                1741339200000,
                "0.792959491919807"
              ]
            ]
          },
          "tags": {
            "id.host": "NUkiVZ0WgRzHUKrR7U6_L2ZZdHs",
            "label.host": "HKTLSRBAST03"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "IBM-5Q08B54.mshome.net",
          "snapshotId": "QD6T2ItYzpD_yvZSUXdeI2lYnAE",
          "time": 1741342740000,
          "metrics": {
            "cpu.used.MEAN.120000": [
              [
                1741338720000,
                0.05375
              ],
              [
                1741338840000,
                "0.0519166666666667"
              ],
              [
                1741338960000,
                "0.0551666666666667"
              ],
              [
                1741339080000,
                "0.0548333333333334"
              ],
              [
                1741339200000,
                "0.0564166666666667"
              ]
            ],
            "memory.used.MEAN.120000": [
              [
                1741338720000,
                "0.540983466788707"
              ],
              [
                1741338840000,
                "0.540804135119766"
              ],
              [
                1741338960000,
                "0.532397506628709"
              ],
              [
                1741339080000,
                "0.531687281997155"
              ],
              [
                1741339200000,
                "0.532661097766652"
              ]
            ]
          },
          "tags": {
            "id.host": "QD6T2ItYzpD_yvZSUXdeI2lYnAE",
            "label.host": "IBM-5Q08B54.mshome.net"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        . . . 
```

## Example 2 Input
```
{
  "input": {
    "type": "jvmRuntimePlatform",
    "timeFrame": {
      "to": 1719464755615,
      "windowSize": 117830000
    },
    "pagination": {
      "retrievalSize": 10
    },
    "tagFilterExpression": {
      "type": "TAG_FILTER",
      "entity": "NOT_APPLICABLE",
      "name": "label",
      "operator": "CONTAINS",
      "value": "app.jar"
    },
    "metrics": [
      {
        "metric": "cpu.usage",
        "aggregation": "MEAN"
      },
      {
        "metric": "memory.usage",
        "aggregation": "MAX"
      }
    ],
    "order": {
      "by": "cpu.usage",
      "direction": "DESC"
    }
  }
}
```

## RESP 2 :
```
{
  "data": {
    "entities": {
      "totalHits": 6,
      "canLoadMore": false,
      "adjustedTimeframe": null,
      "items": [
        {
          "label": "/app/app.jar",
          "snapshotId": "oy8YkUsTIlNKS1Kjc61VdDnbX8o",
          "time": 1719542520000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "oy8YkUsTIlNKS1Kjc61VdDnbX8o",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "/app/app.jar",
          "snapshotId": "g12rtuu9n3wY0-_7njWHHC-g1aI",
          "time": 1719539160000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "g12rtuu9n3wY0-_7njWHHC-g1aI",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "/app/app.jar",
          "snapshotId": "bZQRlpdEmDKQ_xydL0gWllUKSmQ",
          "time": 1719910980000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "bZQRlpdEmDKQ_xydL0gWllUKSmQ",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "/app/app.jar",
          "snapshotId": "PqmFHF3XJVcUF-Jcb0xkrMgfkv4",
          "time": 1719542520000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "PqmFHF3XJVcUF-Jcb0xkrMgfkv4",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "/app/app.jar",
          "snapshotId": "OFSOgDaoT8YFB48xGjIFKXS-rrc",
          "time": 1719542520000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "OFSOgDaoT8YFB48xGjIFKXS-rrc",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        },
        {
          "label": "/app/app.jar",
          "snapshotId": "3mY5YeMZFTa43m69R-6nsbD2tmE",
          "time": 1719910980000,
          "metrics": {},
          "tags": {
            "id.jvmRuntimePlatform": "3mY5YeMZFTa43m69R-6nsbD2tmE",
            "label.jvmRuntimePlatform": "/app/app.jar"
          },
          "entityHealthInfo": {
            "maxSeverity": 0,
            "openIssues": []
          }
        }
      ]
    }
  }
}
```