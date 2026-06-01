#!/bin/bash

curl localhost:8010/classify -X POST -H "Content-Type: application/json" \
  -d '{
  "text": "Eu briguei com um colega do trabalho e agora estou muito puto. Aquele cara é um canalha.",
  "top_k": 3,
  "multi_label": true
}'
