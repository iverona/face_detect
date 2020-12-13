import json
import objectpath

json_data = json.load(open('response.json'))
json_tree = objectpath.Tree(json_data)

result = json_tree.execute("$..*['timestampedObjects']")
entries = {}

for i in result:
    for j in i:
        entries[j['timeOffset']] = { 'attributes': j['attributes'], 'normalizedBoundingBox': j['normalizedBoundingBox']}       

# print(entries['80.080s']['normalizedBoundingBox'])