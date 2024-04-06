import yaml

with open("submodels/fundamentals.yaml", "r") as file:
    test = yaml.safe_load(file)

print(test)
