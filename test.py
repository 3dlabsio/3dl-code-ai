from langchain_community.vectorstores import DeepLake

dataset = DeepLake(path="hub://shanewarner/test_dataset")
print(dataset)
