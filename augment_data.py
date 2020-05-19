from augmentation.framework import Framework
from augmentation.feature_selector import FeatureSelector

feature_selector_1 = FeatureSelector()
framework = Framework(feature_selector_1)
data = framework.run("../datasets/universities.csv", "name", "target", 100)
print("Augmentation 1 done.")

feature_selector_2 = FeatureSelector(numeric_stat="pearson")
framework = Framework(feature_selector_2)
data = framework.run("../datasets/universities.csv", "name", "target", 100)
print("Augmentation 2 done.")
