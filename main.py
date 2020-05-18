from augmentation.framework import Framework
from augmentation.feature_selector import FeatureSelector

feature_selector = FeatureSelector(numeric_stat="pearson", numeric_threshold=0.5)
framework = Framework(feature_selector)
data = framework.run("../datasets/universities.csv", "name", "target", 4)
print("Done.")
