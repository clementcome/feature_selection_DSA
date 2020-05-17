from augmentation.framework import Framework

framework = Framework()
data = framework.run("../datasets/universities.csv",
                     "name", "target", 3)
print("Done.")
