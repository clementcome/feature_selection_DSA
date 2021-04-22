from augmentation.feature_selector import FeatureSelector
from augmentation.framework import Framework

name_query_target_list = [
    # ("universities", "name", "target"),
    # ("presidential", "County", "Votes"),
    ("movie", "movie_title", "imdb_score"),
    # ("pageviews", "name", "visit"),
    # ("worldcitiespop", "City", "Population"),
]

for name, query, target in name_query_target_list:

    print("---")
    print(f"Augmenting the {name} dataset")

    # feature_selector_1 = FeatureSelector()
    # framework = Framework(feature_selector_1)
    # data = framework.run(f"../datasets/{name}.csv", query, target, 100)
    # print("Augmentation 1 done.")

    feature_selector_2 = FeatureSelector(
        numeric_stat="pearson",
        categoric_stat="anova",
        select_strategy="k_best",
        k_best=10,
    )
    framework = Framework(feature_selector_2)
    data = framework.run(f"../datasets/{name}.csv", query, target, 10)
    print("Augmentation 2 done.")

    feature_selector_3 = FeatureSelector(
        numeric_stat="pearson",
        categoric_stat="anova",
        select_strategy="k_best_min_max",
    )
    framework = Framework(feature_selector_3)
    data = framework.run(f"../datasets/{name}.csv", query, target, 10)
    print("Augmentation 3 done.")
