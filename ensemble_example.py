import pandas
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from classification.cross_project.cross_project_evaluation import select_best_classifiers
from classification.default_classifiers import classifiers
from classification.defaults import make_default_pipeline, make_default_gridsearch_parameters, \
    make_classifier_specific_gridsearch_parameters
from classification.utils import load_dataset, get_classifier_by_name, nested_cross_validation, \
    class_f1_scorer, unpandas_parameters, report_classifier_performance
from helpers.Logger import Logger


log = Logger()


def main():
    docs, target, target_names = load_dataset()
    train_docs, validation_docs, train_target, validation_target = train_test_split(docs, target, test_size=0.2, random_state=42)

    nested_cross_val_classifiers = nested_cross_validation_for_model_selection(train_docs, train_target, target_names)

    num_classifiers_per_cat = 1
    ensembe_performance = ensemble_best_classifiers(nested_cross_val_classifiers, num_classifiers_per_cat, train_docs, train_target, validation_docs, validation_target, target_names)
    print(ensembe_performance.to_string())


def ensemble_best_classifiers(nested_cross_val_classifiers, num_classifiers_per_cat, train_docs, train_target, validation_docs, validation_target, target_names):
    classifiers_df = select_best_classifiers(num_classifiers_per_cat, target_names, nested_cross_val_classifiers)
    estimators = []
    for split, target, parameters, classifier in zip(classifiers_df['split_id'], classifiers_df['target'], classifiers_df['params'], classifiers_df['classifier']):
        name = target + str(split) + classifier
        pipeline = make_default_pipeline(get_classifier_by_name(classifier))
        pipeline.set_params(**unpandas_parameters(parameters))
        estimators.append((name, pipeline))

    ensemble_name = 'EnsembleTop' + str(num_classifiers_per_cat)
    final_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    final_params = {}

    final_clf.fit(train_docs, train_target)
    y_predicted = final_clf.predict(validation_docs)

    report = report_classifier_performance(validation_target, y_predicted, target_names, 0, final_params, final_params)
    log.s(ensemble_name)
    log.s(str(report))
    df = pandas.DataFrame(report)
    return df


def nested_cross_validation_for_model_selection(train_docs, train_target, target_names):
    classifiers_df = pandas.DataFrame()

    for bug_cat in target_names:
        bug_cat_report_df = pandas.DataFrame()

        for clf_name in classifiers:
            gs_params = make_default_gridsearch_parameters()
            gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))

            if clf_name != 'MultinomialNB(fit_prior=False)':
                gs_params.update({'clf__class_weight': [{target_names.index(bug_cat): 4}]})
            pipeline = make_default_pipeline(get_classifier_by_name(clf_name))
            scorer = make_scorer(class_f1_scorer, class_idx=target_names.index(bug_cat))
            ncv_results = nested_cross_validation(train_docs, train_target, target_names, None, pipeline,
                                                  gs_params, scoring=scorer)
            bug_cat_report_df = bug_cat_report_df.append(ncv_results)

        bug_cat_report_df = bug_cat_report_df.sort_values('F1_' + bug_cat)
        bug_cat_report_df['target'] = bug_cat
        classifiers_df = classifiers_df.append(bug_cat_report_df)

    return classifiers_df


if __name__ == "__main__":
    main()
