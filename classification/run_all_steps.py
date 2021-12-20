from classification import step_1_0_trivial_approach, step_1_1_trivial_approach_with_artifact_replacement, \
    step_2_0_ncv_multiple_algorithms, step_3_ncv_multiple_algorithms_weighted_classes, \
    step_4_ensembles, step_Z_plot_evaluation
from helpers.Logger import Logger

log = Logger()


def main():
    log.s('step_1_0_trivial_approach.py')
    step_1_0_trivial_approach.main()

    log.s('step_1_1_trivial_approach_with_artifact_replacement.py')
    step_1_1_trivial_approach_with_artifact_replacement.main()

    log.s('step_2_0_ncv_multiple_algorithms.py')
    step_2_0_ncv_multiple_algorithms.main()

    # log.s('step_2_1_ncv_multiple_algorithms_without_artifact_replacement.py')
    # step_2_1_ncv_multiple_algorithms_without_artifact_replacement.main()
    #
    # log.s('step_2_2_ncv_multiple_algorithms_with_artifact_replacement.py')
    # step_2_2_ncv_multiple_algorithms_with_artifact_replacement.main()

    log.s('step_3_ncv_multiple_algorithms_weighted_classes.py')
    step_3_ncv_multiple_algorithms_weighted_classes.main()

    log.s('step_4_ensembles.py')
    step_4_ensembles.main()

    log.s('step_Z_plot_evaluation.py')
    step_Z_plot_evaluation.main()


if __name__ == "__main__":
    main()


