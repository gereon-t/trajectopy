from lazydocs import MarkdownGenerator

generator = MarkdownGenerator()


def create_alignment_docs():
    import trajectopy.alignment

    markdown_docs = generator.import2md(trajectopy.alignment, depth=2)

    with open("trajectopy-docs/docs/Documentation/Alignment.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_evaluation_docs():
    import trajectopy.evaluation

    markdown_docs = generator.import2md(trajectopy.evaluation, depth=2)

    with open("trajectopy-docs/docs/Documentation/Evaluation.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_matching_docs():
    import trajectopy.matching

    markdown_docs = generator.import2md(trajectopy.matching, depth=2)

    with open("trajectopy-docs/docs/Documentation/Matching.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_merging_docs():
    import trajectopy.merging

    markdown_docs = generator.import2md(trajectopy.merging, depth=2)

    with open("trajectopy-docs/docs/Documentation/Merging.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_plotting_docs():
    import trajectopy.plotting

    markdown_docs = generator.import2md(trajectopy.plotting, depth=2)

    with open("trajectopy-docs/docs/Documentation/Plotting.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_report_docs():
    import trajectopy.report

    markdown_docs = generator.import2md(trajectopy.report, depth=2)

    with open("trajectopy-docs/docs/Documentation/Report.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_ate_result_docs():
    import trajectopy.core.evaluation.ate_result

    markdown_docs = generator.import2md(trajectopy.core.evaluation.ate_result, depth=2)

    with open("trajectopy-docs/docs/Documentation/ATEResult.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_rpe_result_docs():
    import trajectopy.core.evaluation.rpe_result

    markdown_docs = generator.import2md(trajectopy.core.evaluation.rpe_result, depth=2)

    with open("trajectopy-docs/docs/Documentation/RPEResult.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_sorting_docs():
    import trajectopy.sorting

    markdown_docs = generator.import2md(trajectopy.sorting, depth=2)

    with open("trajectopy-docs/docs/Documentation/Sorting.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_trajectory_docs():
    from trajectopy.trajectory import Trajectory

    markdown_docs = generator.import2md(Trajectory, depth=2)
    with open("trajectopy-docs/docs/Documentation/Trajectory.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def main():
    create_alignment_docs()
    create_evaluation_docs()
    create_matching_docs()
    create_merging_docs()
    create_plotting_docs()
    create_report_docs()
    create_ate_result_docs()
    create_rpe_result_docs()
    create_sorting_docs()
    create_trajectory_docs()


if __name__ == "__main__":
    main()
