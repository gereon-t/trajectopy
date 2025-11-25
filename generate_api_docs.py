from lazydocs import MarkdownGenerator

generator = MarkdownGenerator()


def create_tools_docs():
    from trajectopy.processing import (
        alignment,
        approximation,
        evaluation,
        interpolation,
        matching,
        merging,
        sorting,
    )

    markdown_docs = generator.import2md(alignment, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Alignment.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(approximation, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Approximation.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(evaluation, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Evaluation.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(interpolation, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Interpolation.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(matching, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Matching.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(merging, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Merging.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)

    markdown_docs = generator.import2md(sorting, depth=2)
    with open("trajectopy-docs/docs/Documentation/Tools/Sorting.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_results_docs():
    import trajectopy.results

    markdown_docs = generator.import2md(trajectopy.results, depth=2)

    with open("trajectopy-docs/docs/Documentation/Results.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_trajectory_docs():
    from trajectopy.core.trajectory import Trajectory

    markdown_docs = generator.import2md(Trajectory, depth=2)
    with open("trajectopy-docs/docs/Documentation/Trajectory.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_parameter_docs():
    import trajectopy.processing.lib.alignment.parameters

    markdown_docs = generator.import2md(trajectopy.processing.lib.alignment.parameters, depth=2)
    with open("trajectopy-docs/docs/Documentation/Parameters.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def create_settings_docs():
    import trajectopy.core.settings

    markdown_docs = generator.import2md(trajectopy.core.settings, depth=2)
    with open("trajectopy-docs/docs/Documentation/Settings.md", "w", encoding="utf-8") as f:
        f.write(markdown_docs)


def main():
    create_tools_docs()
    create_results_docs()
    create_trajectory_docs()
    create_parameter_docs()
    create_settings_docs()


if __name__ == "__main__":
    main()
