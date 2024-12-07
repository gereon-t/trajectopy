import inspect

from lazydocs import MarkdownGenerator

generator = MarkdownGenerator()


def create_alignment_docs():
    import trajectopy.api.alignment

    imports = [trajectopy.api.estimate_alignment, trajectopy.api.AlignmentResult, trajectopy.api.AlignmentData]

    markdown_docs = ""
    for obj in imports:
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Alignment.md", "w") as f:
        f.write(markdown_docs)


def create_evaluation_docs():
    import trajectopy.api.evaluation

    module_attributes = dir(trajectopy.api.evaluation)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.evaluation, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Evaluation.md", "w") as f:
        f.write(markdown_docs)


def create_matching_docs():
    import trajectopy.api.matching

    module_attributes = dir(trajectopy.api.matching)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.matching, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Matching.md", "w") as f:
        f.write(markdown_docs)


def create_merging_docs():
    import trajectopy.api.merging

    module_attributes = dir(trajectopy.api.merging)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.merging, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Merging.md", "w") as f:
        f.write(markdown_docs)


def create_plotting_docs():
    import trajectopy.api.plotting

    module_attributes = dir(trajectopy.api.plotting)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.plotting, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Plotting.md", "w") as f:
        f.write(markdown_docs)


def create_report_docs():
    import trajectopy.api.report

    module_attributes = dir(trajectopy.api.report)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.report, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Report.md", "w") as f:
        f.write(markdown_docs)


def create_result_docs():
    import trajectopy.api.result

    module_attributes = dir(trajectopy.api.result)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.result, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Result.md", "w") as f:
        f.write(markdown_docs)


def create_sorting_docs():
    import trajectopy.api.sorting

    module_attributes = dir(trajectopy.api.sorting)
    imports = {
        name: obj
        for name, obj in ((attr, getattr(trajectopy.api.sorting, attr)) for attr in module_attributes)
        if inspect.isfunction(obj) or inspect.isclass(obj)
    }

    markdown_docs = ""
    for _, obj in imports.items():
        markdown_docs += generator.import2md(obj, depth=2)

    with open("trajectopy-docs/docs/Documentation/Sorting.md", "w") as f:
        f.write(markdown_docs)


def create_trajectory_docs():
    from trajectopy.core.trajectory import Trajectory

    markdown_docs = generator.import2md(Trajectory, depth=2)
    with open("trajectopy-docs/docs/Documentation/Trajectory.md", "w") as f:
        f.write(markdown_docs)


def main():
    create_alignment_docs()
    create_evaluation_docs()
    create_matching_docs()
    create_merging_docs()
    create_plotting_docs()
    create_report_docs()
    create_result_docs()
    create_sorting_docs()
    create_trajectory_docs()


if __name__ == "__main__":
    main()
