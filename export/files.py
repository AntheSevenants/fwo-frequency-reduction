import os

def export_files(graphs, profile_name, export_folder):
    for graph_name in graphs:
        filename = get_figure_filename(profile_name, graph_name)
        print(f"Writing {filename}")

        file_path = os.path.join(export_folder, filename)
        graphs[graph_name].figure.savefig(file_path)

def get_figure_filename(profile_name, graph_name):
    return "".join(["fig-", profile_name, "-", graph_name, ".png"])