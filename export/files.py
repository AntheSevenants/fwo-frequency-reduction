import os

def export_files(graphs, profile_name, export_folder):
    for graph_name in graphs:
        filename = "".join(["fig-", profile_name, "-", graph_name, ".png"])
        print(f"Writing {filename}")

        file_path = os.path.join(export_folder, filename)
        graphs[graph_name].figure.savefig(file_path)
