"""regen_project_graph.py — re-cluster + regenerate the PROJECT graph outputs (graph.json, GRAPH_REPORT.md,
graph.html) from the current graphify-out/graph.json after the session2-8 node additions (310 nodes)."""
import os, json
from networkx.readwrite import json_graph
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "graphify-out"))
GJ = os.path.join(OUT, "graph.json")


def main():
    data = json.loads(open(GJ, encoding="utf-8").read())
    try:
        G = json_graph.node_link_graph(data, edges="links")   # networkx >= 3.2
    except TypeError:
        G = json_graph.node_link_graph(data)                  # older networkx (default link='links')
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {c: "Community " + str(c) for c in communities}
    questions = suggest_questions(G, communities, labels)
    detection = {"total_files": 0, "total_words": 99999, "needs_graph": True, "warning": None,
                 "files": {"code": [], "document": [], "paper": []}}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      {"input": 0, "output": 0}, OUT, suggested_questions=questions)
    open(os.path.join(OUT, "GRAPH_REPORT.md"), "w", encoding="utf-8").write(report)
    to_json(G, communities, GJ, force=True)
    if G.number_of_nodes() <= 5000:
        to_html(G, communities, os.path.join(OUT, "graph.html"), community_labels=labels)
    print(f"project graph regenerated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(communities)} communities -> {OUT}")


if __name__ == "__main__":
    main()
