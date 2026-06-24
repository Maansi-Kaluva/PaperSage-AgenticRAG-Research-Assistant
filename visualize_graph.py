from backend.rag_graph import build_graph

graph = build_graph()

png_data = graph.get_graph().draw_mermaid_png()

with open("rag_graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved as rag_graph.png")