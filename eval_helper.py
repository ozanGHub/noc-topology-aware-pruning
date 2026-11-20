from noc_mapper import NoCMapper

# Teeny helper functions

def eval_noc(graph, *, P: int, N: int, mesh_rows: int, mesh_cols: int, visualize: bool = False, title: str = None, metrics=None):
    noc = NoCMapper(P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols)
    noc.run(graph)

    print("Mapping:", noc.mapping)  # [(task_id, core_id), ...]
    print("Communication cost:", noc.cost)

    if visualize:
        noc.visualize_all(title, metrics=metrics)

    return noc


def eval_nn(model, *, X_test, y_test, device, timings, evaluate_fn):
    report = evaluate_fn(model, X_test, y_test, device, timings)
    print(report)
    return report
