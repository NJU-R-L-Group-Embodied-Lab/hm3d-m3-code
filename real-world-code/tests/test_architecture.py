"""Enforce package dependency boundaries without importing middleware or providers."""

import ast
import importlib.util
from pathlib import Path

import lavira_m3

PACKAGE = Path(lavira_m3.__file__).parent


def module_imports():
    """Resolve static imports, including function-local optional integrations."""
    modules = {}
    for path in PACKAGE.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(["lavira_m3", *parts])] = path
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                targets = [base, *(base + "." + alias.name for alias in node.names)]
            else:
                continue
            graph[name].update(target for target in targets if target in modules)
    return graph


def test_package_has_no_import_cycles():
    graph = module_imports()
    complete = set()

    def visit(name, chain):
        assert name not in chain, "Import cycle: " + " -> ".join([*chain, name])
        if name in complete:
            return
        for target in sorted(graph[name]):
            visit(target, [*chain, name])
        complete.add(name)

    for name in sorted(graph):
        visit(name, [])


def test_contracts_and_mapping_do_not_depend_on_workflows():
    graph = module_imports()
    assert not graph["lavira_m3.contracts"]
    for module, imports in graph.items():
        if module.startswith("lavira_m3.mapping."):
            assert not any(".decision." in target or ".adapters." in target for target in imports)
    workflow = graph["lavira_m3.decision.agent_loop"]
    assert "lavira_m3.decision.board_hub" not in workflow
    assert "lavira_m3.decision.nav_http" not in workflow


def test_pure_geometry_and_messages_do_not_import_external_clients():
    names = (
        "vision_geometry",
        "navigation_geometry",
        "path_geometry",
        "map_geometry",
        "model_messages",
        "navigation_results",
    )
    forbidden = {"rospy", "rclpy", "openai", "requests", "http", "urllib"}
    for name in names:
        tree = ast.parse((PACKAGE / "decision" / (name + ".py")).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                imports = [node.module or ""]
            else:
                continue
            assert not (set(item.split(".")[0] for item in imports) & forbidden), name
