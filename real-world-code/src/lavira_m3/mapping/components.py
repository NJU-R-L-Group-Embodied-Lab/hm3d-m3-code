"""Remove occupied 8-connected components smaller than three cells."""


def remove_small_components(cells):
    unseen = {key for key, state in cells.items() if state == 2}
    while unseen:
        component = [unseen.pop()]
        for x, y in component:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x + dx, y + dy)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
        if len(component) < 3:
            for key in component:
                del cells[key]
