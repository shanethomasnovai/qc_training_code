
def get_classes(loader):
    classes = []
    for _, labels in loader:
        for label in labels:
            if label.item() not in classes:
                classes.append(label.item())
    return classes


def get_class_counts(loader):
    class_counts = {}
    for _, labels in loader:
        for label in labels:
            if label.item() not in class_counts:
                class_counts[label.item()] = 0
            class_counts[label.item()] += 1
    return class_counts
