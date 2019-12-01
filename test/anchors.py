from efficient_det.utils import anchors


def tile_anchors_test():
    anchors_gen = anchors.AnchorGenerator(
        size=32,
        aspect_ratios=[.5, 1, 2],
        stride=8)
    
    boxes = anchors_gen.tile_anchors_over_feature_map()
    print(boxes)


tile_anchors_test()