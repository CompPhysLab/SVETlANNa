import pytest
import torch
from svetlanna.detector import SimpleDetector, DetectorProcessorClf


def test_simple_detector():
    detector = SimpleDetector()
    assert isinstance(detector, torch.nn.Module)
    input_field = torch.rand(size=[7, 9])
    detector_image = detector.forward(input_field)
    assert input_field.shape == detector_image.shape
    assert torch.allclose(detector_image, input_field.abs().pow(2))


@pytest.mark.parametrize(
    "num_classes, detector_x, expected_mask", [
        (4, 8, [[0, 0, 1, 1, 2, 2, 3, 3]]),         # num_classes - even, detector_x - even
        (2, 4, [[0, 0, 1, 1]]),
        (2, 7, [[0, 0, 0, -1, 1, 1, 1]]),           # num_classes - even, detector_x - odd
        (4, 7, [[-1, 0, 1, -1, 2, 3, -1]]),
        (3, 8, [[-1, 0, 0, 1, 1, 2, 2, -1]]),       # num_classes - odd, detector_x - even
        (3, 10, [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]]),
        (3, 7, [[0, 0, 1, 1, 1, 2, 2]]),            # num_classes - odd, detector_x - odd
        (5, 7, [[-1, 0, 1, 2, 3, 4, -1]]),
        (5, 11, [[0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]]),
    ]
)
def test_detector_segmentation_strips(num_classes, detector_x, expected_mask):
    """
    Test of a method of DetectorProcessorClf for strips segmentation of a detector by classes zones.
    ...

    Parameters
    ----------
    num_classes
        Number of classes.
    detector_x
        Horizontal size of a detector.
    expected_mask
        Expected segmentation by strips.
    """
    processor = DetectorProcessorClf(num_classes)
    assert isinstance(processor, torch.nn.Module)

    segmentation = processor.detector_segmentation(torch.Size([1, detector_x]))

    for ind_class in range(num_classes):  # check if all classes zones are marked
        assert ind_class in segmentation

    assert torch.allclose(segmentation, torch.tensor(expected_mask, dtype=torch.int32))


@pytest.mark.parametrize(
    "num_classes, segmented_detector, expected_weights", [
        (2, [[0, 0, 1, 1, 0, 1, 0, 0]], [[3 / 5, 1.0]]),
        (3, [[-1, -1, 0, 0, 0, 1, 2, 0, 1, 2, 2, 2, -1, -1]], [[0.5, 1.0, 0.5]]),
        (4,
         [[-1, -1, 1,  1,   -1, -1,  3,  3],
          [0,  0,  -1, -1,  2,  2,   -1, -1]],
         [[1.0, 1.0, 1.0, 1.0]]
         ),
    ]
)
def test_detector_weight_segments(num_classes, segmented_detector, expected_weights):
    """
    Test of a method of DetectorProcessorClf for weighting classes segments.
    ...

    Parameters
    ----------
    num_classes
        Number of classes.
    segmented_detector
        Segmentized detector.
    expected_weights
        Expected weights for segments.
    """
    segmented_detector_tensor = torch.tensor(segmented_detector, dtype=torch.int32)
    processor = DetectorProcessorClf(
        num_classes,
        segmented_detector=segmented_detector_tensor,
    )
    assert torch.allclose(processor.segments_weights, torch.tensor(expected_weights))

    # probabilities are same for a mono-value detector image
    detector_data = torch.ones(size=segmented_detector_tensor.shape)
    assert torch.unique(processor.forward(detector_data)).shape[0] == 1
