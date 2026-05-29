from nudenet import NudeDetector as _InstalledNudeDetector


class NudeDetector:
    def __init__(self, *args, **kwargs):
        self._detector = _InstalledNudeDetector(*args, **kwargs)

    def detect(self, image_path):
        return self._detector.detect(image_path)

    def detect_batch(self, image_paths, batch_size=4):
        return self._detector.detect_batch(image_paths, batch_size=batch_size)

    def censor(self, image_path, classes=None, output_path=None):
        classes = [] if classes is None else classes
        return self._detector.censor(image_path, classes=classes, output_path=output_path)
