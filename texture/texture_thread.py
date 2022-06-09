from texture import classify
from threading import Thread


class TextureClassifier:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.status = "waiting"

    def start(self):
        Thread(target=self.classify_textures, args=()).start()
        return self

    def classify_textures(self):
        classify.classify_textures()
