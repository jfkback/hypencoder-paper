import threading
from queue import Queue
from typing import Iterable, List, TypeVar

Item = TypeVar("Item")


def batchify_slicing(
    items: Iterable[Item], batch_size: int
) -> Iterable[List[Item]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def batchify(items: Iterable[Item], batch_size: int) -> Iterable[List[Item]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch != []:
        yield batch


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, buffer_size: int = 1):
        threading.Thread.__init__(self)
        self.queue = Queue(buffer_size)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
